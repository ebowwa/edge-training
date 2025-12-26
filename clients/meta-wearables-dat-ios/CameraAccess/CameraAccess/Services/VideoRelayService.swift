/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Modified by humanwritten to add Mac relay functionality via MultipeerConnectivity
 */

//
// VideoRelayService.swift
//
// MultipeerConnectivity service for streaming video frames from iOS to Mac.
// Works over both WiFi and USB for lowest latency.
//

import Foundation
import MultipeerConnectivity
import SwiftUI

#if os(iOS)
import UIKit
typealias PlatformImage = UIImage
#elseif os(macOS)
import AppKit
typealias PlatformImage = NSImage
#endif

// MARK: - Protocol for receiving frames (used by Mac app)

protocol VideoRelayDelegate: AnyObject {
    func videoRelay(_ relay: VideoRelayService, didReceiveFrame frameData: Data, timestamp: TimeInterval)
    func videoRelay(_ relay: VideoRelayService, didChangeState state: VideoRelayService.ConnectionState)
}

// MARK: - VideoRelayService

@MainActor
class VideoRelayService: NSObject, ObservableObject {

    // MARK: - Types

    enum Mode {
        case broadcaster  // iOS: sends frames
        case receiver     // Mac: receives frames
    }

    enum ConnectionState: Equatable {
        case disconnected
        case connecting
        case connected(peerCount: Int)

        var isConnected: Bool {
            if case .connected = self { return true }
            return false
        }
    }

    // MARK: - Published Properties

    @Published var connectionState: ConnectionState = .disconnected
    @Published var connectedPeers: [MCPeerID] = []
    @Published var receivedFrame: PlatformImage?
    @Published var frameRate: Double = 0
    @Published var bytesPerSecond: Int = 0

    // MARK: - Properties

    private let serviceType = "meta-glasses"  // Must be 1-15 chars, lowercase + hyphens
    private let myPeerID: MCPeerID
    // nonisolated(unsafe) because MultipeerConnectivity delegates are called on arbitrary threads
    // but we manage the session lifecycle carefully from MainActor
    nonisolated(unsafe) private var session: MCSession?
    private var advertiser: MCNearbyServiceAdvertiser?
    private var browser: MCNearbyServiceBrowser?
    private let mode: Mode

    weak var delegate: VideoRelayDelegate?

    // Stats tracking
    private var frameCount = 0
    private var byteCount = 0
    private var lastStatsUpdate = Date()

    // Frame throttling for performance
    private var lastFrameSentTime: Date = .distantPast
    private let minFrameInterval: TimeInterval = 1.0 / 15.0  // Max 15 FPS for relay
    private var isProcessingFrame = false
    private let processingQueue = DispatchQueue(label: "com.relay.compression", qos: .userInitiated)

    // MARK: - Initialization

    init(mode: Mode, deviceName: String? = nil) {
        self.mode = mode

        #if os(iOS)
        let name = deviceName ?? UIDevice.current.name
        #else
        let name = deviceName ?? Host.current().localizedName ?? "Mac"
        #endif

        self.myPeerID = MCPeerID(displayName: name)
        super.init()
    }

    // MARK: - Public Methods

    func start() {
        stop()  // Clean up any existing session

        session = MCSession(
            peer: myPeerID,
            securityIdentity: nil,
            encryptionPreference: .none  // No encryption for lowest latency
        )
        session?.delegate = self

        switch mode {
        case .broadcaster:
            // iOS: Advertise availability to Mac receivers
            advertiser = MCNearbyServiceAdvertiser(
                peer: myPeerID,
                discoveryInfo: ["type": "glasses-stream"],
                serviceType: serviceType
            )
            advertiser?.delegate = self
            advertiser?.startAdvertisingPeer()
            print("[VideoRelay] Started advertising as broadcaster")

        case .receiver:
            // Mac: Browse for iOS broadcasters
            browser = MCNearbyServiceBrowser(peer: myPeerID, serviceType: serviceType)
            browser?.delegate = self
            browser?.startBrowsingForPeers()
            print("[VideoRelay] Started browsing for broadcasters")
        }

        Task { @MainActor in
            connectionState = .connecting
        }
    }

    func stop() {
        advertiser?.stopAdvertisingPeer()
        advertiser = nil

        browser?.stopBrowsingForPeers()
        browser = nil

        session?.disconnect()
        session = nil

        Task { @MainActor in
            connectionState = .disconnected
            connectedPeers = []
        }

        print("[VideoRelay] Stopped")
    }

    // MARK: - Frame Broadcasting (iOS side)

    /// Send a video frame to all connected Mac receivers
    /// - Parameters:
    ///   - imageData: JPEG compressed frame data
    ///   - timestamp: Frame timestamp for synchronization
    func broadcastFrame(_ imageData: Data, timestamp: TimeInterval = Date().timeIntervalSince1970) {
        guard mode == .broadcaster,
              let session = session,
              !session.connectedPeers.isEmpty else {
            return
        }

        // Create a simple packet with timestamp + data
        var packet = Data()
        var ts = timestamp
        packet.append(Data(bytes: &ts, count: MemoryLayout<TimeInterval>.size))
        packet.append(imageData)

        do {
            // Use unreliable for lowest latency (UDP-like behavior)
            try session.send(packet, toPeers: session.connectedPeers, with: .unreliable)

            // Update stats
            frameCount += 1
            byteCount += packet.count
            updateStats()
        } catch {
            print("[VideoRelay] Send error: \(error)")
        }
    }

    /// Convenience method to send a UIImage (iOS only)
    /// Uses throttling and background processing to avoid UI stuttering
    #if os(iOS)
    func broadcastImage(_ image: UIImage, compressionQuality: CGFloat = 0.5) {
        // Throttle: Skip if we sent a frame too recently
        let now = Date()
        guard now.timeIntervalSince(lastFrameSentTime) >= minFrameInterval else {
            return // Drop frame - too soon
        }

        // Skip if still processing previous frame
        guard !isProcessingFrame else {
            return // Drop frame - busy
        }

        isProcessingFrame = true
        lastFrameSentTime = now

        // Move compression to background thread
        processingQueue.async { [weak self] in
            guard let self = self else { return }

            // Resize image for better performance (max 480p for relay)
            let maxDimension: CGFloat = 640
            let resizedImage: UIImage
            if image.size.width > maxDimension || image.size.height > maxDimension {
                let scale = min(maxDimension / image.size.width, maxDimension / image.size.height)
                let newSize = CGSize(width: image.size.width * scale, height: image.size.height * scale)
                UIGraphicsBeginImageContextWithOptions(newSize, true, 1.0)
                image.draw(in: CGRect(origin: .zero, size: newSize))
                resizedImage = UIGraphicsGetImageFromCurrentImageContext() ?? image
                UIGraphicsEndImageContext()
            } else {
                resizedImage = image
            }

            // Compress to JPEG
            guard let jpegData = resizedImage.jpegData(compressionQuality: compressionQuality) else {
                self.isProcessingFrame = false
                return
            }

            // Send on background thread (MCSession.send is thread-safe)
            self.broadcastFrame(jpegData)
            self.isProcessingFrame = false
        }
    }
    #endif

    // MARK: - Private Methods

    private func updateStats() {
        let now = Date()
        let elapsed = now.timeIntervalSince(lastStatsUpdate)

        if elapsed >= 1.0 {
            Task { @MainActor in
                frameRate = Double(frameCount) / elapsed
                bytesPerSecond = Int(Double(byteCount) / elapsed)
            }
            frameCount = 0
            byteCount = 0
            lastStatsUpdate = now
        }
    }

    private func handleReceivedData(_ data: Data, from peer: MCPeerID) {
        // Extract timestamp and image data
        guard data.count > MemoryLayout<TimeInterval>.size else { return }

        let timestampData = data.prefix(MemoryLayout<TimeInterval>.size)
        let imageData = data.dropFirst(MemoryLayout<TimeInterval>.size)

        var timestamp: TimeInterval = 0
        _ = withUnsafeMutableBytes(of: &timestamp) { timestampData.copyBytes(to: $0) }

        // Update stats
        frameCount += 1
        byteCount += data.count
        updateStats()

        // Notify delegate
        delegate?.videoRelay(self, didReceiveFrame: Data(imageData), timestamp: timestamp)

        // Update published property for SwiftUI
        #if os(iOS)
        if let image = UIImage(data: Data(imageData)) {
            Task { @MainActor in
                receivedFrame = image
            }
        }
        #elseif os(macOS)
        if let image = NSImage(data: Data(imageData)) {
            Task { @MainActor in
                receivedFrame = image
            }
        }
        #endif
    }
}

// MARK: - MCSessionDelegate

extension VideoRelayService: MCSessionDelegate {
    nonisolated func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        print("[VideoRelay] Peer \(peerID.displayName) state: \(state.rawValue)")

        Task { @MainActor in
            connectedPeers = session.connectedPeers

            switch state {
            case .connected:
                connectionState = .connected(peerCount: session.connectedPeers.count)
            case .connecting:
                connectionState = .connecting
            case .notConnected:
                if session.connectedPeers.isEmpty {
                    connectionState = .disconnected
                } else {
                    connectionState = .connected(peerCount: session.connectedPeers.count)
                }
            @unknown default:
                break
            }

            delegate?.videoRelay(self, didChangeState: connectionState)
        }
    }

    nonisolated func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        Task { @MainActor in
            handleReceivedData(data, from: peerID)
        }
    }

    nonisolated func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {
        // Not used - we send discrete frames
    }

    nonisolated func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {
        // Not used
    }

    nonisolated func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {
        // Not used
    }
}

// MARK: - MCNearbyServiceAdvertiserDelegate (iOS broadcaster)

extension VideoRelayService: MCNearbyServiceAdvertiserDelegate {
    nonisolated func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer peerID: MCPeerID, withContext context: Data?, invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        print("[VideoRelay] Received invitation from \(peerID.displayName)")
        // Auto-accept connections from Mac receivers
        invitationHandler(true, session)
    }

    nonisolated func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didNotStartAdvertisingPeer error: Error) {
        print("[VideoRelay] Failed to advertise: \(error)")
    }
}

// MARK: - MCNearbyServiceBrowserDelegate (Mac receiver)

extension VideoRelayService: MCNearbyServiceBrowserDelegate {
    nonisolated func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        print("[VideoRelay] Found peer: \(peerID.displayName)")
        // Auto-invite any discovered glasses broadcasters
        if info?["type"] == "glasses-stream" {
            browser.invitePeer(peerID, to: session!, withContext: nil, timeout: 10)
        }
    }

    nonisolated func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {
        print("[VideoRelay] Lost peer: \(peerID.displayName)")
    }

    nonisolated func browser(_ browser: MCNearbyServiceBrowser, didNotStartBrowsingForPeers error: Error) {
        print("[VideoRelay] Failed to browse: \(error)")
    }
}
