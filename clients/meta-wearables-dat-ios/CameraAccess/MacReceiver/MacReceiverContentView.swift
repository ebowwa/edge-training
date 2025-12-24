/*
 * Meta Wearables Mac Receiver
 * Created by humanwritten
 *
 * Main view displaying video stream from Meta AI glasses via iOS relay
 */

import SwiftUI
import AppKit

// MARK: - Main Content View

struct MacReceiverContentView: View {
    @StateObject private var viewModel = MacReceiverViewModel()

    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                colors: [Color(white: 0.1), Color.black],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            switch viewModel.viewState {
            case .waitingForConnection:
                WaitingForConnectionView(viewModel: viewModel)

            case .connected:
                ConnectedWaitingView(viewModel: viewModel)

            case .streaming:
                StreamingView(viewModel: viewModel)
            }
        }
        .frame(minWidth: 600, minHeight: 400)
        .onAppear {
            viewModel.startReceiving()
        }
        .onDisappear {
            viewModel.stopReceiving()
        }
    }
}

// MARK: - View States

enum MacReceiverViewState {
    case waitingForConnection
    case connected
    case streaming
}

// MARK: - Waiting For Connection View

struct WaitingForConnectionView: View {
    @ObservedObject var viewModel: MacReceiverViewModel

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            // Icon
            ZStack {
                Circle()
                    .stroke(Color.blue.opacity(0.3), lineWidth: 2)
                    .frame(width: 120, height: 120)

                Circle()
                    .stroke(Color.blue.opacity(0.2), lineWidth: 2)
                    .frame(width: 160, height: 160)

                Image(systemName: "iphone.and.arrow.right.inward")
                    .font(.system(size: 48))
                    .foregroundColor(.blue)
            }
            .padding(.bottom, 16)

            // Title
            Text("Connect Your iPhone")
                .font(.system(size: 28, weight: .semibold))
                .foregroundColor(.white)

            // Instructions
            VStack(alignment: .leading, spacing: 16) {
                InstructionRow(
                    number: "1",
                    icon: "app.badge",
                    text: "Open the CameraAccess app on your iPhone"
                )

                InstructionRow(
                    number: "2",
                    icon: "toggle.on",
                    text: "Enable \"Mac Relay\" in the app"
                )

                InstructionRow(
                    number: "3",
                    icon: "cable.connector",
                    text: "Connect via USB cable for lowest latency",
                    subtext: "Or ensure both devices are on the same WiFi network"
                )
            }
            .padding(.horizontal, 40)

            // Status indicator
            HStack(spacing: 12) {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .blue))
                    .scaleEffect(0.8)

                Text("Searching for iPhone...")
                    .font(.system(size: 14))
                    .foregroundColor(.gray)
            }
            .padding(.top, 24)

            Spacer()

            // Bottom hint
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                    .foregroundColor(.gray)
                Text("Both devices must be on the same network or connected via USB")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }
            .padding(.bottom, 24)
        }
    }
}

struct InstructionRow: View {
    let number: String
    let icon: String
    let text: String
    var subtext: String? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            ZStack {
                Circle()
                    .fill(Color.blue.opacity(0.2))
                    .frame(width: 32, height: 32)

                Text(number)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.blue)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Image(systemName: icon)
                        .foregroundColor(.white.opacity(0.7))
                        .frame(width: 20)

                    Text(text)
                        .font(.system(size: 15))
                        .foregroundColor(.white)
                }

                if let subtext = subtext {
                    Text(subtext)
                        .font(.system(size: 12))
                        .foregroundColor(.gray)
                        .padding(.leading, 28)
                }
            }
        }
    }
}

// MARK: - Connected Waiting View

struct ConnectedWaitingView: View {
    @ObservedObject var viewModel: MacReceiverViewModel

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            // Success animation
            ZStack {
                Circle()
                    .fill(Color.green.opacity(0.1))
                    .frame(width: 120, height: 120)

                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 64))
                    .foregroundColor(.green)
            }

            // Title
            Text("iPhone Connected!")
                .font(.system(size: 28, weight: .semibold))
                .foregroundColor(.white)

            // Source info
            if let source = viewModel.connectedSource {
                HStack(spacing: 8) {
                    Image(systemName: "iphone")
                        .foregroundColor(.green)
                    Text(source)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.green.opacity(0.15))
                .cornerRadius(20)
            }

            // Instructions
            VStack(spacing: 12) {
                Text("Waiting for video stream...")
                    .font(.system(size: 15))
                    .foregroundColor(.gray)

                Text("Tap \"Start streaming\" on your iPhone to begin")
                    .font(.system(size: 14))
                    .foregroundColor(.gray.opacity(0.8))
            }

            // Loading indicator
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: .green))
                .scaleEffect(1.2)
                .padding(.top, 16)

            Spacer()
        }
    }
}

// MARK: - Streaming View

struct StreamingView: View {
    @ObservedObject var viewModel: MacReceiverViewModel
    @State private var showControls = true

    var body: some View {
        ZStack {
            // Video frame
            if let frame = viewModel.currentFrame {
                Image(nsImage: frame)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            showControls.toggle()
                        }
                    }
            }

            // Controls overlay
            if showControls {
                VStack {
                    // Top bar - stats
                    HStack {
                        // Connection status
                        HStack(spacing: 8) {
                            Circle()
                                .fill(Color.green)
                                .frame(width: 10, height: 10)

                            if let source = viewModel.connectedSource {
                                Text(source)
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.white)
                            }
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(8)

                        Spacer()

                        // Stats
                        HStack(spacing: 20) {
                            StatBadge(icon: "speedometer", value: String(format: "%.1f fps", viewModel.frameRate))
                            StatBadge(icon: "arrow.down.circle", value: viewModel.bandwidthText)
                            StatBadge(icon: "clock", value: String(format: "%.0fms", viewModel.latencyMs))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(8)
                    }
                    .padding()

                    Spacer()

                    // Bottom bar - controls
                    HStack(spacing: 24) {
                        Spacer()

                        // Screenshot button
                        ControlButton(
                            icon: "camera.fill",
                            label: "Screenshot",
                            action: { viewModel.saveScreenshot() }
                        )

                        // Record button
                        ControlButton(
                            icon: viewModel.isRecording ? "stop.circle.fill" : "record.circle",
                            label: viewModel.isRecording ? "Stop" : "Record",
                            isActive: viewModel.isRecording,
                            action: { viewModel.toggleRecording() }
                        )

                        // Fullscreen button
                        ControlButton(
                            icon: "arrow.up.left.and.arrow.down.right",
                            label: "Fullscreen",
                            action: { toggleFullscreen() }
                        )

                        Spacer()
                    }
                    .padding(.vertical, 16)
                    .background(
                        LinearGradient(
                            colors: [Color.clear, Color.black.opacity(0.8)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                }
            }

            // Recording indicator
            if viewModel.isRecording {
                VStack {
                    HStack {
                        Spacer()
                        HStack(spacing: 8) {
                            Circle()
                                .fill(Color.red)
                                .frame(width: 12, height: 12)
                                .opacity(viewModel.recordingBlink ? 1 : 0.3)

                            Text("REC")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(.red)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(6)
                        .padding()
                    }
                    Spacer()
                }
            }
        }
    }

    private func toggleFullscreen() {
        if let window = NSApplication.shared.keyWindow {
            window.toggleFullScreen(nil)
        }
    }
}

struct StatBadge: View {
    let icon: String
    let value: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundColor(.gray)

            Text(value)
                .font(.system(size: 13, weight: .medium).monospacedDigit())
                .foregroundColor(.white)
        }
    }
}

struct ControlButton: View {
    let icon: String
    let label: String
    var isActive: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 24))
                    .foregroundColor(isActive ? .red : .white)

                Text(label)
                    .font(.system(size: 11))
                    .foregroundColor(.gray)
            }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - ViewModel

@MainActor
class MacReceiverViewModel: ObservableObject {
    @Published var currentFrame: NSImage?
    @Published var isReceiving = false
    @Published var isRecording = false
    @Published var recordingBlink = true
    @Published var frameRate: Double = 0
    @Published var bytesPerSecond: Int = 0
    @Published var latencyMs: Double = 0
    @Published var connectedSource: String?

    private let relay = VideoRelayService(mode: .receiver, deviceName: Host.current().localizedName ?? "Mac Receiver")
    private var lastFrameTime: Date?
    private var blinkTimer: Timer?

    var viewState: MacReceiverViewState {
        if isReceiving && currentFrame != nil {
            return .streaming
        } else if relay.connectionState.isConnected {
            return .connected
        } else {
            return .waitingForConnection
        }
    }

    var bandwidthText: String {
        if bytesPerSecond > 1_000_000 {
            return String(format: "%.1f MB/s", Double(bytesPerSecond) / 1_000_000)
        } else if bytesPerSecond > 1_000 {
            return String(format: "%.0f KB/s", Double(bytesPerSecond) / 1_000)
        }
        return "\(bytesPerSecond) B/s"
    }

    init() {
        setupObservers()
    }

    private func setupObservers() {
        // Observe relay changes
        Task { @MainActor in
            for await frame in relay.$receivedFrame.values {
                if let frame = frame {
                    self.currentFrame = frame
                    self.isReceiving = true

                    // Calculate latency (approximate - time between frames)
                    if let lastTime = self.lastFrameTime {
                        let delta = Date().timeIntervalSince(lastTime) * 1000
                        self.latencyMs = delta
                    }
                    self.lastFrameTime = Date()
                }
            }
        }

        Task { @MainActor in
            for await _ in relay.$frameRate.values {
                self.frameRate = relay.frameRate
            }
        }

        Task { @MainActor in
            for await _ in relay.$bytesPerSecond.values {
                self.bytesPerSecond = relay.bytesPerSecond
            }
        }

        Task { @MainActor in
            for await _ in relay.$connectedPeers.values {
                self.connectedSource = relay.connectedPeers.first?.displayName
                self.objectWillChange.send()
            }
        }

        Task { @MainActor in
            for await _ in relay.$connectionState.values {
                self.objectWillChange.send()
            }
        }
    }

    func startReceiving() {
        relay.start()
    }

    func stopReceiving() {
        relay.stop()
        blinkTimer?.invalidate()
    }

    func toggleRecording() {
        isRecording.toggle()

        if isRecording {
            // Start blink timer
            blinkTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.recordingBlink.toggle()
                }
            }
            print("[Receiver] Started recording")
        } else {
            blinkTimer?.invalidate()
            blinkTimer = nil
            print("[Receiver] Stopped recording")
        }
    }

    func saveScreenshot() {
        guard let image = currentFrame else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png, .jpeg]
        panel.nameFieldStringValue = "glasses-screenshot-\(Int(Date().timeIntervalSince1970)).png"

        if panel.runModal() == .OK, let url = panel.url {
            if let tiffData = image.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiffData),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
                print("[Receiver] Saved screenshot to \(url.path)")
            }
        }
    }
}

#Preview {
    MacReceiverContentView()
}
