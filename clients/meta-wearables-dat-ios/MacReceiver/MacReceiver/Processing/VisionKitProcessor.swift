/*
 * VisionKitProcessor.swift
 * MacReceiver
 *
 * On-device frame processor using Apple's Vision framework.
 *
 * ## Modes
 *
 * - `.textRecognition`: OCR for detecting and reading text
 * - `.faceDetection`: Detect faces with landmarks
 * - `.objectDetection`: Classify objects (requires CoreML model)
 *
 * ## Performance
 *
 * Runs entirely on-device using Neural Engine when available.
 * No network latency, but limited to Apple's built-in models.
 */

import AppKit
import SwiftUI
import Vision

// MARK: - Vision Kit Processor

class VisionKitProcessor: FrameProcessor {
    
    // MARK: - FrameProcessor Protocol
    
    var name: String { "Vision Kit" }
    var isEnabled = false
    
    // MARK: - Configuration
    
    enum Mode {
        case textRecognition
        case faceDetection
        case rectangleDetection
    }
    
    var mode: Mode = .textRecognition
    
    /// Colors for different detection types
    var textColor: Color = .yellow
    var faceColor: Color = .blue
    var rectangleColor: Color = .purple
    
    // MARK: - Processing
    
    func process(_ frame: NSImage) async throws -> ProcessorResult {
        let startTime = Date()
        
        guard let cgImage = frame.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw VisionError.imageConversionFailed
        }
        
        let overlays: [DetectionOverlay]
        
        switch mode {
        case .textRecognition:
            overlays = try await detectText(in: cgImage)
        case .faceDetection:
            overlays = try await detectFaces(in: cgImage)
        case .rectangleDetection:
            overlays = try await detectRectangles(in: cgImage)
        }
        
        let processingTime = Date().timeIntervalSince(startTime) * 1000
        
        return ProcessorResult(
            overlays: overlays,
            processingTimeMs: processingTime,
            metadata: ["mode": String(describing: mode)]
        )
    }
    
    // MARK: - Text Recognition
    
    private func detectText(in cgImage: CGImage) async throws -> [DetectionOverlay] {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                let observations = request.results as? [VNRecognizedTextObservation] ?? []
                let overlays = observations.compactMap { observation -> DetectionOverlay? in
                    guard let candidate = observation.topCandidates(1).first else { return nil }
                    
                    // Vision uses bottom-left origin, flip Y
                    let box = observation.boundingBox
                    let rect = CGRect(
                        x: box.minX,
                        y: 1 - box.maxY,
                        width: box.width,
                        height: box.height
                    )
                    
                    return DetectionOverlay(
                        label: candidate.string,
                        confidence: candidate.confidence,
                        normalizedRect: rect,
                        color: self.textColor
                    )
                }
                
                continuation.resume(returning: overlays)
            }
            
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            
            let handler = VNImageRequestHandler(cgImage: cgImage)
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    // MARK: - Face Detection
    
    private func detectFaces(in cgImage: CGImage) async throws -> [DetectionOverlay] {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNDetectFaceRectanglesRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                let observations = request.results as? [VNFaceObservation] ?? []
                let overlays = observations.map { observation -> DetectionOverlay in
                    let box = observation.boundingBox
                    let rect = CGRect(
                        x: box.minX,
                        y: 1 - box.maxY,
                        width: box.width,
                        height: box.height
                    )
                    
                    return DetectionOverlay(
                        label: "Face",
                        confidence: observation.confidence,
                        normalizedRect: rect,
                        color: self.faceColor
                    )
                }
                
                continuation.resume(returning: overlays)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage)
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    // MARK: - Rectangle Detection
    
    private func detectRectangles(in cgImage: CGImage) async throws -> [DetectionOverlay] {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNDetectRectanglesRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                let observations = request.results as? [VNRectangleObservation] ?? []
                let overlays = observations.map { observation -> DetectionOverlay in
                    let box = observation.boundingBox
                    let rect = CGRect(
                        x: box.minX,
                        y: 1 - box.maxY,
                        width: box.width,
                        height: box.height
                    )
                    
                    return DetectionOverlay(
                        label: "Rectangle",
                        confidence: observation.confidence,
                        normalizedRect: rect,
                        color: self.rectangleColor
                    )
                }
                
                continuation.resume(returning: overlays)
            }
            
            request.maximumObservations = 10
            request.minimumConfidence = 0.5
            
            let handler = VNImageRequestHandler(cgImage: cgImage)
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

// MARK: - Errors

enum VisionError: LocalizedError {
    case imageConversionFailed
    
    var errorDescription: String? {
        switch self {
        case .imageConversionFailed:
            return "Failed to convert image for Vision processing"
        }
    }
}
