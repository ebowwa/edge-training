/*
 * FrameProcessor.swift
 * MacReceiver
 *
 * Composable frame processing architecture for ML/VLM/AI inference.
 *
 * ## Architecture
 *
 * This module provides a protocol-based system for processing video frames.
 * Processors can be swapped, chained, or run in parallel.
 *
 * ## Usage
 *
 * 1. Conform to `FrameProcessor` protocol
 * 2. Implement `process(_:)` to return `ProcessorResult`
 * 3. Register processor with the ViewModel
 *
 * ## Example
 *
 * ```swift
 * class MyProcessor: FrameProcessor {
 *     var name: String { "My Processor" }
 *     var isEnabled = true
 *
 *     func process(_ frame: NSImage) async throws -> ProcessorResult {
 *         // Your inference logic here
 *         return ProcessorResult(overlays: [], processingTimeMs: 0)
 *     }
 * }
 * ```
 */

import SwiftUI
import AppKit

// MARK: - Detection Overlay

/// Visual overlay for displaying detection results on video frames.
///
/// Uses normalized coordinates (0-1) for position and size,
/// making it resolution-independent.
struct DetectionOverlay: Identifiable {
    let id = UUID()
    
    /// Label text (e.g., "person", "car", detected text)
    let label: String
    
    /// Confidence score 0-1
    let confidence: Float
    
    /// Bounding box in normalized coordinates (origin top-left, 0-1 range)
    /// - x, y: top-left corner
    /// - width, height: dimensions
    let normalizedRect: CGRect
    
    /// Color for the bounding box and label
    let color: Color
    
    /// Optional additional metadata
    var metadata: [String: Any] = [:]
}

// MARK: - Processor Result

/// Result returned by any frame processor.
struct ProcessorResult {
    /// Detection overlays to render on the video frame
    let overlays: [DetectionOverlay]
    
    /// Time taken for processing in milliseconds
    let processingTimeMs: Double
    
    /// Flexible extension point for processor-specific data
    var metadata: [String: Any] = [:]
    
    /// Empty result for convenience
    static let empty = ProcessorResult(overlays: [], processingTimeMs: 0)
}

// MARK: - Frame Processor Protocol

/// Protocol for composable frame processors.
///
/// Implement this protocol to create custom processors for:
/// - Remote ML inference (HTTP APIs)
/// - On-device Vision framework processing
/// - Custom CoreML models
/// - VLM/LLM integrations
///
/// ## Threading
///
/// The `process(_:)` method is called on a background thread.
/// Results are automatically marshaled to the main actor for UI updates.
///
/// ## Throttling
///
/// The ViewModel handles throttling based on `targetInferenceFPS`.
/// Processors don't need to implement their own throttling.
protocol FrameProcessor: AnyObject {
    /// Human-readable name for UI display
    var name: String { get }
    
    /// Whether this processor is currently enabled
    var isEnabled: Bool { get set }
    
    /// Process a single video frame
    /// - Parameter frame: The video frame to process
    /// - Returns: Processing result with overlays and timing
    /// - Throws: Any error during processing
    func process(_ frame: NSImage) async throws -> ProcessorResult
}

// MARK: - Overlay View

/// SwiftUI view for rendering detection overlays on video frames.
struct ProcessorOverlayView: View {
    let results: [ProcessorResult]
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(allOverlays) { overlay in
                makeBox(for: overlay, in: geometry.size)
            }
        }
        .allowsHitTesting(false)
    }
    
    private var allOverlays: [DetectionOverlay] {
        results.flatMap { $0.overlays }
    }
    
    private func makeBox(for overlay: DetectionOverlay, in viewSize: CGSize) -> some View {
        let rect = overlay.normalizedRect
        
        let width = rect.width * viewSize.width
        let height = rect.height * viewSize.height
        let x = rect.minX * viewSize.width
        let y = rect.minY * viewSize.height
        
        return ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(overlay.color, lineWidth: 2)
            
            Text("\(overlay.label) \(Int(overlay.confidence * 100))%")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.black)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(overlay.color)
                .cornerRadius(2)
                .offset(y: -20)
        }
        .frame(width: max(1, width), height: max(1, height))
        .position(x: x + width/2, y: y + height/2)
    }
}
