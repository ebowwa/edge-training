/*
 * HTTPInferenceProcessor.swift
 * MacReceiver
 *
 * HTTP-based frame processor for remote ML inference APIs.
 *
 * ## Configuration
 *
 * Set `baseURL` and `endpoint` to match your inference server.
 * Default expects JSON response with:
 * ```json
 * {
 *   "detections": [
 *     {
 *       "class_name": "person",
 *       "confidence": 0.95,
 *       "bbox_normalized": [cx, cy, w, h]  // center-based
 *     }
 *   ],
 *   "inference_time_ms": 45.2
 * }
 * ```
 */

import AppKit
import SwiftUI

// MARK: - HTTP Inference Processor

class HTTPInferenceProcessor: FrameProcessor {
    
    // MARK: - FrameProcessor Protocol
    
    var name: String { "HTTP Inference" }
    var isEnabled = false
    
    // MARK: - Configuration
    
    /// Base URL of the inference server
    var baseURL: String = "http://localhost:8000"
    
    /// Endpoint path for inference
    var endpoint: String = "/api/v1/infer"
    
    /// Color for detection overlays
    var overlayColor: Color = .green
    
    /// JPEG compression quality (0-1)
    var compressionQuality: Double = 0.8
    
    // MARK: - Processing
    
    func process(_ frame: NSImage) async throws -> ProcessorResult {
        let startTime = Date()
        
        guard let url = URL(string: "\(baseURL)\(endpoint)") else {
            throw HTTPInferenceError.invalidURL
        }
        
        // Convert frame to JPEG
        guard let jpegData = frame.jpegData(compressionQuality: compressionQuality) else {
            throw HTTPInferenceError.imageConversionFailed
        }
        
        // Build multipart request
        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(jpegData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        // Send request
        let (data, _) = try await URLSession.shared.upload(for: request, from: body)
        
        // Parse response
        let response = try JSONDecoder().decode(InferenceResponse.self, from: data)
        
        let processingTime = Date().timeIntervalSince(startTime) * 1000
        
        // Convert to overlays
        let overlays = response.detections.map { detection -> DetectionOverlay in
            // API returns center-based [cx, cy, w, h]
            let cx = CGFloat(detection.bbox_normalized[0])
            let cy = CGFloat(detection.bbox_normalized[1])
            let w = CGFloat(detection.bbox_normalized[2])
            let h = CGFloat(detection.bbox_normalized[3])
            
            // Convert to origin-based rect
            let rect = CGRect(
                x: cx - w/2,
                y: cy - h/2,
                width: w,
                height: h
            )
            
            return DetectionOverlay(
                label: detection.class_name,
                confidence: detection.confidence,
                normalizedRect: rect,
                color: overlayColor
            )
        }
        
        return ProcessorResult(
            overlays: overlays,
            processingTimeMs: processingTime,
            metadata: ["server_time_ms": response.inference_time_ms]
        )
    }
}

// MARK: - Response Types

private struct InferenceResponse: Codable {
    let detections: [Detection]
    let inference_time_ms: Float
}

private struct Detection: Codable {
    let class_id: Int?
    let class_name: String
    let confidence: Float
    let bbox_normalized: [Float]
}

// MARK: - Errors

enum HTTPInferenceError: LocalizedError {
    case invalidURL
    case imageConversionFailed
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid inference server URL"
        case .imageConversionFailed:
            return "Failed to convert image to JPEG"
        }
    }
}

// MARK: - NSImage Extension

private extension NSImage {
    func jpegData(compressionQuality: Double) -> Data? {
        guard let tiffData = tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else {
            return nil
        }
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: compressionQuality])
    }
}
