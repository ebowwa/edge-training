# MacReceiver Processing Module

Composable frame processing architecture for ML/VLM/AI inference on video streams from Meta glasses.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  VideoRelay     │───▶│ ReceiverViewModel│───▶│ FrameProcessor  │
│  (frames)       │    │ (orchestrator)   │    │ (protocol)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              ▼                        ▼                        ▼
                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │ HTTPInference   │    │ VisionKit       │    │ Your Custom     │
                    │ Processor       │    │ Processor       │    │ Processor       │
                    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Built-in Processors

### HTTPInferenceProcessor
Remote ML inference via HTTP API. Configure `baseURL` and `endpoint` to point at any inference server.

```swift
let processor = HTTPInferenceProcessor()
processor.baseURL = "http://192.168.1.100:8000"
processor.endpoint = "/api/v1/infer"
```

### VisionKitProcessor
On-device processing using Apple Vision framework:
- **Text Recognition**: OCR for reading text
- **Face Detection**: Detect faces in frame
- **Rectangle Detection**: Find rectangular objects

```swift
let processor = VisionKitProcessor()
processor.mode = .textRecognition
```

## Adding a Custom Processor

1. Create a new file in `Processing/`
2. Implement `FrameProcessor` protocol
3. Add to `ReceiverViewModel.processors` array

```swift
class MyProcessor: FrameProcessor {
    var name: String { "My Processor" }
    var isEnabled = false
    
    func process(_ frame: NSImage) async throws -> ProcessorResult {
        // Your inference logic
        let overlays: [DetectionOverlay] = []
        return ProcessorResult(overlays: overlays, processingTimeMs: 0)
    }
}
```

## API Response Format (HTTP)

The `HTTPInferenceProcessor` expects this JSON format:

```json
{
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.95,
      "bbox_normalized": [0.5, 0.5, 0.2, 0.4]
    }
  ],
  "inference_time_ms": 45.2
}
```

Where `bbox_normalized` is `[center_x, center_y, width, height]` in 0-1 range.
