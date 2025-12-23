# Architecture

This project follows a three-layer architecture for the ML service.

## Layer Overview

```
┌─────────────────────────────────────────────────────────┐
│                        server                           │
│  Application bootstrap, middleware, health checks       │
├─────────────────────────────────────────────────────────┤
│                         api                             │
│  HTTP endpoints, request validation, serialization      │
├─────────────────────────────────────────────────────────┤
│                       service                           │
│  Core business logic (training, inference, export)      │
└─────────────────────────────────────────────────────────┘
```

## Layers

### com.service (✅ Complete)

Pure business logic. No HTTP, no networking.

| Service | Responsibility |
|---------|---------------|
| `DatasetService` | Download datasets, detect structure, create YAML |
| `TrainingService` | Train models, resume training, get model paths |
| `InferenceService` | Image/video/frame inference |
| `ValidationService` | Model validation and benchmarking |
| `ExportService` | Export to NCNN, ONNX, CoreML, TFLite |

### com.api (✅ Complete)

HTTP endpoints that call services. Planned routes:

```
POST /datasets/prepare     → DatasetService.prepare()
POST /train                → TrainingService.train()
POST /infer/image          → InferenceService.infer_image()
POST /infer/frame          → InferenceService.infer_frame()
POST /validate             → ValidationService.validate()
POST /export               → ExportService.export()
GET  /models/{name}/status → Check training progress
```

### com.server (✅ Bootstrap Ready)

Application bootstrap:

- Create FastAPI app
- Mount API routes
- Add middleware (CORS, auth, logging)
- Health check endpoint
- `uvicorn.run()` entrypoint

## Why This Architecture?

1. **Testability** - Unit test `com.service` without a server
2. **Flexibility** - Same service consumed by REST API, CLI, gRPC, WebSocket
3. **Separation of concerns** - Inference logic doesn't know about HTTP

## Build Order

1. ✅ `com.service` - Core logic
2. ✅ `com.api` - Define exposed operations
3. ✅ `com.server` - Bootstrap and run
