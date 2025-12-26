# Coding Standards

Rules and conventions for contributing to this repository.

## Architecture Rules

**Three-layer separation (strict):**

```
server/       → Bootstrap, middleware, health checks only
api/          → HTTP endpoints, Pydantic schemas, serialization
service/      → Pure business logic (no HTTP, no networking)
pipeline/     → Composable pipelines for real-time inference
integrations/ → External provider integrations (Kaggle, RunPod, etc.)
```

- Service layer must be **HTTP-agnostic** — pure functions/classes
- API layer only validates requests and calls services
- Server layer only bootstraps FastAPI/Uvicorn

---

## Code Style

- **PEP 8** compliant
- Meaningful variable/function names
- **Docstrings** on functions
- Lines under **80 characters** (where reasonable)
- **No `__init__.py`** unless package-level initialization is needed (Python 3.3+ namespace packages)

---

## Project Conventions

| Pattern | Example |
|---------|---------|
| Configs as dataclasses | `TrainingConfig(epochs=60, imgsz=640)` |
| Results as dataclasses | `TrainingResult(best_model_path=...)` |
| Static service methods | `TrainingService.train()` (not instantiated) |
| Module aliases in routes | `from service import X as x_alias` |
| Tests alongside code | `service/tests/`, `api/tests/` |

---

## Directory Structure

```
service/
├── yolo/              # YOLO-specific services
│   └── validation.py
├── yolo_training.py   # YOLO training
├── inference_service.py
└── export_service.py

integrations/
└── kaggle/            # External dataset providers
    └── dataset.py

api/                   # HTTP layer
├── routes.py
├── schemas.py         # Pydantic models
└── tests/

pipeline/              # Composable inference pipelines
```

---

## Dependency Management

- **`pyproject.toml`** is the single source of truth
- Use `uv` over pip (preferred)
- Python `>=3.11` required
