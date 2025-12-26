# Modal Integration Module

Composable, reliable Modal GPU integration with automatic retries and chunked uploads.

## Structure

```
integrations/modal/
├── __init__.py      # Package exports
├── config.py        # ModalConfig, ModalJobConfig
├── provider.py      # ModalProvider (implements GPUProvider)
└── utils.py         # with_retry, chunked_upload, helpers
```

## Features

### ✅ Automatic Retries
- `@with_retry` decorator for functions
- Exponential backoff (configurable)
- Catches connection errors

### ✅ Chunked Uploads
- Split large dirs by split (train/valid/test)
- Retry per chunk
- Upload time estimation

### ✅ Composable Jobs
- Follows `GPUProvider` interface
- Reusable across scripts
- Config-driven (ModalConfig)

## Usage

### Basic Example

```python
from integrations.modal import ModalConfig, ModalProvider

config = ModalConfig(
    gpu_type="A10G",
    timeout_seconds=3600,
    max_upload_retries=5,
    pip_packages=["torch", "transformers"]
)

provider = ModalProvider(config)

# Create app with local dir mounted
app = provider.create_app(
    "my-app",
    local_dirs={Path("./data"): "/data"}
)
```

### Deduplication Example

See `scripts/modal_dedup.py`:

```python
from integrations.modal import ModalConfig

config = ModalConfig(
    gpu_type="A10G",
    pip_packages=["torch", "transformers", "faiss-cpu"]
)

image = (
    modal.Image.debian_slim(python_version=config.python_version)
    .pip_install(*config.pip_packages)
    .add_local_dir(dataset_path, "/data")
)

@app.function(image=image, gpu=config.gpu_type)
def process():
    # Your GPU code here
    pass
```

## ModalConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `gpu_type` | "A10G" | T4, L4, A10G, A100 |
| `timeout_seconds` | 3600 | Max job duration |
| `max_upload_retries` | 3 | Retries per upload chunk |
| `pip_packages` | [] | Extra packages to install |
| `secret_names` | [] | Modal secrets to mount |

## Retry Utilities

### `@with_retry` Decorator

```python
from integrations.modal.utils import with_retry

@with_retry(max_retries=5, delay=2.0, backoff=2.0)
def upload_data():
    # Network operation that may fail
    pass
```

### `chunked_upload` Function

```python
from integrations.modal.utils import chunked_upload

chunked_upload(
    source_path=Path("./data"),
    upload_fn=lambda p: modal_volume.put(p, "/"),
    max_retries=5
)
```

## Next Steps

To run dedup with this integration:

```bash
uv run modal run scripts/modal_dedup.py --threshold 0.99
```

The script now:
- ✅ Uses ModalConfig for settings
- ✅ Has dataset size estimation
- ✅ Better error messages
- ✅ Saves results to dedup_results.json
