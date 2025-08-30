# Development Commands for MLX-RAG

## Setup & Installation
```bash
# Clone and install with uv (recommended)
git clone https://github.com/FahimFarook/mlx-rag.git
cd mlx-rag
uv sync --extra dev --extra audio --extra vision

# Alternative with pip
pip install mlx-rag
```

## Running the Application
```bash
# Development server with reload
uv run mlx-rag start --log-level debug --reload

# System tray mode
uv run mlx-rag tray

# Server only mode
uv run mlx-rag start --port 8000
```

## Testing
```bash
# Run all tests
uv run pytest

# Quick smoke tests
uv run pytest tests/test_audio.py::test_audio_transcription -q
uv run pytest tests/test_embeddings_endpoint.py -q
```

## Building
```bash
# Build standalone macOS app
uv run ./scripts/build_app.sh
# Result: dist/MLX-RAG.app

# Skip UV sync during build (if already synced)
SKIP_UV_SYNC=1 uv run ./scripts/build_app.sh
```

## Common Development Tasks
```bash
# Install language parsers for RAG
uv run python install_language_parsers.py

# Install specific language support
uv run python install_language_parsers.py --install javascript typescript

# Install all RAG language support
uv sync --extra rag-full
```