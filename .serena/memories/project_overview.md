# MLX-RAG Project Overview

MLX-RAG is an AI inference server for Apple Silicon that provides:

## Purpose
- OpenRouter-style v1 API interface for MLX with Ollama-like model management
- Lightweight inference server for Apple's MLX engine with GUI
- Auto-queuing, on-demand model loading, and multi-user serving capabilities
- Native Apple Silicon acceleration via MLX framework

## Tech Stack
- **Backend**: Python 3.11+ with FastAPI
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS
- **AI Framework**: Apple MLX
- **Database**: SQLite with Python models
- **Package Management**: UV (preferred, 10-100x faster than pip)
- **Build Tool**: Custom shell scripts for macOS app bundling

## Key Features
- Text generation, vision models, audio transcription, embeddings
- Real-time model status monitoring with download progress
- Web-based admin interface
- OpenAI-compatible API endpoints
- RAG (Retrieval-Augmented Generation) system supporting 17+ programming languages
- macOS system tray integration

## Architecture
- FastAPI server with CORS middleware
- Model manager with queue system
- System monitor for memory/performance tracking
- Template-based web UI (single admin.html file)
- MLX integration for Apple Silicon optimization