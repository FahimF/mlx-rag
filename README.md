# MLX-GUI

```
███╗   ███╗██╗     ██╗  ██╗      ██████╗ ██╗   ██╗██╗
████╗ ████║██║     ╚██╗██╔╝     ██╔════╝ ██║   ██║██║
██╔████╔██║██║      ╚███╔╝█████╗██║  ███╗██║   ██║██║
██║╚██╔╝██║██║      ██╔██╗╚════╝██║   ██║██║   ██║██║
██║ ╚═╝ ██║███████╗██╔╝ ██╗     ╚██████╔╝╚██████╔╝██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝      ╚═════╝  ╚═════╝ ╚═╝
```

**A lightweight RESTful wrapper around Apple's MLX engine for dynamically loading and serving MLX-compatible models.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Required-orange.svg)](https://support.apple.com/en-us/HT211814)
[![MLX Compatible](https://img.shields.io/badge/MLX-Compatible-green.svg)](https://github.com/ml-explore/mlx)

## 🚀 Features

- **🧠 MLX Engine Integration** - Native Apple Silicon acceleration via MLX
- **🔄 Dynamic Model Loading** - Load/unload models on-demand with memory management
- **🌐 REST API Server** - Complete API for model management and inference
- **🎨 Beautiful Admin Interface** - Modern web GUI for model management
- **📊 System Monitoring** - Real-time memory usage and system status
- **🔍 HuggingFace Integration** - Discover and install MLX-compatible models
- **🍎 macOS System Tray** - Native menu bar integration
- **⚡ OpenAI Compatibility** - Drop-in replacement for OpenAI API
- **📱 Standalone App** - Packaged macOS app bundle (no Python required)

## 📋 Requirements

- **macOS** (Apple Silicon M1/M2/M3/M4 required)
- **Python 3.11+** (for development)
- **8GB+ RAM** (16GB+ recommended for larger models)

## 🏃‍♂️ Quick Start

### Option 1: Download Standalone App
1. Download the latest `.app` from [Releases](https://github.com/RamboRogers/mlx-gui/releases)
2. Drag to `/Applications` 
3. Launch - no Python installation required!

### Option 2: Install from Source
```bash
# Clone the repository
git clone https://github.com/RamboRogers/mlx-gui.git
cd mlx-gui

# Install dependencies
pip install -e ".[app]"

# Launch with system tray
mlx-gui tray

# Or launch server only
mlx-gui start --port 8000
```

## 🎮 Usage

### System Tray (Recommended)
Launch the app and look for **MLX** in your menu bar:
- **Open Admin Interface** - Web GUI for model management
- **System Status** - Real-time monitoring
- **Unload All Models** - Free up memory
- **Network Settings** - Configure binding options

### Web Admin Interface
Navigate to `http://localhost:8000/admin` for:
- 🔍 **Discover Tab** - Browse and install MLX models from HuggingFace
- 🧠 **Models Tab** - Manage installed models (load/unload/remove)
- 📊 **Monitor Tab** - System statistics and performance
- ⚙️ **Settings Tab** - Configure server and model options

### API Usage

#### OpenAI-Compatible Chat
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-6bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

#### Install Models
```bash
curl -X POST http://localhost:8000/v1/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "name": "qwen-7b-4bit"
  }'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  System Tray    │    │   Web Admin GUI  │    │   REST API      │
│  (macOS)        │◄──►│  (localhost:8000)│◄──►│  (/v1/*)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────┐              │
                       │ Model Manager   │◄─────────────┘
                       │ (Queue/Memory)  │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  MLX Engine     │
                       │ (Apple Silicon) │
                       └─────────────────┘
```

## 📚 API Documentation

Full API documentation is available at `/v1/docs` when the server is running, or see [API.md](API.md) for complete endpoint reference.

### Key Endpoints
- `GET /v1/models` - List installed models  
- `POST /v1/models/install` - Install from HuggingFace
- `POST /v1/models/{name}/load` - Load model into memory
- `POST /v1/chat/completions` - OpenAI-compatible chat
- `GET /v1/discover/models` - Search HuggingFace for MLX models
- `GET /v1/system/status` - System and memory status

## 🛠️ Development

### Setup Development Environment
```bash
git clone https://github.com/RamboRogers/mlx-gui.git
cd mlx-gui

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Start development server
mlx-gui start --reload
```

### Build Standalone App
```bash
# Install build dependencies
pip install rumps pyinstaller

# Build macOS app bundle
./build_app.sh

# Result: dist/MLX-GUI.app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚖️ License

<p>
MLX-GUI is licensed under the GNU General Public License v3.0 (GPLv3).<br>
<em>Free Software</em>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)

### Connect With Me 🤝

[![GitHub](https://img.shields.io/badge/GitHub-RamboRogers-181717?style=for-the-badge&logo=github)](https://github.com/RamboRogers)
[![Twitter](https://img.shields.io/badge/Twitter-@rogerscissp-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/rogerscissp)
[![Website](https://img.shields.io/badge/Web-matthewrogers.org-00ADD8?style=for-the-badge&logo=google-chrome)](https://matthewrogers.org)

## 🙏 Acknowledgments

- [Apple MLX Team](https://github.com/ml-explore/mlx) - For the incredible MLX framework
- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) - MLX language model implementations
- [HuggingFace](https://huggingface.co) - For the model hub and transformers library

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RamboRogers/mlx-gui&type=Timeline)](https://star-history.com/#RamboRogers/mlx-gui&Timeline)

---

<div align="center">
  <strong>Made with ❤️ for the Apple Silicon community</strong>
</div>