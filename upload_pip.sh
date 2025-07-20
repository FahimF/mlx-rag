#!/bin/bash

# upload_pip.sh - Automated PyPI publishing script for mlx-gui
# Usage: ./upload_pip.sh

set -e  # Exit on any error

echo "🚀 MLX-GUI PyPI Upload Script"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Check if .pypirc exists
if [ ! -f "$HOME/.pypirc" ]; then
    echo "❌ Error: ~/.pypirc not found. Please configure your PyPI credentials first."
    echo "   Visit: https://pypi.org/manage/account/token/"
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Install/upgrade build tools
echo "🔧 Installing build dependencies..."
python -m pip install --upgrade build twine

# Build the package
echo "📦 Building package..."
python -m build

# Verify the build
echo "✅ Verifying package integrity..."
python -m twine check dist/*

# Show what will be uploaded
echo "📋 Package contents:"
ls -la dist/

# Get version from pyproject.toml
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
echo "📋 Version to upload: $VERSION"

# Ask for confirmation
echo ""
read -p "🔥 Ready to upload mlx-gui v$VERSION to PyPI? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⬆️  Uploading to PyPI..."
    python -m twine upload dist/*
    
    echo ""
    echo "🎉 Success! MLX-GUI v$VERSION published to PyPI!"
    echo "📦 Package URL: https://pypi.org/project/mlx-gui/$VERSION/"
    echo "💡 Users can now install with: pip install mlx-gui"
    echo ""
    echo "🔗 Next steps:"
    echo "   1. Update GitHub release notes"
    echo "   2. Test installation: pip install mlx-gui==$VERSION"
    echo "   3. Announce the release!"
else
    echo "❌ Upload cancelled."
    exit 0
fi