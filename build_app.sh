#!/bin/bash

# MLX-GUI macOS App Builder
# This script builds a TRUE standalone macOS app bundle using PyInstaller

set -e

echo "🚀 Building MLX-GUI macOS App Bundle (TRUE STANDALONE)..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install PyInstaller if not already installed
echo "📦 Installing PyInstaller..."
pip install pyinstaller

# Check for critical dependencies
echo "🔍 Checking critical dependencies..."
CRITICAL_DEPS=("mlx-lm" "mlx" "rumps" "fastapi" "uvicorn" "transformers" "huggingface-hub" "mlx-whisper" "parakeet-mlx" "mlx-vlm" "timm" "torchvision")
MISSING_DEPS=""

for dep in "${CRITICAL_DEPS[@]}"; do
    if ! pip show "$dep" > /dev/null 2>&1; then
        MISSING_DEPS="$MISSING_DEPS $dep"
    fi
done

if [ -n "$MISSING_DEPS" ]; then
    echo "❌ Missing critical dependencies:$MISSING_DEPS"
    echo "💡 Install with: pip install -e \".[app,audio,vision]\""
    echo "💡 Or from requirements: pip install -r requirements.txt"
    echo "💡 For audio support: pip install mlx-whisper parakeet-mlx"
    echo "💡 For vision support: pip install mlx-vlm timm torchvision"
    exit 1
fi

echo "✅ All critical dependencies found"

# Ensure latest audio and vision dependencies
echo "📦 Ensuring latest audio and vision dependencies..."
pip install parakeet-mlx -U
pip install av -U
pip install ffmpeg-binaries -U
pip install mlx-vlm -U
pip install mlx-lm -U
pip install timm -U
pip install torchvision -U
# Replace full OpenCV with headless build to avoid crypto conflicts
pip install opencv-python-headless -U

# Clean previous builds
echo "🧹 Cleaning previous builds..."
pkill -f MLX-GUI || true
sleep 2
rm -rf build/ dist/ MLX-GUI.spec app_icon.icns 2>/dev/null || true

# Create app icon from PNG
echo "🎨 Creating app icon from ./icon.png..."
if [ -f "./icon.png" ]; then
    # Create iconset directory
    mkdir -p app_icon.iconset

    # Generate different icon sizes using sips (built into macOS)
    sips -z 16 16 ./icon.png --out app_icon.iconset/icon_16x16.png
    sips -z 32 32 ./icon.png --out app_icon.iconset/icon_16x16@2x.png
    sips -z 32 32 ./icon.png --out app_icon.iconset/icon_32x32.png
    sips -z 64 64 ./icon.png --out app_icon.iconset/icon_32x32@2x.png
    sips -z 128 128 ./icon.png --out app_icon.iconset/icon_128x128.png
    sips -z 256 256 ./icon.png --out app_icon.iconset/icon_128x128@2x.png
    sips -z 256 256 ./icon.png --out app_icon.iconset/icon_256x256.png
    sips -z 512 512 ./icon.png --out app_icon.iconset/icon_256x256@2x.png
    sips -z 512 512 ./icon.png --out app_icon.iconset/icon_512x512.png
    sips -z 1024 1024 ./icon.png --out app_icon.iconset/icon_512x512@2x.png

    # Convert to icns format
    iconutil -c icns app_icon.iconset -o app_icon.icns

    # Clean up temporary iconset
    rm -rf app_icon.iconset

    echo "✅ App icon created: app_icon.icns"
else
    echo "⚠️  Warning: ./icon.png not found, using default icon"
fi

# Build the app using PyInstaller directly
echo "🔨 Building app bundle with PyInstaller..."

# Set environment variables to prevent model downloads during build
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_DISABLE_CUDA_MALLOC=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create PyInstaller hooks directory if it doesn't exist
mkdir -p hooks

# Create custom hook for parakeet-mlx
cat > hooks/hook-parakeet_mlx.py << 'EOF'
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = collect_all('parakeet_mlx')

# Bundle all submodules so STT works out of the box
hiddenimports.extend(collect_submodules('parakeet_mlx'))
EOF

# Create custom hook for audiofile
cat > hooks/hook-audiofile.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('audiofile')

# Additional hidden imports for audiofile
hiddenimports += [
    'audiofile.core',
    'audmath',
    'audeer',
    'soundfile',
    'cffi',
    'pycparser',
]
EOF

# Create custom hook for audresample
cat > hooks/hook-audresample.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('audresample')

# Additional hidden imports for audresample
hiddenimports += [
    'soxr',
    'numba',
    'llvmlite',
]
EOF

# Note: Removed ffmpeg-python hook as we're using Python av package instead

# Create custom hook for av (PyAV)
cat > hooks/hook-av.py << 'EOF'
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = collect_all('av')

# Collect all av dynamic libraries (libav* dylibs)
av_dylibs = collect_dynamic_libs('av')
binaries.extend(av_dylibs)

# Additional hidden imports for av
hiddenimports += [
    'av',
    'av.audio',
    'av.codec',
    'av.container',
    'av.format',
    'av.stream',
    'av.video',
    'av.filter',
    'av.packet',
    'av.frame',
    'av.plane',
    'av.subtitles',
    'av.logging',
    'av.utils',
]
EOF

# Create custom hook for mlx-whisper
cat > hooks/hook-mlx_whisper.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('mlx_whisper')

# Additional hidden imports for mlx-whisper
hiddenimports += [
    'mlx_whisper.transcribe',
    'mlx_whisper.load_models',
    'mlx_whisper.audio',
]
EOF

# Create custom hook for transformers to ensure all processor modules are included
cat > hooks/hook-transformers.py << 'EOF'
from PyInstaller.utils.hooks import collect_all, collect_submodules

# The "Sledgehammer" approach: collect EVERYTHING from transformers.
# This is the most robust way to ensure all dynamic modules, models,
# and processors are included, preventing "Could not import module" errors.

datas, binaries, hiddenimports = collect_all('transformers')

# Recursively collect all submodules. This is the key to solving the problem.
hiddenimports.extend(collect_submodules('transformers'))

print("✅ Aggressive transformers hook: All submodules collected.")
EOF

# Create cv2 hook that filters out SSL/crypto libraries that conflict with Python
cat > hooks/hook-cv2.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

# Collect cv2 but filter out problematic crypto libraries
datas, binaries, hiddenimports = collect_all('cv2')

# Filter out specific SSL/crypto libraries that conflict with Python's SSL
problematic_libs = [
    'libcrypto.3.dylib',
    'libssl.3.dylib',
    'libmbedcrypto.3.5.1.dylib',
    'libcrypto',
    'libssl',
    'crypto.3',
    'ssl.3'
]

filtered_binaries = []
for src, dest in binaries:
    # Check if this binary contains any problematic libraries
    skip = False
    for lib in problematic_libs:
        if lib in src:
            print(f"🔧 Filtering out problematic crypto library: {src}")
            skip = True
            break

    if not skip:
        filtered_binaries.append((src, dest))

binaries = filtered_binaries

# Essential cv2 imports for transformers and basic vision functionality
hiddenimports += [
    'cv2',
    'cv2.cv2',
    'numpy',
]

print(f"✅ OpenCV hook: filtered {len(binaries) - len(filtered_binaries)} problematic libraries")
EOF

# Create custom hook for mlx-vlm
cat > hooks/hook-mlx_vlm.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('mlx_vlm')

# Additional hidden imports for mlx-vlm
hiddenimports += [
    'mlx_vlm.generate',
    'mlx_vlm.load',
    'mlx_vlm.utils',
    'mlx_vlm.prompt_utils',
    'mlx_vlm.models',
    'mlx_vlm.models.base',
    'mlx_vlm.models.gemma3n',
    'mlx_vlm.models.qwen2_vl',
    'mlx_vlm.models.llava',
    'mlx_vlm.models.idefics',
    'timm',
    'timm.models',
    'timm.models.vision_transformer',
    'timm.models.convnext',
    'timm.models.swin_transformer',
    'timm.layers',
    'timm.data',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
]
EOF

# Create custom hook for mlx to ensure internal helper modules are bundled without duplicating the core lib
cat > hooks/hook-mlx.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('mlx')

# Explicitly include lazy-imported helpers
hiddenimports += [
    'mlx._reprlib_fix',
    'mlx._os_warning',
]
EOF

# Create a temporary directory for runtime hooks
HOOKS_DIR="rthooks"
mkdir -p "$HOOKS_DIR"

# Path for the consolidated runtime hook
ALL_FIXES_HOOK="$HOOKS_DIR/pyi_rth_all_fixes.py"

# Create the consolidated runtime hook file
echo "Creating consolidated runtime hook: $ALL_FIXES_HOOK"
cat > "$ALL_FIXES_HOOK" << EOL
# rthooks/pyi_rth_all_fixes.py
# This file is dynamically generated by build_app.sh

import sys
import os

print("--- Running MLX-GUI Runtime Fixes ---")

# -- Fix for ffmpeg/av --
try:
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS

        # Fix for ffmpeg
        ffmpeg_path = os.path.join(bundle_dir, 'ffmpeg')
        if os.path.exists(ffmpeg_path):
            os.environ['PATH'] = f"{os.path.dirname(ffmpeg_path)}{os.pathsep}{os.environ.get('PATH', '')}"
            # print("✅ FFmpeg binary path configured.")
        else:
            # Fallback for older ffmpeg-binaries structure
            ffmpeg_dir = os.path.join(bundle_dir, 'ffmpeg-binaries', 'bin')
            if os.path.exists(ffmpeg_dir):
                os.environ['PATH'] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                # print("✅ FFmpeg binary (fallback) path configured.")
            else:
                import ffmpeg
                print("⚠️ FFmpeg binary path not found in bundle.")


        # Fix for av
        av_dir = os.path.join(bundle_dir, 'av')
        if os.path.exists(av_dir):
            os.environ['AV_ROOT'] = av_dir
            # print("✅ PyAV (av) libraries configured.")
        else:
            print("⚠️ PyAV (av) libraries not found in bundle.")

except Exception as e:
    print(f"⚠️ Error in ffmpeg/av fix: {e}")


# -- Basic transformers sanity check --
try:
    import transformers  # noqa: F401
    # If we reach here, transformers is importable with its metadata.
except Exception as e:
    print(f"⚠️ Transformers import sanity check failed: {e}")

# -- Fix for cv2/OpenCV --
# This is more of a check to ensure transformers can find cv2
try:
    import cv2
    print("✅ Minimal cv2 available for feature detection.")
except ImportError:
    print("⚠️ cv2 (OpenCV) not found, which may affect some vision models.")
except Exception as e:
    print(f"⚠️ An unexpected error occurred during cv2 check: {e}")

# -- Patch for missing Gemma3N VLM bias parameter --
try:
    import mlx_vlm.utils as vlm_utils
    if not hasattr(vlm_utils, '__bias_patch_applied'):
        orig_sanitize = vlm_utils.sanitize_weights
        def patched_sanitize(model_obj, weights, config=None):
            weights = orig_sanitize(model_obj, weights, config)
            bias_key = 'vision_tower.timm_model.conv_stem.conv.bias'
            weight_key = 'vision_tower.timm_model.conv_stem.conv.weight'
            if bias_key not in weights and weight_key in weights:
                try:
                    import mlx.core as mx
                    w = weights[weight_key]
                    weights[bias_key] = mx.zeros((w.shape[0],), dtype=w.dtype)
                    print('✅ Patched missing VLM conv_stem bias')
                except Exception as e:
                    print(f'⚠️ Failed to patch VLM bias: {e}')
            return weights
        vlm_utils.sanitize_weights = patched_sanitize
        vlm_utils.__bias_patch_applied = True
except Exception as e:
    print(f'⚠️ Unable to apply VLM bias patch: {e}')

print("--- Runtime Fixes Completed ---")

EOL

# Base PyInstaller command
PYINSTALLER_CMD=(
    "pyinstaller"
    "src/mlx_gui/app_main.py"
    "--name" "MLX-GUI"
    "--windowed"
    "--noconfirm"
    "--clean"
    "--onedir" # Use onedir for macOS .app bundles
    "--additional-hooks-dir" "hooks"
    "--runtime-hook" "$ALL_FIXES_HOOK"
    "--icon" "app_icon.icns"
    "--osx-bundle-identifier" "org.matthewrogers.mlx-gui"
    "--copy-metadata" "tqdm"
    "--copy-metadata" "regex"
    "--copy-metadata" "safetensors"
    "--copy-metadata" "filelock"
    "--copy-metadata" "numpy"
    "--copy-metadata" "requests"
    "--copy-metadata" "packaging"
    "--copy-metadata" "pyyaml"
    "--copy-metadata" "tokenizers"
    "--copy-metadata" "huggingface-hub"
    "--copy-metadata" "transformers"
    "--copy-metadata" "timm"
    "--copy-metadata" "torch"
    "--copy-metadata" "torchvision"
    "--copy-metadata" "sentencepiece"
    "--copy-metadata" "Pillow"
    "--copy-metadata" "av"
    "--copy-metadata" "parakeet-mlx"
    "--copy-metadata" "mlx-vlm"
    "--copy-metadata" "mlx-lm"
    "--copy-metadata" "Jinja2"
    "--copy-metadata" "opencv-python-headless"
    # Include HTML templates and media assets
    "--add-data" "src/mlx_gui/templates:mlx_gui/templates"
    "--add-data" "media:media"
    "--hidden-import" "scipy.sparse.csgraph._validation"
    "--hidden-import" "mlx._reprlib_fix"
    "--hidden-import" "Jinja2"
    "--exclude-module" "tkinter"
    "--exclude-module" "PySide6"
    "--exclude-module" "PyQt6"
    "--exclude-module" "wx"
)

# Read version from Python module
VERSION=$(python3 -c "from src.mlx_gui import __version__; print(__version__)")
echo "📝 Building version: $VERSION"

# Run PyInstaller with all the options
echo "🔨 Building app bundle with PyInstaller..."
# shellcheck disable=SC2086
"${PYINSTALLER_CMD[@]}"

# Clean up temporary hook files
echo "🧹 Cleaning up temporary hook files..."
rm -rf "$HOOKS_DIR"

# Check if build was successful
if [ -d "dist/MLX-GUI.app" ]; then
    echo "✅ App bundle built successfully!"
    echo "📍 Location: dist/MLX-GUI.app"

    # Fix the Info.plist to make it a menu bar app (no dock icon) - BEFORE signing
    echo "🔧 Converting to menu bar app (removing dock icon)..."
    INFO_PLIST="dist/MLX-GUI.app/Contents/Info.plist"

    if [ -f "$INFO_PLIST" ]; then
        # Add LSUIElement=true to make it a menu bar app
        /usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$INFO_PLIST" 2>/dev/null || \
        /usr/libexec/PlistBuddy -c "Set :LSUIElement true" "$INFO_PLIST"

        # Add version information to Info.plist
        /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string $VERSION" "$INFO_PLIST" 2>/dev/null || \
        /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$INFO_PLIST"

        /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string $VERSION" "$INFO_PLIST" 2>/dev/null || \
        /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$INFO_PLIST"

        echo "✅ App converted to menu bar app (no dock icon)"
        echo "   - App will only appear in the menu bar"
        echo "   - No dock icon will be shown"
        echo "   - Version set to: $VERSION"
    else
        echo "⚠️  Warning: Could not find Info.plist at $INFO_PLIST"
    fi

    # Code signing section
    echo ""
    echo "🔐 Code Signing..."

    # Check if we have a Developer ID Application certificate
    CERT_NAME=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)".*/\1/')

    if [ -n "$CERT_NAME" ]; then
        echo "📝 Found certificate: $CERT_NAME"
        echo "🔏 Signing app bundle..."

        # Sign all executables and libraries first (deep signing)
        codesign --force --deep --sign "$CERT_NAME" --options runtime --entitlements entitlements.plist "dist/MLX-GUI.app"

        # Verify the signature
        if codesign --verify --verbose "dist/MLX-GUI.app" 2>/dev/null; then
            echo "✅ App successfully signed!"
            echo "🛡️  This will eliminate macOS security warnings"

            # Show signature info
            echo ""
            echo "📜 Signature Info:"
            codesign -dv --verbose=4 "dist/MLX-GUI.app" 2>&1 | grep -E "(Identifier|TeamIdentifier|Authority)"
        else
            echo "⚠️  Warning: Code signing verification failed"
            echo "   The app was built but may show security warnings"
        fi
    else
        echo "⚠️  No Developer ID Application certificate found"
        echo "   App will show security warnings when downloaded"
        echo "   To fix this:"
        echo "   1. Get an Apple Developer account ($99/year)"
        echo "   2. Create a Developer ID Application certificate"
        echo "   3. Install it in Keychain Access"
        echo "   4. Re-run this build script"
    fi

    echo ""
    echo "🎉 You can now:"
    echo "   1. Run: open dist/MLX-GUI.app"
    echo "   2. Copy to /Applications: cp -R dist/MLX-GUI.app /Applications/"
    echo "   3. Create a DMG installer"
    echo ""
    echo "📋 App Info:"
    echo "   - Size: $(du -sh dist/MLX-GUI.app | cut -f1)"
    echo "   - Type: TRUE STANDALONE (no Python required!)"
    echo "   - Includes: All Python runtime, MLX binaries, audio & vision support, and dependencies"
    if [ -n "$CERT_NAME" ]; then
        echo "   - Code Signed: ✅ (no security warnings)"
    else
        echo "   - Code Signed: ❌ (will show security warnings)"
    fi
    echo ""
    echo "🎯 This is a REAL standalone app!"
    echo "   - No Python installation required on target system"
    echo "   - No virtual environment needed"
    echo "   - Fully self-contained"
else
    echo "❌ Build failed! App bundle not found at dist/MLX-GUI.app"
    echo "   Check the output above for errors."
    exit 1
fi

echo ""
echo "🔗 Next steps:"
echo "   • Test the app: open dist/MLX-GUI.app"
echo "   • Create DMG installer for easy distribution"
echo "   • App is ready for sharing with anyone - no setup required!"
echo "   • Audio & Vision support included: Whisper, Parakeet, and MLX-VLM models work out of the box (filtered OpenCV - no SSL conflicts)"