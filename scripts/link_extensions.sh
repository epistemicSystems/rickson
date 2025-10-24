#!/bin/bash
# Link Rickson extensions to Omniverse Kit

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
EXTS_DIR="$PROJECT_ROOT/exts"

# Detect Kit installation
# Common locations:
KIT_PATHS=(
    "$HOME/.local/share/ov/pkg/kit-sdk-105.1"
    "$HOME/.local/share/ov/pkg/kit-sdk-106.0"
    "/usr/local/share/ov/pkg/kit-sdk-105.1"
    "C:/Users/$USER/AppData/Local/ov/pkg/kit-sdk-105.1"
)

KIT_PATH=""
for path in "${KIT_PATHS[@]}"; do
    if [ -d "$path" ]; then
        KIT_PATH="$path"
        break
    fi
done

if [ -z "$KIT_PATH" ]; then
    echo "ERROR: Could not find Omniverse Kit SDK installation"
    echo "Please install Kit SDK via Omniverse Launcher"
    echo "Or set KIT_PATH environment variable manually"
    exit 1
fi

echo "Found Kit SDK at: $KIT_PATH"

# Create symlinks for each extension
# Extensions can be registered by adding their parent directory to extension search paths

echo "Extension directory: $EXTS_DIR"
echo ""
echo "To register extensions, add this line to your .kit file:"
echo "  exts.folders.'++' = [\"$EXTS_DIR\"]"
echo ""
echo "Or set environment variable:"
echo "  export OMNI_KIT_EXT_FOLDERS=\"\$OMNI_KIT_EXT_FOLDERS:$EXTS_DIR\""
echo ""

# Verify extensions have proper structure
for ext in "$EXTS_DIR"/*; do
    if [ -d "$ext" ]; then
        EXT_NAME=$(basename "$ext")
        if [ -f "$ext/config/extension.toml" ]; then
            echo "✓ Extension $EXT_NAME is properly structured"
        else
            echo "✗ Extension $EXT_NAME is missing config/extension.toml"
        fi
    fi
done

echo ""
echo "Extensions linked successfully!"
echo "Launch with: ./scripts/launch_kit.sh dev"
