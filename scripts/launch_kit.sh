#!/bin/bash
# Launch Rickson Kit app

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Mode: dev or prod
MODE="${1:-dev}"

if [ "$MODE" = "dev" ]; then
    KIT_FILE="$PROJECT_ROOT/app/rickson.dev.kit"
    echo "Launching in DEVELOPMENT mode (hot-reload enabled)"
else
    KIT_FILE="$PROJECT_ROOT/app/rickson.kit"
    echo "Launching in PRODUCTION mode"
fi

# Detect Kit installation
KIT_PATHS=(
    "$HOME/.local/share/ov/pkg/kit-sdk-105.1"
    "$HOME/.local/share/ov/pkg/kit-sdk-106.0"
    "/usr/local/share/ov/pkg/kit-sdk-105.1"
    "C:/Users/$USER/AppData/Local/ov/pkg/kit-sdk-105.1"
)

KIT_EXEC=""
for path in "${KIT_PATHS[@]}"; do
    if [ -f "$path/kit" ] || [ -f "$path/kit.exe" ]; then
        KIT_EXEC="$path/kit"
        [ -f "$path/kit.exe" ] && KIT_EXEC="$path/kit.exe"
        break
    fi
done

if [ -z "$KIT_EXEC" ]; then
    echo "ERROR: Could not find Kit executable"
    echo "Please install Omniverse Kit SDK via Omniverse Launcher"
    echo ""
    echo "Or set KIT_EXEC environment variable:"
    echo "  export KIT_EXEC=/path/to/kit"
    exit 1
fi

echo "Kit executable: $KIT_EXEC"
echo "Kit file: $KIT_FILE"
echo ""

# Set extension search path
export OMNI_KIT_EXT_FOLDERS="$PROJECT_ROOT/exts"

# Launch Kit
"$KIT_EXEC" \
    --enable omni.kit.window.extensions \
    --enable zs.ui \
    --enable zs.evm \
    --ext-folder "$PROJECT_ROOT/exts" \
    "$KIT_FILE"
