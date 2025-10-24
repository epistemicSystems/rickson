---
description: "Launch Rickson Kit app in development mode"
---

# Run: Kit Development Mode

Launch the Rickson training assistant in Omniverse Kit with hot-reload enabled.

## Steps

1. **Verify Kit SDK installation**
   - Check common installation paths
   - Report Kit version found

2. **Set up environment**
   - Export extension search paths
   - Set development flags

3. **Launch Kit**
   - Use `rickson.dev.kit` configuration
   - Enable extensions: zs.ui, zs.evm
   - Enable hot-reload

## Command

```bash
./scripts/launch_kit.sh dev
```

## Manual Launch

If script fails, launch manually:

```bash
# Replace with your Kit SDK path
~/.local/share/ov/pkg/kit-sdk-105.1/kit \
    --enable omni.kit.window.extensions \
    --enable zs.ui \
    --enable zs.evm \
    --ext-folder ./exts \
    app/rickson.dev.kit
```

## Expected Behavior

1. Kit window opens with Rickson title
2. Viewport shows the training_gym.usda stage
3. "Rickson Training Assistant" panel appears on the right
4. Console shows extension startup logs:
   - `[zs.ui] Rickson UI Extension starting...`
   - `[zs.evm] Rickson EVM Extension starting...`

## Troubleshooting

**Extension not found:**
- Check `exts/` directory structure
- Verify `config/extension.toml` exists for each extension
- Check extension search paths in Kit file

**UI panel doesn't appear:**
- Go to Window > Extensions
- Search for "zs.ui"
- Click to enable manually

**Hot reload not working:**
- Verify `exts.hotReload.enabled = true` in rickson.dev.kit
- After editing files, go to Window > Extensions and click reload icon

## Next Steps

After successful launch:
1. Test slider interactions
2. Check console for parameter change logs
3. Open OmniGraph editor (Window > Visual Scripting > Action Graph)
4. Add EVMBandPass node and connect parameters
