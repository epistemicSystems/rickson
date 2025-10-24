---
description: "Bootstrap Omniverse Kit app with zs.ui and zs.evm extensions"
---

# Plan: Omniverse Bootstrap

You are bootstrapping the Rickson BJJ/Muay Thai mocap MVP as an NVIDIA Omniverse Kit application.

## Tasks

1. **Scaffold Kit app configuration**
   - Create `app/rickson.kit` (production config)
   - Create `app/rickson.dev.kit` (dev config with hot-reload)

2. **Create zs.ui Extension**
   - Extension config: `exts/zs.ui/config/extension.toml`
   - Main module: `exts/zs.ui/zs/ui/__init__.py`
   - UI panel: `exts/zs.ui/zs/ui/ui_panel.py`
   - Features:
     - Scrubbable sliders for EVM parameters (alpha, low/high freq, levels)
     - Live metrics display (breath rate, balance score)
     - Insights panel with "Explain" button
     - Control buttons (Start, Pause, Reset)

3. **Create zs.evm Extension**
   - Extension config: `exts/zs.evm/config/extension.toml`
   - Main module: `exts/zs.evm/zs/evm/__init__.py`
   - OmniGraph node: `exts/zs.evm/zs/evm/nodes/EVMBandPass.py`
   - Node definition: `exts/zs.evm/zs/evm/nodes/EVMBandPass.ogn`
   - Features:
     - Temporal band-pass filter (CPU reference implementation)
     - Breath rate estimation stub
     - TODO markers for GPU/CUDA optimization

4. **Create sample USD stage**
   - Path: `data/stages/training_gym.usda`
   - Contains:
     - Training space with floor
     - Camera rig (main, left, right cameras)
     - Lighting setup
     - Placeholders for athlete pose and gym scan

5. **Documentation and setup**
   - README.md with quick start guide
   - requirements.txt for Python dependencies
   - scripts/link_extensions.sh
   - scripts/launch_kit.sh

## Principles

- **Small, testable steps**: Each component should be minimal and functional
- **Scrubbable parameters**: Follow Bret Victor's principles
- **Pure data transforms**: Follow Rich Hickey's principles
- **Local-first**: No cloud dependencies

## Expected Output

After running this command, the project should have:
- A working Kit app that launches with both extensions enabled
- UI panel with functional sliders (values logged when changed)
- USD stage that can be opened in viewport
- Clear documentation for next steps

## Next Steps

After bootstrap:
1. Test the UI sliders and verify logging
2. Implement actual EVM temporal filtering
3. Connect UI sliders to OmniGraph node parameters
4. Add video capture pipeline
