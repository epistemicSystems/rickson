# Rickson - BJJ/Muay Thai Mocap Training Assistant

A local-first markerless motion capture training assistant for BJJ and Muay Thai athletes, built on NVIDIA Omniverse.

## Features

- **Eulerian Video Magnification (EVM)**: Reveal subtle breathing patterns and micro-movements
- **Markerless Pose Tracking**: Real-time pose estimation with 3D fusion
- **3D Gaussian Splatting**: Use pre-scanned gym environments as spatial priors
- **Scrubbable Parameters**: Bret Victor-inspired immediate feedback UI
- **Event-Driven Architecture**: Immutable event logs for session replay and analysis
- **Local-First**: No cloud dependencies, full privacy control

## Architecture

Built on **NVIDIA Omniverse Kit** with modular extensions:

- **zs.ui**: Main UI panel with EVM parameter controls and insights display
- **zs.evm**: OmniGraph compute nodes for temporal band-pass filtering
- **OpenUSD**: Scene graph for cameras, athletes, and gym environments
- **RTX**: Real-time ray-traced rendering with GPU acceleration

## Prerequisites

- NVIDIA GPU (RTX 2060 or better recommended)
- NVIDIA Omniverse Launcher installed
- Ubuntu 20.04+ or Windows 10/11
- Python 3.10+
- CUDA Toolkit 11.8+ (for custom kernels)

## Quick Start

### 1. Install Omniverse

1. Download and install [NVIDIA Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/)
2. In the Launcher, install **Omniverse Kit SDK** (under "Exchange" or "Library")
3. Note the installation path (e.g., `~/.local/share/ov/pkg/kit-sdk-105.1`)

### 2. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url> rickson
cd rickson

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Link extensions to Omniverse
./scripts/link_extensions.sh
```

### 3. Launch the App

```bash
# Development mode (with hot-reload)
./scripts/launch_kit.sh dev

# Or manually:
# Replace path with your Kit SDK installation
~/.local/share/ov/pkg/kit-sdk-105.1/kit \
    --enable omni.kit.window.extensions \
    --enable zs.ui \
    --enable zs.evm \
    app/rickson.dev.kit
```

### 4. Explore the UI

1. The main **Rickson Training Assistant** panel will open on the right
2. Adjust **EVM Parameters** sliders to see live updates
3. Click **Start Capture** to begin (currently stub - connects to camera in next iteration)
4. Use **Explain This Alert** to see derivation graphs

## Project Structure

```
rickson/
├── CLAUDE.md                 # Development megaprompt
├── README.md                 # This file
├── app/
│   ├── rickson.kit           # Production config
│   └── rickson.dev.kit       # Dev config (hot-reload)
├── exts/
│   ├── zs.ui/                # UI panel extension
│   │   ├── config/extension.toml
│   │   └── zs/ui/
│   │       ├── __init__.py
│   │       └── ui_panel.py
│   └── zs.evm/               # EVM compute extension
│       ├── config/extension.toml
│       └── zs/evm/
│           ├── __init__.py
│           └── nodes/
│               ├── EVMBandPass.py
│               └── EVMBandPass.ogn
├── data/
│   └── stages/
│       └── training_gym.usda # Sample USD stage
├── scripts/
│   ├── link_extensions.sh
│   └── launch_kit.sh
└── docs/
    ├── architecture.md
    └── evm_explained.md
```

## Development Workflow

### Using Claude Code

This project is designed for **Claude Code** with custom slash commands:

```bash
# Bootstrap the project (if starting from scratch)
/plan:omniverse-bootstrap

# Launch Kit in dev mode
/run:kit

# Run GPU kernel tests
/test:gpu-kernels
```

See `CLAUDE.md` for full development instructions.

### Hot Reload

In dev mode, extensions support hot-reload:

1. Edit any Python file in `exts/`
2. In Kit, go to **Window > Extensions**
3. Find your extension and click the reload icon
4. Changes take effect immediately (no restart needed)

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_evm_kernel.py

# Visual regression tests (golden images)
pytest tests/ --golden-compare
```

## Roadmap

### Milestone 1: Bootstrap ✓

- [x] Kit app configuration
- [x] zs.ui extension with parameter sliders
- [x] zs.evm OmniGraph node stub
- [x] Sample USD stage with cameras

### Milestone 2: EVM GPU Pass

- [ ] Implement temporal band-pass filter (CPU reference)
- [ ] CUDA kernel for GPU acceleration
- [ ] Spatial pyramid construction
- [ ] Live breath rate estimation

### Milestone 3: Pose Overlay

- [ ] Integrate pose estimation (MediaPipe or OpenPose)
- [ ] Draw keypoints in viewport
- [ ] Compute support polygon
- [ ] Balance drift metrics

### Milestone 4: 3D Priors

- [ ] Load 3DGS gym scan
- [ ] Multi-cam calibration
- [ ] Constrained 3D pose fusion
- [ ] COM vs. support polygon visualization

### Milestone 5: Insights

- [ ] Breath cadence analysis
- [ ] Breath-hold detection
- [ ] Balance-edge alerts
- [ ] "Explain" panel with OmniGraph visualization

### Milestone 6: Opponent Proto

- [ ] Offline feature extraction from opponent footage
- [ ] Training game suggestions
- [ ] Side-by-side comparison view

### Milestone 7: Record/Replay

- [ ] Append-only event log
- [ ] USD timeline scrubbing
- [ ] Annotation tools
- [ ] Export with face blurring

## Architecture Principles

Following **Rich Hickey** (simplicity over ease):

- **Data-oriented**: Events are immutable; state is derived
- **Pure transforms**: IO at edges, pure functions in the middle
- **Explicit state**: No hidden global state

Following **Bret Victor** (immediate feedback):

- **Scrubbable parameters**: Every value can be adjusted in real-time
- **Visible causality**: Click any metric to see its derivation
- **Freeze-frame lab**: Pause and explore with instant re-computation

## References

- [NVIDIA Omniverse Kit Manual](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/)
- [OmniGraph Documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html)
- [Eulerian Video Magnification Paper](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- [OpenUSD Documentation](https://openusd.org/)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)

## License

[Your License Here]

## Contributing

See `CLAUDE.md` for development guidelines and Claude Code workflows.

## Support

For issues or questions, please open a GitHub issue.
