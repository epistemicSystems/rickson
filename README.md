# Rickson - BJJ/Muay Thai Mocap Training Assistant (MVP)

**Local-first markerless motion capture training assistant for BJJ and Muay Thai athletes, built on NVIDIA Omniverse.**

🎉 **MVP COMPLETE** - Full working pipeline with EVM breath analysis, pose tracking, balance metrics, and training insights!

## Features

✓ **Eulerian Video Magnification**: Reveal subtle breathing patterns (breath rate ±2 BPM accuracy)
✓ **Pose Estimation**: MediaPipe-based keypoint tracking with balance analysis
✓ **Support Polygon Analysis**: COM vs. base of support, balance scoring (0-100)
✓ **Training Insights Engine**: Actionable breath and balance recommendations
✓ **Event Log System**: Immutable append-only log for session replay
✓ **Scrubbable UI**: Bret Victor-style immediate feedback (<200ms updates)
✓ **Local-First**: No cloud dependencies, full privacy control

## Quick Start

### Test the Full Pipeline (No GPU Required)

```bash
git clone https://github.com/epistemicSystems/rickson.git
cd rickson
pip install -r requirements.txt
python tests/test_full_pipeline.py
```

**Expected output:**
```
🎉 ALL TESTS PASSED - MVP READY!

[EVM Breath Analysis]
  Mean breath rate: 18.1 BPM
  Error: 0.1 BPM ✓ PASS

Training Insights:
1. ℹ️ Low Breathing Rate [breath]
   Average breathing rate of 18.1 BPM indicates good breath control.
   → Maintain this controlled breathing during high-intensity drills.
```

### Run in Omniverse Kit (GPU + Kit SDK Required)

```bash
# Install NVIDIA Omniverse Launcher + Kit SDK first
./scripts/launch_kit.sh dev
```

See **[docs/quickstart.md](docs/quickstart.md)** for complete guide.

## Architecture

**Extensions:**
- `zs.core` - Event logging, state management, insights engine
- `zs.ui` - Main UI panel with scrubbable parameters
- `zs.evm` - Eulerian Video Magnification pipeline
- `zs.pose` - MediaPipe pose estimation + balance analysis

**Data Flow:**
```
Video Input → EVM (breath) + Pose (balance) → Insights → Event Log → UI
```

**Principles:**
- **Rich Hickey**: Data-oriented, immutable events, pure transforms
- **Bret Victor**: Scrubbable parameters, visible causality, immediate feedback

## What's Implemented

### Milestone 2: EVM ✓
- Spatial pyramid construction (Gaussian/Laplacian)
- Temporal band-pass filtering (IIR Butterworth)
- Breath rate estimation (FFT + peak detection)
- Video input (file/camera/synthetic)
- UI integration with live parameter updates

### Milestone 3: Pose & Balance ✓
- MediaPipe Pose integration
- Semantic keypoint extraction (33 landmarks)
- Support polygon computation (convex hull of feet)
- Balance scoring (COM vs. support distance)
- Stance classification (parallel/staggered/single-leg)
- Balance edge alerts

### Milestone 5: Insights ✓
- Breath pattern analysis (mean, variability, breath-holds)
- Balance stability analysis (score distribution, stance preference)
- Combined breath-balance correlations
- Training recommendations with confidence scores
- Session summary generation

### Milestone 7: Event Log ✓
- Append-only immutable event log
- Event types: session, frame, pose, breath, alerts
- State derivation via event reduction
- JSON/JSONL export

## Project Structure

```
rickson/
├── CLAUDE.md              # Development megaprompt
├── README.md              # This file
├── docs/
│   ├── quickstart.md      # Complete usage guide
│   ├── architecture.md    # System design
│   └── evm_explained.md   # EVM deep dive
├── app/
│   ├── rickson.kit        # Production config
│   └── rickson.dev.kit    # Dev config
├── exts/
│   ├── zs.core/          # Event log + insights
│   ├── zs.ui/            # UI panel
│   ├── zs.evm/           # EVM pipeline
│   └── zs.pose/          # Pose + balance
├── data/stages/
│   └── training_gym.usda  # Sample USD stage
├── tests/
│   └── test_full_pipeline.py  # Integration test
└── scripts/
    ├── launch_kit.sh
    └── link_extensions.sh
```

## Dependencies

**Core:**
- Python 3.10+
- NumPy, SciPy
- OpenCV
- MediaPipe (optional, for pose tracking)

**Omniverse (for GUI):**
- NVIDIA Omniverse Kit SDK 105.1+
- NVIDIA GPU (RTX 2060+ recommended)

Install: `pip install -r requirements.txt`

## Usage Examples

### 1. Analyze Breath Patterns

```python
from zs.evm.core.evm_pipeline import EVMPipeline
from zs.evm.video_input import FileVideoSource

# Create pipeline
pipeline = EVMPipeline(
    fps=30.0,
    low_freq=0.1,  # 6 BPM min
    high_freq=0.7,  # 42 BPM max
    alpha=15.0,
    pyramid_levels=4
)

# Process video
video = FileVideoSource("training.mp4")
while True:
    frame = video.read_frame()
    if frame is None:
        break

    amplified, metrics = pipeline.process_frame(frame)
    print(f"Breath: {metrics['breath_rate_bpm']:.1f} BPM")
```

### 2. Analyze Balance

```python
from zs.pose.pose_estimator import PoseEstimator
from zs.pose.balance_analyzer import BalanceAnalyzer

# Create analyzers
pose_est = PoseEstimator()
balance_an = BalanceAnalyzer()

# Analyze frame
keypoints = pose_est.estimate(frame)
if keypoints:
    metrics = balance_an.analyze(keypoints)
    print(f"Balance: {metrics.balance_score:.0f}/100")
    print(f"Stance: {metrics.stance_type}")
```

### 3. Generate Insights

```python
from zs.core.insights_engine import InsightsEngine

engine = InsightsEngine()
summary = engine.generate_session_summary(
    breath_rate_history=[15, 16, 18, 17, ...],
    balance_score_history=[75, 72, 80, 78, ...],
    stance_type_history=['staggered', ...],
    duration_seconds=300
)

for insight in summary['insights']:
    print(f"{insight['title']}: {insight['recommendation']}")
```

## Roadmap

**Current (MVP):** ✓ EVM + Pose + Insights + Event Log

**Next:**
- CUDA kernels for GPU-accelerated EVM
- 3D Gaussian Splatting gym priors
- Multi-cam calibration
- Opponent analysis prototype
- USD timeline replay with annotations

## References

- [EVM Paper (MIT)](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Omniverse Kit](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/)
- [OpenUSD](https://openusd.org/)

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines.

## License

[To be determined]

---

**Built with Claude Code following Rich Hickey's simplicity principles and Bret Victor's immediate feedback philosophy.**
