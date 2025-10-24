# Rickson Quick Start Guide

## Running the Full Pipeline Test

The fastest way to see Rickson in action is to run the comprehensive test:

```bash
cd rickson
python tests/test_full_pipeline.py
```

This test demonstrates:
- ✓ Synthetic video generation (10s @ 30fps with breathing pattern)
- ✓ EVM temporal filtering and breath rate estimation
- ✓ Pose estimation and keypoint tracking (if MediaPipe available)
- ✓ Balance analysis with support polygon computation
- ✓ Training insights generation
- ✓ Event logging with immutable append-only log

**Expected output:**
```
======================================================================
RICKSON MVP - Full Pipeline Test
======================================================================

[Video] Creating synthetic video source...
  ✓ Synthetic video: 300 frames
[EVM] Initializing breath analysis pipeline...
  ✓ EVM pipeline ready
[Pose] Initializing pose estimator...
  ✓ Pose estimator ready
...
Processing 300 frames...
  Frame 60/300 (20%) - Breath: 17.8 BPM
  Frame 120/300 (40%) - Breath: 18.2 BPM
  ...

✓ Processed 300 frames

[EVM Breath Analysis]
  Mean breath rate: 18.1 BPM
  Error: 0.1 BPM ✓ PASS

Training Insights:
1. ℹ️ Low Breathing Rate [breath]
   Average breathing rate of 18.1 BPM indicates good breath control.
   → Maintain this controlled breathing during high-intensity drills.

🎉 ALL TESTS PASSED - MVP READY!
```

## Running in Omniverse Kit (GUI)

### Option 1: Use Launch Script

```bash
./scripts/launch_kit.sh dev
```

### Option 2: Manual Launch

```bash
# Replace with your Kit SDK path
~/.local/share/ov/pkg/kit-sdk-105.1/kit \
    --enable omni.kit.window.extensions \
    --enable zs.core \
    --enable zs.ui \
    --enable zs.evm \
    --enable zs.pose \
    --ext-folder ./exts \
    app/rickson.dev.kit
```

### Using the UI

1. **Start Training Session**
   - Click **"Start Capture"** button
   - Synthetic video will begin processing
   - Live metrics update in real-time

2. **Adjust EVM Parameters** (Scrubbable!)
   - **Alpha**: Amplification factor (0-50)
     - Lower (5-10): Subtle amplification
     - Higher (20-30): More visible breathing motion
   - **Low Freq**: Lower cutoff (0.05-2 Hz)
     - Default 0.1 Hz = 6 breaths/min minimum
   - **High Freq**: Upper cutoff (0.1-3 Hz)
     - Default 0.7 Hz = 42 breaths/min maximum
   - **Pyramid Levels**: Spatial decomposition (2-8)
     - More levels = more detail, slower processing

3. **Monitor Live Metrics**
   - **Breath Rate**: Current breathing rate in BPM
   - **Balance Score**: Stability score (0-100)
   - **Insights Panel**: Real-time training feedback

4. **Explore Insights**
   - Click **"Explain This Alert"** to see derivation
   - Shows which parameters contributed to current metrics

5. **Control Playback**
   - **Pause**: Stop processing, adjust parameters
   - **Reset**: Clear all state, restart from beginning

## Using Real Video

### From File

Edit `exts/zs.ui/zs/ui/ui_panel.py`, line 247:

```python
# Replace this:
self._video_manager.open_synthetic(...)

# With this:
self._video_manager.open_file("/path/to/your/video.mp4")
```

### From Camera

```python
# Use webcam:
self._video_manager.open_camera(camera_id=0, fps=30.0)
```

## Architecture Overview

```
Video Input → EVM Pipeline → Breath Metrics
     ↓           ↓              ↓
 Pose Estimator → Balance → Training Insights
     ↓           ↓              ↓
  Event Log  ←  All Data  →  UI Display
```

### Extensions

- **zs.core**: Event logging, state management, insights engine
- **zs.ui**: Main UI panel with scrubbable parameters
- **zs.evm**: Eulerian Video Magnification pipeline
- **zs.pose**: MediaPipe pose estimation and balance analysis

### Data Flow

1. **Input**: Video frames (file, camera, or synthetic)
2. **EVM**: Spatial pyramid + temporal band-pass → breath rate
3. **Pose**: MediaPipe keypoints → support polygon → balance score
4. **Insights**: Analyze breath & balance → training recommendations
5. **Events**: Log all measurements (append-only, immutable)
6. **UI**: Display metrics with <200ms latency (Bret Victor style)

## Understanding the Metrics

### Breath Rate (BPM)

- **8-12 BPM**: Excellent control (typical resting rate)
- **12-20 BPM**: Good, normal during light activity
- **20-30 BPM**: Elevated, high intensity or stress
- **>30 BPM**: Very high, near maximal exertion

**Insights:**
- Consistent low rate = good endurance
- High variability = irregular breathing
- Frequent <8 BPM = breath holding (avoid during exertion)

### Balance Score (0-100)

- **75-100**: Strong stability, COM well inside support polygon
- **50-75**: Moderate stability, some balance drift
- **30-50**: Unstable, COM near edge of support
- **<30**: Very unstable, risk of losing balance

**Insights:**
- High score + narrow stance = excellent technique
- Low score = widen stance or lower center of mass
- Score drops during movement = improve transitions

### Stance Types

- **Parallel**: Feet side-by-side (square stance)
- **Staggered**: Front-back positioning (fighting stance)
- **Single Leg**: One foot not visible/grounded

## Customization

### EVM Frequency Bands

Target different physiological signals:

```python
# Breathing (default)
low_freq=0.1, high_freq=0.7  # 6-42 BPM

# Heart rate (pulse)
low_freq=0.8, high_freq=2.0  # 48-120 BPM

# Micro-motions
low_freq=0.05, high_freq=0.3  # Very slow drift
```

### Balance Thresholds

Edit `balance_analyzer.py`:

```python
def detect_balance_edge_alert(self, threshold: float = 30.0):
    # Lower threshold = more sensitive alerts
    # Higher threshold = only critical alerts
```

### Event Log Location

Default: `~/.rickson/logs/session_YYYYMMDD_HHMMSS.jsonl`

Custom:

```python
from pathlib import Path
from zs.core.event_log import EventLog

log_file = Path("/custom/path/my_session.jsonl")
event_log = EventLog(log_file)
```

## Troubleshooting

### "EVM modules not available"

Extensions didn't load properly. Check:

```bash
# Verify extension structure
ls exts/zs.evm/config/extension.toml
ls exts/zs.evm/zs/evm/__init__.py

# Check Kit extension search path
echo $OMNI_KIT_EXT_FOLDERS
```

### "MediaPipe not available"

Pose estimation requires MediaPipe:

```bash
pip install mediapipe>=0.10.8
```

Or skip pose features (breath analysis still works).

### Low Breath Estimation Accuracy

Adjust parameters:

1. **Increase buffer**: More history = better frequency resolution
   ```python
   buffer_seconds=15.0  # Default is 10.0
   ```

2. **Tighter frequency band**: Reduce range if you know expected rate
   ```python
   low_freq=0.15, high_freq=0.4  # For ~10-24 BPM range
   ```

3. **More pyramid levels**: Better spatial decomposition
   ```python
   pyramid_levels=5  # Default is 4
   ```

### UI Not Updating

Check async loop is running:

```python
# In ui_panel.py, add logging:
carb.log_info(f"Update loop iteration: {frame_count}")
```

If no logs, ensure `_processing = True` and `_update_task` started.

## Next Steps

1. **Try with Real Video**
   - Record yourself doing BJJ/Muay Thai drills
   - Process with Rickson
   - Analyze breath patterns and balance

2. **Export Session Data**
   ```python
   from zs.core.event_log import get_event_log

   log = get_event_log()
   log.export_json(Path("my_session.json"))
   ```

3. **Build Custom Insights**
   - Edit `insights_engine.py`
   - Add domain-specific rules (e.g., detect sweep attempts)

4. **Integrate with Training Journal**
   - Read event logs
   - Track progress over time
   - Correlate with training outcomes

## Performance Tips

### For Real-Time Processing

- Use **lite pose model**: `model_complexity=0`
- Reduce **pyramid levels**: `pyramid_levels=3`
- Shorter **buffer**: `buffer_seconds=5.0`
- Lower **resolution**: `640x480` instead of `1920x1080`

### For Maximum Accuracy

- Use **full pose model**: `model_complexity=2`
- More **pyramid levels**: `pyramid_levels=6`
- Longer **buffer**: `buffer_seconds=15.0`
- Higher **resolution**: `1920x1080`
- Process **offline** (not real-time)

## References

- [EVM Paper (MIT CSAIL)](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Omniverse Kit Docs](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/)
- [Architecture Deep Dive](architecture.md)
- [EVM Explained](evm_explained.md)
