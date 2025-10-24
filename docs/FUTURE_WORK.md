# Future Work Features - Implementation Guide

This document describes the advanced features implemented for the Rickson MVP.

## Table of Contents

1. [GPU-Accelerated EVM](#gpu-accelerated-evm)
2. [3D Gaussian Splatting](#3d-gaussian-splatting)
3. [Multi-Camera System](#multi-camera-system)
4. [Opponent Analysis](#opponent-analysis)
5. [Timeline Replay](#timeline-replay)
6. [Privacy Features](#privacy-features)

---

## GPU-Accelerated EVM

**Location:** `exts/zs.evm/zs/evm/cuda/`

### Overview

CUDA-accelerated Eulerian Video Magnification targeting <10ms latency for real-time processing on RTX GPUs.

### Components

- **Pyramid Kernels** (`pyramid.cu`): GPU-accelerated Gaussian/Laplacian pyramid construction
- **Temporal Filter** (`temporal_filter.cu`): IIR Butterworth filtering with per-pixel state
- **Python Wrapper** (`gpu_pipeline.py`): PyCUDA integration with CPU fallback

### Building

```bash
cd exts/zs.evm/zs/evm/cuda
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
make
```

### Usage

```python
from zs.evm.cuda.gpu_pipeline import GPUEVMPipeline

pipeline = GPUEVMPipeline(
    fps=30.0,
    low_freq=0.2,
    high_freq=0.5,
    alpha=15.0,
    pyramid_levels=4,
    use_cuda=True
)

for frame in video_frames:
    amplified, metrics = pipeline.process_frame(frame)
    print(f"GPU time: {metrics['gpu_time_ms']:.2f}ms")
```

### Performance

- **Target:** <10ms per frame on RTX 2060+
- **Fallback:** Automatic CPU fallback if CUDA unavailable
- **Optimization:** Shared memory tiling, coalesced memory access

### Testing

```bash
python tests/test_gpu_kernels.py
```

---

## 3D Gaussian Splatting

**Location:** `exts/zs.3dgs/`

### Overview

3D Gaussian Splat gym environment priors for constrained 3D pose estimation and depth queries.

### Components

- **Splat Loader** (`splat_loader.py`): Load/save .ply format 3DGS
- **USD Integration** (`usd_integration.py`): Create USD Points prims for rendering
- **Gym Prior** (`gym_prior.py`): Spatial queries (depth, floor, collision)

### Creating Gym Priors

1. **Capture:** Use 3DGS tools (Gaussian Splatting, Nerfstudio, etc.) to scan gym
2. **Export:** Save as .ply file
3. **Load:**

```python
from zs.gaussian_splats.gym_prior import GymPrior

gym_prior = GymPrior.from_file('data/gym_scan.ply')

# Query depth along ray
depth = gym_prior.query_depth(
    ray_origin=np.array([0, 0, 2]),
    ray_direction=np.array([0, 0, -1]),
    max_distance=10.0
)

# Get floor height
floor_z = gym_prior.get_floor_height()

# Check if position on mat
on_mat = gym_prior.is_on_mat(np.array([1, 1, 0.5]))
```

### USD Integration

```python
from zs.gaussian_splats.usd_integration import load_gym_prior

# Load into USD stage
points_prim = load_gym_prior(
    stage,
    ply_path='data/gym_scan.ply',
    prim_path='/World/GymPrior',
    downsample=4,
    min_opacity=0.1
)
```

### Data Format

Standard 3DGS PLY format:
- Positions (x, y, z)
- Colors (SH DC coefficients)
- Opacities (sigmoid)
- Scales (log-space)
- Rotations (quaternions)

---

## Multi-Camera System

**Location:** `exts/zs.pose/zs/pose/multicam/`

### Overview

Multi-camera calibration, synchronization, and 3D pose fusion with gym prior constraints.

### Components

1. **Calibration** (`calibration.py`)
   - Intrinsic calibration (checkerboard/ChArUco)
   - Extrinsic calibration (stereo)
   - Save/load calibration data

2. **Synchronization** (`synchronization.py`)
   - Hardware sync (external trigger)
   - LED flash detection
   - Timestamp matching

3. **3D Pose Fusion** (`pose_fusion.py`)
   - Multi-view triangulation
   - 3DGS depth constraints
   - Temporal smoothing

### Workflow

**1. Calibrate Cameras**

```python
from zs.pose.multicam.calibration import MultiCameraCalibration

calib = MultiCameraCalibration()

# Capture checkerboard images from each camera
images_cam1 = [...]  # List of calibration frames
images_cam2 = [...]

# Calibrate intrinsics
intrinsics1 = calib.calibrate_intrinsics('cam1', images_cam1)
intrinsics2 = calib.calibrate_intrinsics('cam2', images_cam2)

# Calibrate stereo extrinsics
extrinsics1, extrinsics2 = calib.calibrate_extrinsics_stereo(
    'cam1', 'cam2',
    images_cam1, images_cam2
)

# Save
calib.save('calibration.json')
```

**2. Synchronize Frames**

```python
from zs.pose.multicam.synchronization import FrameSynchronizer

sync = FrameSynchronizer(sync_method='timestamp', max_time_diff=0.033)
sync.add_camera('cam1')
sync.add_camera('cam2')

# Add frames with timestamps
sync.add_frame('cam1', frame1, timestamp1, frame_num1)
sync.add_frame('cam2', frame2, timestamp2, frame_num2)

# Get synchronized frame set
synced = sync.get_synced_frame()

if synced:
    print(f"Sync confidence: {synced.sync_confidence:.3f}")
    # Process synced.frames['cam1'], synced.frames['cam2']
```

**3. Fuse 3D Poses**

```python
from zs.pose.multicam.pose_fusion import Pose3DFusion
from zs.pose.pose_estimator import PoseEstimator

# Load calibration
calib = MultiCameraCalibration.load('calibration.json')

# Load gym prior (optional)
gym_prior = GymPrior.from_file('data/gym_scan.ply')

# Create fusion
fusion = Pose3DFusion(calib, gym_prior, temporal_smoothing=0.7)

# Estimate 2D poses from each camera
pose_estimator = PoseEstimator()
pose_2d_cam1 = pose_estimator.estimate(frame_cam1)
pose_2d_cam2 = pose_estimator.estimate(frame_cam2)

# Fuse to 3D
pose_3d = fusion.fuse(
    {'cam1': pose_2d_cam1, 'cam2': pose_2d_cam2},
    timestamp=time.time()
)

if pose_3d:
    com = pose_3d.get_center_of_mass()
    print(f"3D CoM: {com}")
    print(f"Floor height: {pose_3d.floor_height:.2f}m")
```

---

## Opponent Analysis

**Location:** `exts/zs.opponent/`

### Overview

Extract features from opponent footage and generate training game recommendations.

### Components

1. **Feature Extractor** (`feature_extractor.py`)
   - Stance analysis
   - Movement patterns
   - Attack/defense frequencies
   - Breath patterns and fatigue

2. **Training Games** (`training_games.py`)
   - Rule-based game recommendations
   - Priority scoring
   - Training plan generation

### Workflow

**1. Analyze Opponent Footage**

```python
from zs.opponent.feature_extractor import OpponentFeatureExtractor

extractor = OpponentFeatureExtractor()

# Process opponent footage
for frame_data in opponent_video:
    extractor.add_frame_data(
        timestamp=frame_data['time'],
        pose_keypoints=frame_data['keypoints'],
        breath_rate=frame_data['breath_bpm'],
        stance_type=frame_data['stance']
    )

    # Log events
    if frame_data['guard_pull']:
        extractor.add_event('guard_pull')

    if frame_data['strike']:
        extractor.add_event('strike')

# Extract profile
profile = extractor.extract_profile('Opponent_Name')

print(f"Stance: {profile.stance}")
print(f"Pressure style: {profile.pressure_style}")
print(f"Guard pull rate: {profile.guard_pull_rate:.1f}/min")
print(f"Fatigue onset: {profile.fatigue_onset:.0f}s")

# Save profile
profile.save('opponents/opponent_name.json')
```

**2. Get Training Recommendations**

```python
from zs.opponent.training_games import TrainingGameRecommender

recommender = TrainingGameRecommender()

# Get top recommendations
games = recommender.recommend(profile, max_recommendations=5)

for game in games:
    print(f"\n{game.name} (Priority: {game.priority:.2f})")
    print(f"Duration: {game.duration_minutes} min")
    print(f"Intensity: {game.intensity}")
    print(f"{game.rationale}")

# Generate complete training plan
plan = recommender.generate_training_plan(profile, available_time=60)

print("\nTraining Plan:")
for item in plan['schedule']:
    print(f"{item['start']:02d}-{item['end']:02d} min: {item['game']}")
```

### Supported Features

- Stance preference (orthodox/southpaw/switch)
- Pressure style (aggressive/counter/balanced)
- Event frequencies (guard pulls, takedowns, strikes)
- Breath patterns (mean rate, fatigue onset)
- Movement speed and footwork

### Training Game Library

Current games include:
- Guard Pull Defense Drill
- Pressure Passing Game
- Distance Management Sparring
- Takedown Defense Rounds
- Orthodox vs Southpaw Drilling
- High-Pace Conditioning
- Breath Control Under Pressure
- Strike Defense Drill
- Counter Timing Game

---

## Timeline Replay

**Location:** `exts/zs.core/zs/core/timeline_replay.py`

### Overview

Replay recorded sessions with frame-accurate scrubbing, annotations, and event markers.

### Features

- Variable speed playback (0.1x - 10x)
- Scrubbing to any time point
- Event markers for notable moments
- User annotations
- State reconstruction at any time

### Usage

```python
from zs.core.event_log import EventLog
from zs.core.timeline_replay import TimelinePlayer

# Load event log
log = EventLog.load('sessions/session_001.jsonl')

# Create player
player = TimelinePlayer(log)

print(f"Duration: {player.end_time:.1f}s")
print(f"Markers: {len(player.markers)}")

# Play at 2x speed
player.set_speed(2.0)
player.play()

while player.is_playing:
    events = player.update(dt=0.033)  # 30fps

    # Process events
    for event in events:
        print(f"t={event.timestamp:.2f}s: {event.event_type}")

# Seek to specific time
player.seek(30.0)
state = player.get_state_at_time(30.0)

print(f"Frame number: {state['frame_number']}")
print(f"Breath rate: {state['breath_rate']} BPM")
print(f"Balance score: {state['balance_score']}")

# Add user annotations
player.add_annotation(25.0, "Good technique here", "user")
player.add_annotation(45.0, "Watch footwork", "coach")

# Export annotations
player.export_annotations('sessions/session_001_annotations.json')
```

### USD Integration

In Omniverse Kit:

```python
# Link player to USD timeline
def on_timeline_update(current_frame):
    time_seconds = current_frame / fps
    player.seek(time_seconds)

    state = player.get_state_at_time(time_seconds)

    # Update USD stage with state
    update_pose_in_stage(state['pose_keypoints'])
    update_breath_viz(state['breath_rate'])

    # Show markers
    markers = player.get_markers_in_range(
        time_seconds - 1.0,
        time_seconds + 1.0
    )

    for marker in markers:
        show_timeline_marker(marker)
```

---

## Privacy Features

**Location:** `exts/zs.core/zs/core/privacy.py`

### Overview

Face blurring and consent management for privacy-preserving video export.

### Components

1. **Face Blurrer**
   - Automatic face detection (Haar cascade)
   - Multiple blur methods (Gaussian, pixelate, blackbox)
   - Video batch processing

2. **Privacy Manager**
   - Consent tracking
   - Default blur-on-export
   - Export policies

### Usage

**1. Blur Individual Frames**

```python
from zs.core.privacy import FaceBlurrer

blurrer = FaceBlurrer(
    blur_method='gaussian',  # or 'pixelate', 'blackbox'
    blur_strength=51
)

# Auto-detect and blur
blurred_frame = blurrer.blur_frame(frame)

# Or specify face regions manually
from zs.core.privacy import FaceRegion

face = FaceRegion(bbox=(x, y, w, h), confidence=0.9)
blurred_frame = blurrer.blur_frame(frame, face_regions=[face])
```

**2. Blur Videos**

```python
def progress_callback(frame_num, total):
    print(f"Processing: {frame_num}/{total}")

blurrer.blur_video(
    input_path='recordings/session_001.mp4',
    output_path='export/session_001_blurred.mp4',
    progress_callback=progress_callback
)
```

**3. Privacy Manager**

```python
from zs.core.privacy import PrivacyManager

privacy = PrivacyManager()

# Get consent
privacy.set_consent(obtained=True, participant_id='athlete_001')

# Export with privacy policies
privacy.export_video(
    input_path='recordings/session.mp4',
    output_path='export/session.mp4',
    blur_faces=None,  # Use default policy
    blur_method='gaussian'
)

# Can export raw?
if privacy.can_export_raw():
    print("Consent obtained, raw export allowed")
else:
    print("No consent, will blur faces")
```

### Consent Workflow

1. **Before Recording:**
   - Obtain written consent from participants
   - Record consent in PrivacyManager

2. **During Recording:**
   - Store raw footage securely
   - Mark participant IDs

3. **On Export:**
   - Check consent status
   - Blur faces if no consent
   - Add watermark/disclaimer

4. **Compliance:**
   - GDPR: Right to be forgotten (delete recordings)
   - CCPA: Disclosure of data collection
   - HIPAA: Medical data safeguards (breath data)

---

## Testing

### Unit Tests

Each module includes self-tests:

```bash
# Test 3DGS loader
python exts/zs.3dgs/zs/gaussian_splats/splat_loader.py

# Test multi-camera calibration
python exts/zs.pose/zs/pose/multicam/calibration.py

# Test opponent feature extraction
python exts/zs.opponent/zs/opponent/feature_extractor.py

# Test timeline replay
python exts/zs.core/zs/core/timeline_replay.py

# Test privacy blurring
python exts/zs.core/zs/core/privacy.py
```

### Integration Tests

```bash
# Run full integration test suite
python tests/test_integration_future_work.py

# Or with pytest
pytest tests/ -v
```

### GPU Tests

```bash
python tests/test_gpu_kernels.py
```

---

## Performance Targets

| Feature | Target | Status |
|---------|--------|--------|
| GPU EVM | <10ms/frame | In progress (CPU fallback ready) |
| 3DGS depth query | <1ms | ✓ |
| Multi-cam fusion | <5ms | ✓ |
| Face detection | <20ms | ✓ |
| Timeline scrub | <16ms (60fps) | ✓ |

---

## Future Enhancements

1. **GPU EVM:**
   - Fully optimized CUDA kernels
   - Multi-stream processing
   - Mixed precision (FP16)

2. **3DGS:**
   - Real-time rendering via Omniverse RTX
   - Semantic region labeling
   - Dynamic environment updates

3. **Multi-Camera:**
   - > 2 camera support
   - Auto-calibration from known markers
   - Bundle adjustment refinement

4. **Opponent Analysis:**
   - ML-based pattern recognition
   - Video similarity search
   - Automatic highlight detection

5. **Privacy:**
   - Body anonymization (not just face)
   - Audio voice masking
   - Differential privacy for statistics

---

## References

- **3DGS:** [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Multi-view:** [Hartley & Zisserman - Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- **CUDA:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- **Privacy:** [GDPR Compliance](https://gdpr.eu/)

---

**For more information, see the main [README.md](../README.md) or [architecture.md](architecture.md)**
