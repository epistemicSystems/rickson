# Rickson Architecture

## Overview

Rickson is a local-first BJJ/Muay Thai training assistant built on NVIDIA Omniverse. It combines markerless motion capture, Eulerian Video Magnification for breath analysis, and 3D Gaussian Splatting for spatial priors.

## Architecture Principles

### Rich Hickey (Simplicity)

- **Data-oriented design**: Everything is data; events are immutable
- **Pure transforms**: IO at edges; computation in the middle
- **Explicit state**: No hidden globals; state derived from events
- **Simplicity over ease**: Prefer composable primitives over monolithic features

### Bret Victor (Immediate Feedback)

- **Scrubbable parameters**: Every value can be adjusted in real-time
- **Visible causality**: Click any metric to see its derivation graph
- **Direct manipulation**: No edit-compile-run; change and see immediately
- **Freeze-frame lab**: Pause time, explore space, re-compute

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Omniverse Kit App                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   RTX Viewport                         │ │
│  │  - Rendered USD stage (gym, athlete, overlays)        │ │
│  │  - Real-time ray-traced visualization                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │   zs.ui         │  │   zs.evm (OmniGraph)             │ │
│  │   Extension     │  │                                  │ │
│  │                 │  │  ┌────────────────────────────┐  │ │
│  │  ┌───────────┐  │  │  │  EVMBandPass Node          │  │ │
│  │  │ UI Panel  │──┼──┼─▶│  - Temporal filter         │  │ │
│  │  │ Sliders   │  │  │  │  - Breath estimation       │  │ │
│  │  │ Metrics   │◀─┼──┼──│  - GPU acceleration        │  │ │
│  │  │ Insights  │  │  │  └────────────────────────────┘  │ │
│  │  └───────────┘  │  │                                  │ │
│  └─────────────────┘  └──────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              OpenUSD Stage (omni.usd)                  │ │
│  │                                                        │ │
│  │  /World/CameraRig        - Multi-cam setup            │ │
│  │  /World/TrainingSpace    - Gym floor, walls           │ │
│  │  /World/GymScanPlaceholder - 3DGS environment         │ │
│  │  /World/AthletePoseOrigin  - Pose keypoints          │ │
│  │  /World/HUD                - Overlays, metrics        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

           ▲                                 ▼
           │                                 │
    ┌──────┴──────────┐           ┌─────────┴─────────┐
    │  Camera Input   │           │   Event Log       │
    │  (Multi-cam)    │           │   (EDN/JSON)      │
    └─────────────────┘           └───────────────────┘
           ▲                                 ▼
           │                                 │
    ┌──────┴──────────┐           ┌─────────┴─────────┐
    │  Video Files    │           │   Session DB      │
    │  (.mp4, .mov)   │           │   (Derived State) │
    └─────────────────┘           └───────────────────┘
```

## Data Flow

### 1. Capture Pipeline

```
Video Frames → Pose Estimation → 3D Fusion → USD Scene Update
     ↓              ↓                ↓              ↓
  EVM Pass → Breath Analysis → Event Log → UI Update
```

### 2. Event-Driven State

All state changes are events (append-only):

```edn
{:event/type :session-started
 :event/timestamp 1698765432000
 :session/id "sess-001"
 :athlete/id "athlete-123"}

{:event/type :frame-ingested
 :event/timestamp 1698765432033
 :frame/number 1
 :frame/source :camera-main}

{:event/type :pose-estimated
 :event/timestamp 1698765432056
 :frame/number 1
 :pose/keypoints [...]}

{:event/type :evm-breath-cycle
 :event/timestamp 1698765434120
 :breath/rate 18.5
 :breath/confidence 0.92}

{:event/type :alert-balance-edge
 :event/timestamp 1698765435200
 :balance/score 42
 :alert/severity :warning}
```

Current session state is **derived** by reducing events.

### 3. OmniGraph Compute Graph

```
[Video Frames] → [Spatial Pyramid] → [Temporal Filter] → [Amplify] → [Output]
                        ↓                    ↓                ↓
                   [Level 0-N]      [Band-Pass IIR]    [Alpha Gain]
                                           ↓
                                    [Peak Detection] → [Breath Rate]
```

## Extension Architecture

### zs.ui Extension

**Purpose:** Main UI panel with scrubbable controls

**Components:**
- `ui_panel.py`: Main panel class with sliders, metrics, buttons
- `__init__.py`: Extension lifecycle (startup/shutdown)

**Responsibilities:**
- Display EVM parameters (alpha, freq band, levels)
- Show live metrics (breath rate, balance)
- Provide "Explain" derivation view
- Control capture pipeline (start/pause/reset)

**Communication:**
- Sends parameter changes to OmniGraph nodes
- Receives metric updates via USD attributes or events
- Publishes user actions as events

### zs.evm Extension

**Purpose:** EVM compute as OmniGraph nodes

**Components:**
- `nodes/EVMBandPass.py`: Band-pass filter node implementation
- `nodes/EVMBandPass.ogn`: Node definition (inputs/outputs)
- `__init__.py`: Extension lifecycle

**Responsibilities:**
- Temporal band-pass filtering (CPU reference, GPU accelerated)
- Spatial pyramid construction
- Breath rate estimation via peak detection
- Micro-motion amplification

**Optimization Path:**
1. CPU reference (NumPy) ← **current**
2. CUDA kernel for IIR filter
3. RTX compute shader for pyramid
4. End-to-end GPU pipeline

## USD Scene Structure

```
World (Xform)
├── TrainingSpace (Xform)
│   ├── Floor (Mesh)
│   ├── AthletePoseOrigin (Xform)
│   │   ├── Skeleton (custom prim with keypoints)
│   │   └── SupportPolygon (Mesh - dynamic)
│   └── GymScanPlaceholder (PointInstancer for 3DGS)
│       └── Splats (custom prim or .ply reference)
├── CameraRig (Xform)
│   ├── MainCamera (Camera)
│   ├── LeftCamera (Camera)
│   └── RightCamera (Camera)
├── Lighting (Xform)
│   ├── SkyDome (DomeLight)
│   ├── KeyLight (RectLight)
│   └── FillLight (RectLight)
└── HUD (Xform)
    ├── BreathOverlay (custom prim)
    ├── BalanceVector (Line prim)
    └── AlertAnnotations (Point/Text prims)
```

**USD Advantages:**
- Time-varying attributes (keyframe pose over timeline)
- Layer composition (base gym + athlete + overlays)
- Efficient streaming (only changed prims updated)
- Interoperability (export to other DCC tools)

## Performance Considerations

### Latency Budget (per frame @ 30fps)

| Stage                | Budget | Current | Notes                           |
|----------------------|--------|---------|----------------------------------|
| Frame capture        | 5ms    | TBD     | Multi-cam sync                  |
| Pose estimation      | 10ms   | TBD     | GPU inference                   |
| EVM band-pass        | 5ms    | TBD     | CUDA kernel                     |
| 3D fusion            | 5ms    | TBD     | Constrain with splat prior      |
| USD update           | 3ms    | TBD     | Delta updates only              |
| RTX render           | 5ms    | TBD     | Depends on scene complexity     |
| **Total**            | **33ms** | TBD   | **Target: <33ms (30fps real-time)** |

### Memory Budget

- **Video frames buffer**: 1GB (60 frames @ 1920x1080 RGBA)
- **Spatial pyramids**: 500MB (4 levels, double-buffer)
- **USD stage**: 100MB (static geometry + dynamic pose)
- **3DGS splat cloud**: 200MB (gym scan)
- **Event log**: 50MB/hour (append-only)
- **Total**: ~2GB active memory

### GPU Utilization

- **Pose estimation**: 40% (inference)
- **EVM compute**: 30% (band-pass + pyramid)
- **RTX rendering**: 20% (viewport)
- **Headroom**: 10% (for burst processing)

## Security & Privacy

### Local-First Guarantees

- **No cloud dependency**: All computation local
- **Consent by default**: Faces blurred on export
- **Data ownership**: User controls all data
- **Network isolation**: Can run air-gapped

### Event Log Security

- **Immutable**: Events never modified, only appended
- **Auditable**: Full history of session
- **Exportable**: EDN/JSON for portability
- **Anonymizable**: Strip PII before sharing

## Testing Strategy

### Unit Tests

- Pure functions (EVM kernels, pose transforms)
- USD scene validation
- Event reducers (state derivation)

### Integration Tests

- OmniGraph node wiring
- UI ↔ compute communication
- Extension lifecycle

### Golden Image Tests

- Visual regression (EVM output, pose overlays)
- Performance benchmarks (latency, GPU util)

### End-to-End Tests

- Full capture → analysis → replay pipeline
- Multi-cam synchronization
- Export with face blurring

## Future Architecture

### Milestone 3+: Pose Overlay

Add extension: `zs.pose`
- MediaPipe or OpenPose integration
- Keypoint drawing in viewport
- Support polygon computation

### Milestone 4+: 3DGS Integration

Add extension: `zs.spatial`
- Load 3DGS gym scans (via community extension)
- Multi-cam calibration tools
- Constrained 3D pose fusion

### Milestone 5+: Insights Engine

Add extension: `zs.insights`
- Breath cadence analysis (frequency domain)
- Breath-hold detection (threshold crossing)
- Balance edge alerts (COM vs. support)
- "Explain" OmniGraph visualization

### Milestone 6+: Opponent Analysis

Add extension: `zs.scouting`
- Offline feature extraction
- Training game suggestions
- Side-by-side comparison view

### Milestone 7+: Record/Replay

Enhance event log:
- USD timeline integration
- Annotation tools (text, arrows, regions)
- Export with privacy controls

## References

- [Omniverse Kit Architecture](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/)
- [USD Scene Graph](https://openusd.org/release/index.html)
- [OmniGraph Programming Guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/tutorials.html)
- [EVM Paper (MIT CSAIL)](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
