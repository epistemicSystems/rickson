---
description: "Run GPU kernel tests with golden image comparison"
---

# Test: GPU Kernels

Run tests for EVM GPU kernels with golden image comparison for visual regression.

## Test Suite

### 1. EVM Band-Pass Filter Tests

**Test file:** `tests/test_evm_kernel.py`

Tests:
- Temporal band-pass filter correctness
- Breath rate estimation accuracy
- Spatial pyramid construction
- GPU vs CPU reference comparison
- Performance benchmarks

### 2. Golden Image Tests

Compare output frames against reference images to detect visual regressions.

**Test file:** `tests/test_golden_images.py`

Tests:
- EVM amplified output matches golden reference
- Breath visualization overlay matches golden reference
- Pose keypoint rendering matches golden reference

## Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_evm_kernel.py

# With coverage
pytest tests/ --cov=exts --cov-report=html

# Golden image comparison
pytest tests/ --golden-compare

# Update golden images (when intentional changes are made)
pytest tests/ --golden-update
```

## Test Structure

```
tests/
├── test_evm_kernel.py          # EVM computation tests
├── test_ui_interactions.py     # UI panel tests
├── test_usd_stage.py           # USD stage validation
├── golden/                     # Golden reference images
│   ├── evm_breath_001.png
│   ├── evm_breath_002.png
│   └── pose_overlay_001.png
└── fixtures/                   # Test data
    ├── sample_video.mp4
    └── calibration_data.json
```

## GPU Test Requirements

- NVIDIA GPU (for CUDA kernels)
- CUDA Toolkit installed
- `pytest-gpu` plugin

## Expected Output

```
tests/test_evm_kernel.py ..................... [ 45%]
tests/test_ui_interactions.py ............... [ 70%]
tests/test_usd_stage.py ..................... [100%]

====== 42 passed, 0 failed, 2 skipped in 3.42s ======
```

## Performance Benchmarks

Tests also generate performance reports:

```
EVM Band-Pass Filter (1920x1080, 30fps):
  CPU Reference: 42.3 ms/frame
  GPU CUDA:      2.1 ms/frame (20x speedup)

Breath Rate Estimation:
  Latency: 15 ms
  Accuracy: 95.2% (vs ground truth)
```

## Golden Image Workflow

1. **Initial creation:**
   ```bash
   pytest tests/ --golden-update
   ```

2. **Validate after changes:**
   ```bash
   pytest tests/ --golden-compare
   ```

3. **Review differences:**
   - Failed tests generate diff images in `tests/golden/diffs/`
   - Visually inspect differences
   - If intentional, update golden images

## CI Integration

Tests run automatically on:
- Every commit (fast tests only)
- Pull requests (full suite + golden images)
- Nightly (with performance benchmarks)

## Next Steps

After passing tests:
1. Review performance metrics
2. Check for any warnings or deprecations
3. Update documentation if APIs changed
4. Commit golden images if intentionally updated
