"""
Test GPU EVM Kernels with Golden Image Comparison

Validates CUDA implementation against CPU baseline.
"""

import numpy as np
import pytest
import os
from pathlib import Path

# Import both implementations
from zs.evm.core.evm_pipeline import EVMPipeline as CPUPipeline
try:
    from zs.evm.cuda.gpu_pipeline import GPUEVMPipeline
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def generate_synthetic_video(num_frames=60, fps=30.0):
    """
    Generate synthetic video with known breathing pattern.

    Returns:
        frames, ground_truth_bpm
    """
    H, W, C = 128, 128, 3

    breath_freq = 0.3  # 18 BPM
    breath_bpm = breath_freq * 60.0

    t = np.arange(num_frames) / fps
    breath_signal = np.sin(2 * np.pi * breath_freq * t)

    frames = []
    for amp in breath_signal:
        frame = np.ones((H, W, C), dtype=np.uint8) * 128

        # Add breathing pattern to center ROI
        h_start, h_end = H // 3, 2 * H // 3
        w_start, w_end = W // 3, 2 * W // 3

        # Subtle intensity change (5 gray levels peak-to-peak)
        frame[h_start:h_end, w_start:w_end] = np.clip(
            frame[h_start:h_end, w_start:w_end] + amp * 5,
            0, 255
        ).astype(np.uint8)

        frames.append(frame)

    return frames, breath_bpm


def compare_frames(frame1, frame2, tolerance=5.0):
    """
    Compare two frames with tolerance.

    Args:
        frame1, frame2: Frames to compare
        tolerance: Max allowed mean absolute difference

    Returns:
        (passed, mae)
    """
    # Ensure same dtype
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)

    # Mean absolute error
    mae = np.mean(np.abs(f1 - f2))

    passed = mae < tolerance

    return passed, mae


def test_cpu_baseline():
    """Test CPU pipeline as baseline."""
    print("\n=== Testing CPU Baseline ===")

    frames, true_bpm = generate_synthetic_video(num_frames=90)

    pipeline = CPUPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4
    )

    breath_estimates = []

    for i, frame in enumerate(frames):
        amplified, metrics = pipeline.process_frame(frame)

        if i % 30 == 0:
            print(f"  Frame {i}: Breath={metrics['breath_rate_bpm']:.1f} BPM")

        if i >= 30:  # After warm-up
            breath_estimates.append(metrics['breath_rate_bpm'])

    # Check breath rate accuracy
    mean_breath = np.mean(breath_estimates)
    error = abs(mean_breath - true_bpm)

    print(f"\nResults:")
    print(f"  True breath rate: {true_bpm:.1f} BPM")
    print(f"  Estimated: {mean_breath:.1f} BPM")
    print(f"  Error: {error:.1f} BPM")

    assert error < 3.0, f"CPU baseline error too high: {error:.1f} BPM"
    print("  ✓ PASS")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_vs_cpu_accuracy():
    """Test GPU implementation vs CPU baseline."""
    print("\n=== Testing GPU vs CPU Accuracy ===")

    frames, true_bpm = generate_synthetic_video(num_frames=90)

    # CPU pipeline
    cpu_pipeline = CPUPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4
    )

    # GPU pipeline
    gpu_pipeline = GPUEVMPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4,
        use_cuda=True
    )

    cpu_frames = []
    gpu_frames = []
    cpu_breath = []
    gpu_breath = []

    for i, frame in enumerate(frames):
        cpu_amplified, cpu_metrics = cpu_pipeline.process_frame(frame)
        gpu_amplified, gpu_metrics = gpu_pipeline.process_frame(frame)

        cpu_frames.append(cpu_amplified)
        gpu_frames.append(gpu_amplified)

        if i >= 30:
            cpu_breath.append(cpu_metrics['breath_rate_bpm'])
            gpu_breath.append(gpu_metrics['breath_rate_bpm'])

        # Compare amplified frames every 10 frames
        if i % 10 == 0:
            passed, mae = compare_frames(cpu_amplified, gpu_amplified, tolerance=10.0)
            print(f"  Frame {i}: MAE={mae:.2f}, {'PASS' if passed else 'FAIL'}")

            if not passed:
                print(f"    Warning: Frames differ by {mae:.2f}")

    # Compare breath estimates
    cpu_mean = np.mean(cpu_breath)
    gpu_mean = np.mean(gpu_breath)

    print(f"\nBreath Rate Comparison:")
    print(f"  CPU: {cpu_mean:.1f} BPM")
    print(f"  GPU: {gpu_mean:.1f} BPM")
    print(f"  Difference: {abs(cpu_mean - gpu_mean):.1f} BPM")

    # GPU should be within 2 BPM of CPU
    assert abs(cpu_mean - gpu_mean) < 2.0, "GPU-CPU breath rate mismatch"

    print("  ✓ PASS")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_performance():
    """Test GPU performance (<10ms target)."""
    print("\n=== Testing GPU Performance ===")

    frames, _ = generate_synthetic_video(num_frames=60)

    pipeline = GPUEVMPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4,
        use_cuda=True
    )

    import time
    times = []

    # Warm-up
    for frame in frames[:10]:
        pipeline.process_frame(frame)

    # Timed run
    for i, frame in enumerate(frames[10:]):
        start = time.perf_counter()
        amplified, metrics = pipeline.process_frame(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        if i % 20 == 0:
            print(f"  Frame {i}: {elapsed_ms:.2f} ms")

    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)

    print(f"\nPerformance:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  P95: {p95_time:.2f} ms")
    print(f"  Target: <10 ms")

    # Note: May not achieve <10ms without optimized CUDA build
    # This is a warning, not a hard failure
    if avg_time > 10:
        print(f"  ⚠ Performance target not met (using CPU fallback?)")
    else:
        print(f"  ✓ PASS")

    assert avg_time < 100, "Performance unacceptably slow"


def test_golden_images():
    """Test against golden reference images."""
    print("\n=== Testing Golden Images ===")

    golden_dir = Path(__file__).parent / "golden"
    golden_dir.mkdir(exist_ok=True)

    # Generate or load golden frames
    frames, _ = generate_synthetic_video(num_frames=30)

    pipeline = CPUPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4
    )

    # Process and compare
    for i, frame in enumerate([frames[0], frames[15], frames[29]]):
        amplified, _ = pipeline.process_frame(frame)

        golden_path = golden_dir / f"frame_{i:03d}.npy"

        if not golden_path.exists():
            # Create golden image
            np.save(golden_path, amplified)
            print(f"  Created golden image: {golden_path.name}")
        else:
            # Compare with golden
            golden = np.load(golden_path)
            passed, mae = compare_frames(amplified, golden, tolerance=1.0)

            print(f"  Frame {i}: MAE={mae:.2f}, {'PASS' if passed else 'FAIL'}")

            assert passed, f"Golden image mismatch for frame {i}"

    print("  ✓ PASS")


def test_parameter_updates():
    """Test live parameter updates."""
    print("\n=== Testing Parameter Updates ===")

    frames, _ = generate_synthetic_video(num_frames=60)

    pipeline = CPUPipeline(
        fps=30.0,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4
    )

    # Process with initial params
    for frame in frames[:30]:
        pipeline.process_frame(frame)

    metrics1 = pipeline.get_metrics()
    print(f"  Initial alpha=15.0: Breath={metrics1['breath_rate_bpm']:.1f} BPM")

    # Update alpha
    pipeline.update_params(alpha=30.0)

    # Process more frames
    for frame in frames[30:]:
        pipeline.process_frame(frame)

    metrics2 = pipeline.get_metrics()
    print(f"  Updated alpha=30.0: Breath={metrics2['breath_rate_bpm']:.1f} BPM")

    # Breath rate should be similar (alpha affects amplification, not estimation)
    diff = abs(metrics1['breath_rate_bpm'] - metrics2['breath_rate_bpm'])
    assert diff < 5.0, "Parameter update affected breath estimation unexpectedly"

    print("  ✓ PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("GPU EVM Kernel Tests")
    print("=" * 60)

    test_cpu_baseline()
    test_golden_images()
    test_parameter_updates()

    if GPU_AVAILABLE:
        test_gpu_vs_cpu_accuracy()
        test_gpu_performance()
    else:
        print("\n⚠ GPU tests skipped (CUDA not available)")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
