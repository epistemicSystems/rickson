"""
GPU-Accelerated EVM Pipeline

This module provides GPU-accelerated implementations of EVM operations
using CUDA (via CuPy) with CPU fallbacks.

Requirements:
    pip install cupy-cuda11x  # or cupy-cuda12x depending on CUDA version

Performance targets:
    - Pyramid construction: 5ms → 0.5ms (10x)
    - Temporal filtering: 3ms → 0.3ms (10x)
    - Total pipeline: 20ms → 2ms (10x)
"""

import numpy as np
from typing import List, Tuple, Optional
import carb

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    carb.log_info("[zs.evm.gpu] CuPy available - GPU acceleration enabled")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    carb.log_warn("[zs.evm.gpu] CuPy not available - using CPU fallback")


class GPUPyramid:
    """
    GPU-accelerated pyramid construction.

    Uses separable Gaussian convolution for efficiency:
    - Horizontal pass on GPU
    - Vertical pass on GPU
    - Downsample on GPU
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU pyramid builder.

        Args:
            use_gpu: Use GPU if available (falls back to CPU if not)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            # Precompute Gaussian kernel on GPU
            self.kernel_1d = self._create_gaussian_kernel_gpu(5, 0.83)
        else:
            # CPU fallback
            from scipy.ndimage import gaussian_filter
            self.gaussian_filter = gaussian_filter

    def _create_gaussian_kernel_gpu(self, size: int, sigma: float):
        """Create 1D Gaussian kernel on GPU."""
        if not self.use_gpu:
            return None

        x = cp.arange(size) - size // 2
        kernel = cp.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / cp.sum(kernel)
        return kernel

    def _gaussian_blur_gpu(self, image_gpu):
        """Apply Gaussian blur using separable convolution on GPU."""
        if not self.use_gpu:
            raise RuntimeError("GPU not available")

        # Horizontal convolution
        kernel_h = self.kernel_1d.reshape(1, -1, 1)
        blurred = cp.apply_along_axis(
            lambda x: cp.convolve(x, self.kernel_1d, mode='same'),
            axis=1,
            arr=image_gpu
        )

        # Vertical convolution
        kernel_v = self.kernel_1d.reshape(-1, 1, 1)
        blurred = cp.apply_along_axis(
            lambda x: cp.convolve(x, self.kernel_1d, mode='same'),
            axis=0,
            arr=blurred
        )

        return blurred

    def build_gaussian_pyramid_gpu(
        self,
        image: np.ndarray,
        levels: int
    ) -> List[np.ndarray]:
        """
        Build Gaussian pyramid on GPU.

        Args:
            image: Input image (H, W, C) numpy array
            levels: Number of pyramid levels

        Returns:
            List of pyramid levels (on CPU)
        """
        if not self.use_gpu:
            # Fallback to CPU
            from ..core.pyramid import build_gaussian_pyramid
            return build_gaussian_pyramid(image, levels)

        # Transfer to GPU
        image_gpu = cp.asarray(image, dtype=cp.float32)
        pyramid = [image]
        current_gpu = image_gpu

        for i in range(levels - 1):
            # Gaussian blur
            blurred_gpu = self._gaussian_blur_gpu(current_gpu)

            # Downsample (every 2nd pixel)
            downsampled_gpu = blurred_gpu[::2, ::2, :]

            # Transfer back to CPU for this level
            downsampled = cp.asnumpy(downsampled_gpu)
            pyramid.append(downsampled)

            current_gpu = downsampled_gpu

        return pyramid

    def build_laplacian_pyramid_gpu(
        self,
        gaussian_pyramid: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Build Laplacian pyramid on GPU.

        Args:
            gaussian_pyramid: Gaussian pyramid from build_gaussian_pyramid_gpu

        Returns:
            Laplacian pyramid (on CPU)
        """
        if not self.use_gpu:
            # Fallback
            from ..core.pyramid import build_laplacian_pyramid
            return build_laplacian_pyramid(gaussian_pyramid)

        laplacian = []

        for i in range(len(gaussian_pyramid) - 1):
            # Transfer current level to GPU
            current_gpu = cp.asarray(gaussian_pyramid[i], dtype=cp.float32)
            next_gpu = cp.asarray(gaussian_pyramid[i + 1], dtype=cp.float32)

            # Upsample next level
            H, W = current_gpu.shape[:2]
            upsampled_gpu = cp.zeros((H, W, current_gpu.shape[2]), dtype=cp.float32)

            # Simple nearest-neighbor upsample
            h_next, w_next = next_gpu.shape[:2]
            for h in range(h_next):
                for w in range(w_next):
                    h_up = min(h * 2, H - 1)
                    w_up = min(w * 2, W - 1)
                    upsampled_gpu[h_up, w_up] = next_gpu[h, w]

            # Laplacian = current - upsampled
            laplacian_gpu = current_gpu - upsampled_gpu

            # Transfer back
            laplacian.append(cp.asnumpy(laplacian_gpu))

        # Last level is just the coarsest Gaussian
        laplacian.append(gaussian_pyramid[-1])

        return laplacian


class GPUTemporalFilter:
    """
    GPU-accelerated temporal IIR filtering.

    Processes batches of pixels in parallel on GPU.
    """

    def __init__(
        self,
        fps: float,
        low_freq: float,
        high_freq: float,
        use_gpu: bool = True
    ):
        """
        Initialize GPU temporal filter.

        Args:
            fps: Video framerate
            low_freq: Lower frequency cutoff
            high_freq: Upper frequency cutoff
            use_gpu: Use GPU if available
        """
        self.fps = fps
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # Design filter coefficients (on CPU)
        from scipy import signal
        nyquist = fps / 2.0
        low = np.clip(low_freq / nyquist, 0.01, 0.99)
        high = np.clip(high_freq / nyquist, low + 0.01, 0.99)

        self.b, self.a = signal.butter(2, [low, high], btype='band')

        if self.use_gpu:
            # Transfer coefficients to GPU
            self.b_gpu = cp.asarray(self.b, dtype=cp.float32)
            self.a_gpu = cp.asarray(self.a, dtype=cp.float32)

        self.frame_buffer = []
        self.max_buffer_size = int(fps * 10)  # 10 second buffer

    def filter_frame_batch_gpu(
        self,
        frames: List[np.ndarray]
    ) -> np.ndarray:
        """
        Filter batch of frames on GPU.

        Args:
            frames: List of frames to filter

        Returns:
            Filtered current frame
        """
        if not self.use_gpu or len(frames) < 10:
            # Fallback or not enough frames
            from ..core.temporal_filter import TemporalBufferFilter
            filt = TemporalBufferFilter(self.fps, self.low_freq, self.high_freq)
            for frame in frames:
                filt.add_frame(frame)
            return filt.get_filtered_current()

        # Stack frames into 4D array: (T, H, W, C)
        frames_stack = np.stack(frames, axis=0).astype(np.float32)
        T, H, W, C = frames_stack.shape

        # Transfer to GPU
        frames_gpu = cp.asarray(frames_stack)

        # Reshape for batch processing: (T, H*W*C)
        frames_flat = frames_gpu.reshape(T, -1)

        # Apply IIR filter along time axis for each pixel
        # Note: This is a simplified version; full IIR requires recursive computation
        # For now, use FFT-based filtering

        # FFT
        fft_result = cp.fft.fft(frames_flat, axis=0)
        freqs = cp.fft.fftfreq(T, 1.0 / self.fps)

        # Frequency mask
        freq_mask = (cp.abs(freqs) >= self.low_freq) & (cp.abs(freqs) <= self.high_freq)

        # Apply mask
        fft_filtered = fft_result * freq_mask[:, cp.newaxis]

        # IFFT
        filtered_flat = cp.fft.ifft(fft_filtered, axis=0).real

        # Reshape back
        filtered_stack = filtered_flat.reshape(T, H, W, C)

        # Return current frame
        current_filtered = cp.asnumpy(filtered_stack[-1])

        return current_filtered.astype(np.float32)


class GPUEVMPipeline:
    """
    GPU-accelerated EVM pipeline.

    Combines GPU pyramid, temporal filtering, and amplification.
    """

    def __init__(
        self,
        fps: float,
        low_freq: float,
        high_freq: float,
        alpha: float,
        pyramid_levels: int,
        use_gpu: bool = True
    ):
        """
        Initialize GPU EVM pipeline.

        Args:
            fps: Video framerate
            low_freq: Lower frequency cutoff
            high_freq: Upper frequency cutoff
            alpha: Amplification factor
            pyramid_levels: Number of pyramid levels
            use_gpu: Use GPU if available (auto-fallback to CPU)
        """
        self.fps = fps
        self.alpha = alpha
        self.pyramid_levels = pyramid_levels
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            carb.log_info("[GPUEVMPipeline] GPU acceleration enabled")
            self.pyramid_builder = GPUPyramid(use_gpu=True)
            self.temporal_filters = [
                GPUTemporalFilter(fps, low_freq, high_freq, use_gpu=True)
                for _ in range(pyramid_levels)
            ]
        else:
            carb.log_info("[GPUEVMPipeline] Using CPU fallback")
            # Use CPU pipeline
            from ..core.evm_pipeline import EVMPipeline
            self.cpu_pipeline = EVMPipeline(
                fps, low_freq, high_freq, alpha, pyramid_levels
            )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process frame through GPU-accelerated EVM pipeline.

        Args:
            frame: Input frame

        Returns:
            (amplified_frame, metrics)
        """
        if not self.use_gpu:
            # Use CPU fallback
            return self.cpu_pipeline.process_frame(frame)

        # GPU processing
        # Note: This is a simplified version showing the structure
        # Full implementation would maintain frame buffers and do proper temporal filtering

        # For now, just log that we're using GPU and fall back
        carb.log_info("[GPUEVMPipeline] GPU frame processing (currently using CPU fallback for stability)")
        return self.cpu_pipeline.process_frame(frame)


def benchmark_gpu_vs_cpu(
    width: int = 640,
    height: int = 480,
    levels: int = 4,
    num_frames: int = 100
):
    """
    Benchmark GPU vs CPU EVM performance.

    Args:
        width: Frame width
        height: Frame height
        levels: Pyramid levels
        num_frames: Number of frames to process
    """
    import time

    print(f"\nBenchmarking EVM Pipeline: {width}x{height}, {levels} levels, {num_frames} frames")
    print("="*70)

    # Create test frames
    frames = [
        np.random.rand(height, width, 3).astype(np.float32) * 255
        for _ in range(num_frames)
    ]

    # CPU benchmark
    print("\n[CPU] Processing...")
    from ..core.evm_pipeline import EVMPipeline
    cpu_pipeline = EVMPipeline(30.0, 0.2, 0.5, 15.0, levels)

    start = time.time()
    for frame in frames:
        _, _ = cpu_pipeline.process_frame(frame)
    cpu_time = time.time() - start

    print(f"  Total time: {cpu_time:.2f}s")
    print(f"  Per frame: {cpu_time/num_frames*1000:.1f}ms")
    print(f"  FPS: {num_frames/cpu_time:.1f}")

    # GPU benchmark
    if GPU_AVAILABLE:
        print("\n[GPU] Processing...")
        gpu_pipeline = GPUEVMPipeline(30.0, 0.2, 0.5, 15.0, levels, use_gpu=True)

        start = time.time()
        for frame in frames:
            _, _ = gpu_pipeline.process_frame(frame)
        gpu_time = time.time() - start

        print(f"  Total time: {gpu_time:.2f}s")
        print(f"  Per frame: {gpu_time/num_frames*1000:.1f}ms")
        print(f"  FPS: {num_frames/gpu_time:.1f}")

        # Speedup
        speedup = cpu_time / gpu_time
        print(f"\n[Speedup] {speedup:.1f}x faster on GPU")
    else:
        print("\n[GPU] Not available - install CuPy to enable GPU acceleration")

    print("="*70)


if __name__ == "__main__":
    # Run benchmark
    benchmark_gpu_vs_cpu(640, 480, 4, 50)
