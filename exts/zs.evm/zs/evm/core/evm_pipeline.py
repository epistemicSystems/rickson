"""
Complete EVM Pipeline

Combines spatial pyramids, temporal filtering, and amplification
into a single composable pipeline.
"""

import numpy as np
from typing import List, Tuple, Optional
from .pyramid import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    collapse_laplacian_pyramid,
    amplify_spatial_frequencies
)
from .temporal_filter import TemporalBufferFilter
from .breath_estimator import BreathEstimator, extract_roi_signal


class EVMPipeline:
    """
    Complete Eulerian Video Magnification pipeline.

    Processes video frames through:
    1. Spatial decomposition (Laplacian pyramid)
    2. Temporal filtering (band-pass per level)
    3. Amplification
    4. Reconstruction
    5. Breath rate estimation
    """

    def __init__(
        self,
        fps: float,
        low_freq: float,
        high_freq: float,
        alpha: float,
        pyramid_levels: int,
        buffer_seconds: float = 10.0,
        wavelength_attenuation: bool = True
    ):
        """
        Initialize EVM pipeline.

        Args:
            fps: Video framerate
            low_freq: Lower frequency cutoff (Hz)
            high_freq: Upper frequency cutoff (Hz)
            alpha: Amplification factor
            pyramid_levels: Number of pyramid levels
            buffer_seconds: Temporal buffer length
            wavelength_attenuation: Apply wavelength-dependent attenuation
        """
        self.fps = fps
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.alpha = alpha
        self.pyramid_levels = pyramid_levels
        self.wavelength_attenuation = wavelength_attenuation

        # Temporal filter for each pyramid level
        self.temporal_filters = [
            TemporalBufferFilter(fps, low_freq, high_freq, buffer_seconds)
            for _ in range(pyramid_levels)
        ]

        # Breath estimator
        self.breath_estimator = BreathEstimator(
            fps,
            buffer_seconds,
            freq_range=(low_freq, high_freq),
            method='fft'
        )

        # Statistics
        self.frame_count = 0
        self.current_breath_rate = 0.0
        self.breath_confidence = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        roi_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a single video frame through EVM pipeline.

        Args:
            frame: Input frame (H, W, C), uint8 or float32
            roi_mask: Optional mask for breath ROI

        Returns:
            (amplified_frame, metrics)
            where metrics = {
                'breath_rate_bpm': float,
                'breath_confidence': float,
                'frame_count': int
            }
        """
        # Convert to float if needed
        if frame.dtype == np.uint8:
            frame_float = frame.astype(np.float32)
        else:
            frame_float = frame.astype(np.float32)

        # 1. Build spatial pyramid
        gaussian_pyr = build_gaussian_pyramid(frame_float, self.pyramid_levels)
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)

        # 2. Add to temporal filters
        for level_idx, level_frame in enumerate(laplacian_pyr):
            self.temporal_filters[level_idx].add_frame(level_frame)

        # 3. Get temporally filtered pyramid
        filtered_pyr = []
        for level_idx in range(self.pyramid_levels):
            filtered = self.temporal_filters[level_idx].get_filtered_current()
            if filtered is not None:
                filtered_pyr.append(filtered)
            else:
                # Not enough frames yet, use zeros
                filtered_pyr.append(np.zeros_like(laplacian_pyr[level_idx]))

        # 4. Amplify
        amplified_pyr = amplify_spatial_frequencies(
            filtered_pyr,
            self.alpha,
            self.wavelength_attenuation
        )

        # 5. Add amplified signal back to original pyramid
        combined_pyr = [
            laplacian_pyr[i] + amplified_pyr[i]
            for i in range(len(laplacian_pyr))
        ]

        # 6. Reconstruct
        amplified_frame = collapse_laplacian_pyramid(combined_pyr)

        # Clip to valid range
        if frame.dtype == np.uint8:
            amplified_frame = np.clip(amplified_frame, 0, 255).astype(np.uint8)
        else:
            amplified_frame = amplified_frame.astype(np.float32)

        # 7. Extract signal for breath estimation
        # Use the filtered lowest-frequency level (coarsest)
        if len(filtered_pyr) > 0 and filtered_pyr[-1].size > 0:
            signal_val = extract_roi_signal(filtered_pyr[-1], roi_mask)
            self.breath_estimator.add_frame_signal(signal_val)

        # 8. Estimate breath rate
        self.current_breath_rate, self.breath_confidence = \
            self.breath_estimator.estimate()

        self.frame_count += 1

        metrics = {
            'breath_rate_bpm': self.current_breath_rate,
            'breath_confidence': self.breath_confidence,
            'frame_count': self.frame_count
        }

        return amplified_frame, metrics

    def update_params(
        self,
        alpha: Optional[float] = None,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None
    ):
        """
        Update pipeline parameters.

        Note: Changing frequencies will reset temporal filters.

        Args:
            alpha: New amplification factor
            low_freq: New low frequency cutoff
            high_freq: New high frequency cutoff
        """
        if alpha is not None:
            self.alpha = alpha

        if low_freq is not None or high_freq is not None:
            if low_freq is not None:
                self.low_freq = low_freq
            if high_freq is not None:
                self.high_freq = high_freq

            # Reset filters with new frequencies
            buffer_seconds = self.temporal_filters[0].buffer_size / self.fps
            self.temporal_filters = [
                TemporalBufferFilter(self.fps, self.low_freq, self.high_freq, buffer_seconds)
                for _ in range(self.pyramid_levels)
            ]

            # Reset breath estimator
            self.breath_estimator = BreathEstimator(
                self.fps,
                buffer_seconds,
                freq_range=(self.low_freq, self.high_freq),
                method='fft'
            )

    def reset(self):
        """Reset pipeline state."""
        for filt in self.temporal_filters:
            filt.reset()
        self.breath_estimator.reset()
        self.frame_count = 0
        self.current_breath_rate = 0.0
        self.breath_confidence = 0.0

    def get_metrics(self) -> dict:
        """Get current metrics without processing a frame."""
        return {
            'breath_rate_bpm': self.current_breath_rate,
            'breath_confidence': self.breath_confidence,
            'frame_count': self.frame_count
        }


def test_evm_pipeline():
    """Test full EVM pipeline with synthetic video."""
    print("Testing EVM Pipeline...")

    # Parameters
    fps = 30.0
    duration = 5.0
    num_frames = int(fps * duration)
    H, W, C = 128, 128, 3

    # Create synthetic video with breathing motion
    breath_freq = 0.3  # Hz (18 BPM)
    t = np.arange(num_frames) / fps
    breath_signal = np.sin(2 * np.pi * breath_freq * t)

    frames = []
    for amp in breath_signal:
        frame = np.ones((H, W, C), dtype=np.float32) * 128
        # Add breathing pattern to center region
        h_start, h_end = H // 3, 2 * H // 3
        w_start, w_end = W // 3, 2 * W // 3
        frame[h_start:h_end, w_start:w_end] += amp * 5  # Subtle motion
        frames.append(frame)

    # Create pipeline
    pipeline = EVMPipeline(
        fps=fps,
        low_freq=0.2,
        high_freq=0.5,
        alpha=15.0,
        pyramid_levels=4
    )

    # Process frames
    print(f"Processing {num_frames} frames...")
    for i, frame in enumerate(frames):
        amplified, metrics = pipeline.process_frame(frame)

        if (i + 1) % 30 == 0:
            print(f"  Frame {i+1}/{num_frames}: "
                  f"Breath={metrics['breath_rate_bpm']:.1f} BPM, "
                  f"Conf={metrics['breath_confidence']:.2f}")

    # Final estimate
    final_metrics = pipeline.get_metrics()
    estimated_bpm = final_metrics['breath_rate_bpm']
    true_bpm = breath_freq * 60

    print(f"\nResults:")
    print(f"  True breath rate: {true_bpm:.1f} BPM")
    print(f"  Estimated: {estimated_bpm:.1f} BPM")
    print(f"  Error: {abs(estimated_bpm - true_bpm):.1f} BPM")

    success = abs(estimated_bpm - true_bpm) < 3.0
    print(f"  Result: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    test_evm_pipeline()
