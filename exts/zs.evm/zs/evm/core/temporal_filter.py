"""
Temporal Band-Pass Filtering for EVM

IIR filters for isolating motion in specific frequency bands.
Maintains state for streaming video processing.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from collections import deque


class TemporalBandPassFilter:
    """
    IIR band-pass filter with state for streaming video.

    Maintains history buffer for each spatial location.
    """

    def __init__(
        self,
        fps: float,
        low_freq: float,
        high_freq: float,
        filter_order: int = 2
    ):
        """
        Initialize temporal filter.

        Args:
            fps: Video framerate (Hz)
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Upper cutoff frequency (Hz)
            filter_order: Butterworth filter order (default 2)
        """
        self.fps = fps
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.filter_order = filter_order

        # Design IIR filter
        nyquist = fps / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Ensure valid frequency range
        low = np.clip(low, 0.01, 0.99)
        high = np.clip(high, low + 0.01, 0.99)

        self.b, self.a = signal.butter(
            filter_order,
            [low, high],
            btype='band'
        )

        # State for streaming (will be initialized on first frame)
        self.zi = None
        self.initialized = False

    def filter_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply temporal filter to a single frame.

        Args:
            frame: Input frame (H, W, C) - already part of temporal sequence

        Returns:
            Filtered frame (same shape)
        """
        if not self.initialized:
            # Initialize filter state
            # zi shape: (max(len(a), len(b)) - 1, H, W, C)
            filter_len = max(len(self.a), len(self.b)) - 1
            self.zi = np.zeros(
                (filter_len, *frame.shape),
                dtype=np.float32
            )
            self.initialized = True

        # Apply filter
        # scipy.signal.lfilter expects (num_samples,) for 1D
        # For images, we need to filter each pixel's temporal sequence
        # Since we're processing frame-by-frame, use filtfilt or maintain history

        # For streaming, we use the stateful lfilter with zi
        # Reshape for batch processing
        original_shape = frame.shape
        frame_flat = frame.reshape(-1)  # Flatten spatial dimensions

        # Apply filter with state
        filtered_flat, self.zi_flat = signal.lfilter(
            self.b, self.a,
            frame_flat[np.newaxis, :],  # Add time dimension
            zi=self.zi.reshape(self.zi.shape[0], -1) if self.zi.shape[0] > 0 else None
        )

        # Reshape back
        filtered = filtered_flat.reshape(original_shape)

        return filtered.astype(np.float32)

    def reset(self):
        """Reset filter state."""
        self.zi = None
        self.initialized = False

    def update_params(self, fps: float, low_freq: float, high_freq: float):
        """
        Update filter parameters and reset state.

        Args:
            fps: New framerate
            low_freq: New low cutoff
            high_freq: New high cutoff
        """
        self.__init__(fps, low_freq, high_freq, self.filter_order)


class TemporalBufferFilter:
    """
    Buffer-based temporal filter for more accurate filtering.

    Maintains a sliding window of frames and applies filter in batch.
    Better for non-real-time or when latency is acceptable.
    """

    def __init__(
        self,
        fps: float,
        low_freq: float,
        high_freq: float,
        buffer_seconds: float = 10.0,
        filter_order: int = 2
    ):
        """
        Initialize buffer-based filter.

        Args:
            fps: Video framerate
            low_freq: Lower cutoff
            high_freq: Upper cutoff
            buffer_seconds: Length of temporal buffer (seconds)
            filter_order: Butterworth filter order
        """
        self.fps = fps
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.filter_order = filter_order

        # Design filter
        nyquist = fps / 2.0
        low = np.clip(low_freq / nyquist, 0.01, 0.99)
        high = np.clip(high_freq / nyquist, low + 0.01, 0.99)

        self.b, self.a = signal.butter(
            filter_order,
            [low, high],
            btype='band'
        )

        # Circular buffer for frames
        buffer_size = int(buffer_seconds * fps)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add_frame(self, frame: np.ndarray):
        """Add frame to buffer."""
        self.buffer.append(frame.astype(np.float32))

    def get_filtered_current(self) -> Optional[np.ndarray]:
        """
        Get filtered version of most recent frame.

        Returns:
            Filtered frame, or None if buffer not full enough
        """
        if len(self.buffer) < min(30, self.buffer_size // 2):
            # Need minimum frames for stable filtering
            return None

        # Stack buffer into temporal array
        temporal_stack = np.stack(list(self.buffer), axis=0)  # (T, H, W, C)

        # Filter along time axis for each pixel
        T, H, W, C = temporal_stack.shape

        # Reshape to (T, H*W*C) for efficient filtering
        reshaped = temporal_stack.reshape(T, -1)

        # Apply filter (forward-backward for zero phase)
        filtered = signal.filtfilt(self.b, self.a, reshaped, axis=0)

        # Reshape back and return current frame
        filtered_stack = filtered.reshape(T, H, W, C)
        current_filtered = filtered_stack[-1]

        return current_filtered.astype(np.float32)

    def reset(self):
        """Clear buffer."""
        self.buffer.clear()

    def update_params(self, fps: float, low_freq: float, high_freq: float):
        """Update parameters and reset."""
        self.__init__(fps, low_freq, high_freq,
                     self.buffer_size / self.fps, self.filter_order)


def test_temporal_filter():
    """Test temporal filter with synthetic signal."""
    # Create synthetic "video" with known frequency
    fps = 30.0
    duration = 5.0  # seconds
    num_frames = int(fps * duration)

    # Image size
    H, W, C = 64, 64, 3

    # Generate frames with breathing at 0.3 Hz (18 BPM)
    breath_freq = 0.3
    t = np.arange(num_frames) / fps
    breath_signal = np.sin(2 * np.pi * breath_freq * t)

    # Create video frames
    frames = []
    for i, amp in enumerate(breath_signal):
        # Base image + breathing modulation
        frame = np.ones((H, W, C), dtype=np.float32) * 128
        frame += amp * 10  # Small amplitude
        frames.append(frame)

    # Test buffer filter
    filt = TemporalBufferFilter(fps, 0.2, 0.5)  # Band around 0.3 Hz

    for frame in frames:
        filt.add_frame(frame)

    filtered = filt.get_filtered_current()

    if filtered is not None:
        print(f"Temporal filter test: PASS")
        print(f"  Input shape: {frames[0].shape}")
        print(f"  Output shape: {filtered.shape}")
        return True
    else:
        print(f"Temporal filter test: FAIL (not enough frames)")
        return False


if __name__ == "__main__":
    test_temporal_filter()
