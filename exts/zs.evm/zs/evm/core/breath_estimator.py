"""
Breath Rate Estimation

Extract breathing rate from temporally filtered video via peak detection
and frequency analysis.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, List


def extract_roi_signal(
    frame: np.ndarray,
    roi_mask: Optional[np.ndarray] = None
) -> float:
    """
    Extract representative signal from frame (e.g., torso region average).

    Args:
        frame: Input frame (H, W, C)
        roi_mask: Optional binary mask for region of interest

    Returns:
        Scalar signal value (mean intensity in ROI)
    """
    if roi_mask is not None:
        # Apply mask
        masked = frame[roi_mask > 0]
        if len(masked) == 0:
            return 0.0
        return float(np.mean(masked))
    else:
        # Use center region as default ROI
        H, W = frame.shape[:2]
        h_start, h_end = H // 3, 2 * H // 3
        w_start, w_end = W // 3, 2 * W // 3
        roi = frame[h_start:h_end, w_start:w_end]
        return float(np.mean(roi))


def estimate_breath_rate_fft(
    signal_buffer: np.ndarray,
    fps: float,
    freq_range: Tuple[float, float] = (0.1, 0.7)
) -> Tuple[float, float]:
    """
    Estimate breath rate using FFT (frequency domain).

    Args:
        signal_buffer: Temporal signal buffer (T,)
        fps: Sampling rate (frames per second)
        freq_range: Expected breathing frequency range (Hz)

    Returns:
        (breath_rate_bpm, confidence)
    """
    if len(signal_buffer) < 30:
        # Not enough data
        return 0.0, 0.0

    # Detrend signal
    signal_detrended = signal.detrend(signal_buffer)

    # Apply window to reduce spectral leakage
    window = signal.windows.hann(len(signal_detrended))
    signal_windowed = signal_detrended * window

    # Compute FFT
    fft_vals = np.abs(fft(signal_windowed))
    freqs = fftfreq(len(signal_windowed), 1.0 / fps)

    # Only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]

    # Limit to breathing frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(freq_mask):
        return 0.0, 0.0

    freqs_band = freqs[freq_mask]
    fft_band = fft_vals[freq_mask]

    # Find peak
    peak_idx = np.argmax(fft_band)
    peak_freq = freqs_band[peak_idx]
    peak_magnitude = fft_band[peak_idx]

    # Confidence: ratio of peak to mean
    mean_magnitude = np.mean(fft_band)
    confidence = min(1.0, peak_magnitude / (mean_magnitude + 1e-6))

    # Convert to BPM
    breath_rate_bpm = peak_freq * 60.0

    return breath_rate_bpm, confidence


def estimate_breath_rate_peaks(
    signal_buffer: np.ndarray,
    fps: float,
    freq_range: Tuple[float, float] = (0.1, 0.7)
) -> Tuple[float, float]:
    """
    Estimate breath rate using peak detection (time domain).

    Args:
        signal_buffer: Temporal signal buffer (T,)
        fps: Sampling rate
        freq_range: Expected breathing frequency range (Hz)

    Returns:
        (breath_rate_bpm, confidence)
    """
    if len(signal_buffer) < 30:
        return 0.0, 0.0

    # Detrend
    signal_detrended = signal.detrend(signal_buffer)

    # Find peaks
    # Minimum distance between peaks based on max frequency
    min_distance = int(fps / freq_range[1])  # frames

    peaks, properties = signal.find_peaks(
        signal_detrended,
        distance=min_distance,
        prominence=np.std(signal_detrended) * 0.3  # Require significant peaks
    )

    if len(peaks) < 2:
        # Not enough peaks
        return 0.0, 0.0

    # Compute inter-peak intervals
    intervals = np.diff(peaks) / fps  # seconds
    mean_interval = np.mean(intervals)

    # Convert to BPM
    breath_rate_bpm = 60.0 / mean_interval

    # Check if in valid range
    if not (freq_range[0] * 60 <= breath_rate_bpm <= freq_range[1] * 60):
        return 0.0, 0.0

    # Confidence based on consistency of intervals
    interval_std = np.std(intervals)
    confidence = np.exp(-interval_std / mean_interval)  # High when consistent

    return breath_rate_bpm, confidence


class BreathEstimator:
    """
    Stateful breath rate estimator.

    Maintains temporal buffer and provides continuous estimates.
    """

    def __init__(
        self,
        fps: float,
        buffer_seconds: float = 10.0,
        freq_range: Tuple[float, float] = (0.1, 0.7),
        method: str = 'fft'
    ):
        """
        Initialize estimator.

        Args:
            fps: Video framerate
            buffer_seconds: Length of signal buffer
            freq_range: Expected breathing frequency range (Hz)
            method: 'fft' or 'peaks'
        """
        self.fps = fps
        self.freq_range = freq_range
        self.method = method

        self.buffer_size = int(buffer_seconds * fps)
        self.signal_buffer: List[float] = []

    def add_frame_signal(self, signal_value: float):
        """Add a scalar signal value from current frame."""
        self.signal_buffer.append(signal_value)

        # Keep buffer at fixed size
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

    def estimate(self) -> Tuple[float, float]:
        """
        Estimate current breath rate.

        Returns:
            (breath_rate_bpm, confidence)
        """
        if len(self.signal_buffer) < 30:
            return 0.0, 0.0

        signal_array = np.array(self.signal_buffer, dtype=np.float32)

        if self.method == 'fft':
            return estimate_breath_rate_fft(signal_array, self.fps, self.freq_range)
        elif self.method == 'peaks':
            return estimate_breath_rate_peaks(signal_array, self.fps, self.freq_range)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_signal_history(self) -> np.ndarray:
        """Get the current signal buffer for visualization."""
        return np.array(self.signal_buffer, dtype=np.float32)

    def reset(self):
        """Clear buffer."""
        self.signal_buffer.clear()


def detect_breath_hold(
    breath_rate_history: List[float],
    threshold_bpm: float = 6.0,
    duration_seconds: float = 3.0,
    fps: float = 30.0
) -> bool:
    """
    Detect if athlete is holding breath.

    Args:
        breath_rate_history: Recent breath rate estimates (BPM)
        threshold_bpm: Below this rate is considered breath hold
        duration_seconds: How long below threshold to confirm
        fps: Framerate

    Returns:
        True if breath hold detected
    """
    required_frames = int(duration_seconds * fps)

    if len(breath_rate_history) < required_frames:
        return False

    recent = breath_rate_history[-required_frames:]
    below_threshold = [r < threshold_bpm for r in recent]

    # If most recent frames are below threshold
    if sum(below_threshold) >= int(required_frames * 0.8):
        return True

    return False


def test_breath_estimator():
    """Test breath estimator with synthetic signal."""
    fps = 30.0
    duration = 10.0
    num_frames = int(fps * duration)

    # Synthetic breathing at 18 BPM (0.3 Hz)
    true_breath_rate = 18.0
    true_freq = true_breath_rate / 60.0

    t = np.arange(num_frames) / fps
    signal_vals = np.sin(2 * np.pi * true_freq * t) + \
                  np.random.randn(num_frames) * 0.1  # Add noise

    # Test estimator
    estimator = BreathEstimator(fps, buffer_seconds=10.0, method='fft')

    for val in signal_vals:
        estimator.add_frame_signal(val)

    # Estimate after full buffer
    estimated_bpm, confidence = estimator.estimate()

    print(f"Breath estimator test:")
    print(f"  True BPM: {true_breath_rate:.1f}")
    print(f"  Estimated BPM: {estimated_bpm:.1f}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Error: {abs(estimated_bpm - true_breath_rate):.1f} BPM")

    # Success if within 2 BPM
    success = abs(estimated_bpm - true_breath_rate) < 2.0
    print(f"  Result: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    test_breath_estimator()
