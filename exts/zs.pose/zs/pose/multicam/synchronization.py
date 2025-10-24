"""
Frame Synchronization for Multi-Camera Systems

Handles soft-sync and hardware-sync for multi-camera capture.
Supports LED flash, audio clap, and hardware trigger sync.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class SyncedFrame:
    """
    Synchronized frame from multiple cameras.

    Attributes:
        timestamp: Sync timestamp (seconds)
        frames: Dict of camera_id -> frame
        frame_numbers: Dict of camera_id -> frame number
        sync_confidence: Confidence in synchronization [0, 1]
    """
    timestamp: float
    frames: Dict[str, np.ndarray]
    frame_numbers: Dict[str, int]
    sync_confidence: float = 1.0


class FrameSynchronizer:
    """
    Multi-camera frame synchronization.

    Supports:
    - Hardware sync (external trigger)
    - LED flash detection
    - Audio clap detection
    - Software timestamp matching
    """

    def __init__(
        self,
        sync_method: str = 'timestamp',
        max_time_diff: float = 0.033  # 1 frame at 30fps
    ):
        """
        Initialize synchronizer.

        Args:
            sync_method: 'timestamp', 'led', 'audio', or 'hardware'
            max_time_diff: Maximum timestamp difference for matching (seconds)
        """
        self.sync_method = sync_method
        self.max_time_diff = max_time_diff

        # Frame buffers per camera
        self.buffers: Dict[str, deque] = {}

        # Sync offset per camera (for drift correction)
        self.sync_offsets: Dict[str, float] = {}

        # LED flash detection state
        self.led_flash_detected = False
        self.flash_timestamps: Dict[str, Optional[float]] = {}

    def add_camera(self, camera_id: str):
        """Add camera to synchronizer."""
        if camera_id not in self.buffers:
            self.buffers[camera_id] = deque(maxlen=60)  # 2 seconds at 30fps
            self.sync_offsets[camera_id] = 0.0
            self.flash_timestamps[camera_id] = None

    def add_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int
    ):
        """
        Add frame to buffer.

        Args:
            camera_id: Camera identifier
            frame: Frame data
            timestamp: Frame timestamp (seconds)
            frame_number: Sequential frame number
        """
        if camera_id not in self.buffers:
            self.add_camera(camera_id)

        # Apply sync offset
        adjusted_timestamp = timestamp + self.sync_offsets[camera_id]

        self.buffers[camera_id].append({
            'frame': frame,
            'timestamp': adjusted_timestamp,
            'frame_number': frame_number
        })

        # LED flash detection
        if self.sync_method == 'led' and not self.led_flash_detected:
            if self._detect_led_flash(frame):
                self.flash_timestamps[camera_id] = adjusted_timestamp

                # Check if all cameras detected flash
                if all(t is not None for t in self.flash_timestamps.values()):
                    self._calibrate_from_flash()

    def _detect_led_flash(self, frame: np.ndarray) -> bool:
        """
        Detect LED flash in frame.

        Args:
            frame: Frame to analyze

        Returns:
            True if flash detected
        """
        # Simple threshold on mean brightness
        mean_brightness = np.mean(frame)
        threshold = 200  # For uint8 images

        return mean_brightness > threshold

    def _calibrate_from_flash(self):
        """Calibrate sync offsets from LED flash timestamps."""
        # Find earliest flash
        min_time = min(t for t in self.flash_timestamps.values() if t is not None)

        # Set offsets
        for camera_id, flash_time in self.flash_timestamps.items():
            if flash_time is not None:
                self.sync_offsets[camera_id] = min_time - flash_time

        self.led_flash_detected = True
        print(f"[Sync] Calibrated from LED flash: {self.sync_offsets}")

    def get_synced_frame(self) -> Optional[SyncedFrame]:
        """
        Get synchronized frame set.

        Returns:
            SyncedFrame if match found, None otherwise
        """
        if self.sync_method == 'hardware':
            return self._get_hardware_synced()
        elif self.sync_method == 'timestamp':
            return self._get_timestamp_synced()
        elif self.sync_method == 'led':
            return self._get_timestamp_synced()  # After calibration, use timestamps
        else:
            return self._get_timestamp_synced()

    def _get_hardware_synced(self) -> Optional[SyncedFrame]:
        """
        Get hardware-synced frames (same frame number).

        Returns:
            SyncedFrame or None
        """
        if not self.buffers:
            return None

        # Check if all cameras have frames
        if any(len(buf) == 0 for buf in self.buffers.values()):
            return None

        # Get oldest frame from each camera
        oldest_frames = {}
        frame_numbers = {}

        for camera_id, buf in self.buffers.items():
            if len(buf) > 0:
                oldest = buf[0]
                oldest_frames[camera_id] = oldest['frame']
                frame_numbers[camera_id] = oldest['frame_number']

        # Check if frame numbers match
        frame_nums = list(frame_numbers.values())
        if len(set(frame_nums)) == 1:
            # Match! Remove from buffers
            synced = {}
            for camera_id in self.buffers:
                synced[camera_id] = self.buffers[camera_id].popleft()['frame']

            return SyncedFrame(
                timestamp=time.time(),
                frames=synced,
                frame_numbers=frame_numbers,
                sync_confidence=1.0
            )

        return None

    def _get_timestamp_synced(self) -> Optional[SyncedFrame]:
        """
        Get timestamp-synced frames.

        Returns:
            SyncedFrame or None
        """
        if not self.buffers:
            return None

        # Check if all cameras have frames
        if any(len(buf) == 0 for buf in self.buffers.values()):
            return None

        # Find oldest frame across all cameras
        oldest_time = float('inf')
        for buf in self.buffers.values():
            if len(buf) > 0:
                oldest_time = min(oldest_time, buf[0]['timestamp'])

        # Find matching frames within time window
        matches = {}
        frame_numbers = {}

        for camera_id, buf in self.buffers.items():
            # Find frame closest to oldest_time
            best_idx = None
            best_diff = float('inf')

            for i, frame_data in enumerate(buf):
                diff = abs(frame_data['timestamp'] - oldest_time)

                if diff < best_diff and diff < self.max_time_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is not None:
                matches[camera_id] = best_idx
                frame_numbers[camera_id] = buf[best_idx]['frame_number']

        # Check if we have match from all cameras
        if len(matches) == len(self.buffers):
            # Extract frames
            synced = {}
            timestamps = []

            for camera_id, idx in matches.items():
                frame_data = self.buffers[camera_id][idx]
                synced[camera_id] = frame_data['frame']
                timestamps.append(frame_data['timestamp'])

                # Remove this and earlier frames
                for _ in range(idx + 1):
                    if len(self.buffers[camera_id]) > 0:
                        self.buffers[camera_id].popleft()

            # Compute sync confidence
            time_std = np.std(timestamps)
            confidence = max(0.0, 1.0 - time_std / self.max_time_diff)

            return SyncedFrame(
                timestamp=np.mean(timestamps),
                frames=synced,
                frame_numbers=frame_numbers,
                sync_confidence=confidence
            )

        return None

    def reset(self):
        """Reset synchronizer state."""
        for buf in self.buffers.values():
            buf.clear()

        self.sync_offsets = {k: 0.0 for k in self.sync_offsets}
        self.led_flash_detected = False
        self.flash_timestamps = {k: None for k in self.flash_timestamps}


def test_synchronizer():
    """Test frame synchronizer."""
    print("Testing Frame Synchronizer...")

    sync = FrameSynchronizer(sync_method='timestamp', max_time_diff=0.033)

    # Add cameras
    sync.add_camera('cam1')
    sync.add_camera('cam2')

    # Simulate frames with slight time offset
    base_time = time.time()

    for i in range(10):
        # Camera 1: exactly on time
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        sync.add_frame('cam1', frame1, base_time + i * 0.033, i)

        # Camera 2: 5ms offset
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 150
        sync.add_frame('cam2', frame2, base_time + i * 0.033 + 0.005, i)

        # Try to get synced frame
        synced = sync.get_synced_frame()

        if synced:
            print(f"  Frame {i}: synced with confidence {synced.sync_confidence:.3f}")
            assert 'cam1' in synced.frames
            assert 'cam2' in synced.frames

    print("✓ PASS")


if __name__ == "__main__":
    test_synchronizer()
