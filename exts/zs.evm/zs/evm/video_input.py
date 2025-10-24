"""
Video Input Management

Handles video capture from files, cameras, or image sequences.
Provides unified interface for different video sources.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path
import carb


class VideoSource:
    """Base class for video sources."""

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame. Returns None if no more frames."""
        raise NotImplementedError

    def get_fps(self) -> float:
        """Get video framerate."""
        raise NotImplementedError

    def get_frame_size(self) -> Tuple[int, int]:
        """Get (width, height)."""
        raise NotImplementedError

    def release(self):
        """Release resources."""
        pass

    def reset(self):
        """Reset to beginning."""
        pass


class FileVideoSource(VideoSource):
    """Video from file."""

    def __init__(self, file_path: str):
        """
        Initialize file video source.

        Args:
            file_path: Path to video file
        """
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        carb.log_info(f"[VideoInput] Opened {file_path}: "
                     f"{self.width}x{self.height} @ {self.fps:.1f} fps, "
                     f"{self.frame_count} frames")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def get_fps(self) -> float:
        return self.fps

    def get_frame_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def release(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()

    def reset(self):
        """Reset to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


class CameraVideoSource(VideoSource):
    """Video from camera."""

    def __init__(self, camera_id: int = 0, fps: float = 30.0):
        """
        Initialize camera video source.

        Args:
            camera_id: Camera device ID (0 for default)
            fps: Requested framerate
        """
        self.camera_id = camera_id
        self.requested_fps = fps

        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        # Try to set FPS
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        carb.log_info(f"[VideoInput] Opened camera {camera_id}: "
                     f"{self.width}x{self.height} @ {self.fps:.1f} fps")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from camera."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def get_fps(self) -> float:
        return self.fps

    def get_frame_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def release(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()


class SyntheticVideoSource(VideoSource):
    """
    Synthetic video for testing.

    Generates frames with a breathing pattern.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        duration: float = 30.0,
        breath_freq: float = 0.3
    ):
        """
        Initialize synthetic video source.

        Args:
            width: Frame width
            height: Frame height
            fps: Framerate
            duration: Total duration (seconds)
            breath_freq: Breathing frequency (Hz)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.breath_freq = breath_freq

        self.total_frames = int(fps * duration)
        self.current_frame = 0

        carb.log_info(f"[VideoInput] Created synthetic video: "
                     f"{width}x{height} @ {fps:.1f} fps, "
                     f"{self.total_frames} frames, breath={breath_freq*60:.1f} BPM")

    def read_frame(self) -> Optional[np.ndarray]:
        """Generate next synthetic frame."""
        if self.current_frame >= self.total_frames:
            return None

        # Time
        t = self.current_frame / self.fps

        # Breathing signal
        breath_amp = np.sin(2 * np.pi * self.breath_freq * t)

        # Create base frame
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128

        # Add breathing pattern to center region (torso)
        h_start, h_end = self.height // 3, 2 * self.height // 3
        w_start, w_end = self.width // 3, 2 * self.width // 3

        # Modulate intensity
        modulation = int(breath_amp * 10)
        frame[h_start:h_end, w_start:w_end] = np.clip(
            frame[h_start:h_end, w_start:w_end] + modulation,
            0, 255
        )

        # Add some noise
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self.current_frame += 1

        return frame

    def get_fps(self) -> float:
        return self.fps

    def get_frame_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def reset(self):
        """Reset to beginning."""
        self.current_frame = 0


class VideoInputManager:
    """
    Manages video input and provides frames to the pipeline.
    """

    def __init__(self):
        """Initialize video input manager."""
        self.source: Optional[VideoSource] = None
        self.is_playing = False
        self.current_frame_idx = 0

    def open_file(self, file_path: str):
        """
        Open video file.

        Args:
            file_path: Path to video file
        """
        self.release()
        self.source = FileVideoSource(file_path)
        carb.log_info(f"[VideoInput] Opened file: {file_path}")

    def open_camera(self, camera_id: int = 0, fps: float = 30.0):
        """
        Open camera.

        Args:
            camera_id: Camera device ID
            fps: Requested framerate
        """
        self.release()
        self.source = CameraVideoSource(camera_id, fps)
        carb.log_info(f"[VideoInput] Opened camera: {camera_id}")

    def open_synthetic(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        duration: float = 30.0,
        breath_freq: float = 0.3
    ):
        """
        Create synthetic video for testing.

        Args:
            width: Frame width
            height: Frame height
            fps: Framerate
            duration: Duration (seconds)
            breath_freq: Breathing frequency (Hz)
        """
        self.release()
        self.source = SyntheticVideoSource(width, height, fps, duration, breath_freq)
        carb.log_info("[VideoInput] Created synthetic video")

    def is_opened(self) -> bool:
        """Check if video source is open."""
        return self.source is not None

    def get_fps(self) -> float:
        """Get video framerate."""
        if self.source is None:
            return 30.0
        return self.source.get_fps()

    def get_frame_size(self) -> Tuple[int, int]:
        """Get frame size."""
        if self.source is None:
            return (640, 480)
        return self.source.get_frame_size()

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame from source.

        Returns:
            Frame as numpy array (H, W, 3) RGB, or None if no more frames
        """
        if self.source is None:
            return None

        frame = self.source.read_frame()

        if frame is not None:
            self.current_frame_idx += 1

        return frame

    def play(self):
        """Start playing video."""
        self.is_playing = True
        carb.log_info("[VideoInput] Playing")

    def pause(self):
        """Pause video."""
        self.is_playing = False
        carb.log_info("[VideoInput] Paused")

    def reset(self):
        """Reset video to beginning."""
        if self.source is not None:
            self.source.reset()
            self.current_frame_idx = 0
            carb.log_info("[VideoInput] Reset")

    def release(self):
        """Release current video source."""
        if self.source is not None:
            self.source.release()
            self.source = None
            self.is_playing = False
            self.current_frame_idx = 0
            carb.log_info("[VideoInput] Released")


# Global singleton instance
_video_manager: Optional[VideoInputManager] = None


def get_video_manager() -> VideoInputManager:
    """Get global video input manager instance."""
    global _video_manager
    if _video_manager is None:
        _video_manager = VideoInputManager()
    return _video_manager
