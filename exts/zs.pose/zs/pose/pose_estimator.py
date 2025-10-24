"""
Pose Estimation using MediaPipe

Detects human pose keypoints from video frames for balance and
movement analysis in martial arts training.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict
import carb

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    carb.log_warn("[zs.pose] MediaPipe not available")
    MEDIAPIPE_AVAILABLE = False
    mp = None


class PoseKeypoints:
    """
    Container for pose keypoints with semantic access.

    MediaPipe Pose Landmarks:
    0-10: Face (nose, eyes, ears, mouth)
    11-12: Shoulders
    13-14: Elbows
    15-16: Wrists
    17-22: Hands (fingers)
    23-24: Hips
    25-26: Knees
    27-28: Ankles
    29-32: Feet (heels, toes)
    """

    def __init__(self, landmarks: Optional[List] = None):
        """
        Initialize keypoints.

        Args:
            landmarks: MediaPipe landmarks list
        """
        self.landmarks = landmarks
        self.keypoints_2d: Dict[str, Tuple[float, float]] = {}
        self.keypoints_3d: Dict[str, Tuple[float, float, float]] = {}
        self.visibility: Dict[str, float] = {}

        if landmarks is not None:
            self._extract_keypoints(landmarks)

    def _extract_keypoints(self, landmarks):
        """Extract semantic keypoints from MediaPipe landmarks."""
        landmark_map = {
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot': 31,
            'right_foot': 32
        }

        for name, idx in landmark_map.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                self.keypoints_2d[name] = (lm.x, lm.y)
                self.keypoints_3d[name] = (lm.x, lm.y, lm.z)
                self.visibility[name] = lm.visibility

    def get_2d(self, name: str) -> Optional[Tuple[float, float]]:
        """Get 2D keypoint (normalized coords 0-1)."""
        return self.keypoints_2d.get(name)

    def get_3d(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D keypoint (normalized coords)."""
        return self.keypoints_3d.get(name)

    def get_visibility(self, name: str) -> float:
        """Get keypoint visibility (0-1)."""
        return self.visibility.get(name, 0.0)

    def is_visible(self, name: str, threshold: float = 0.5) -> bool:
        """Check if keypoint is visible above threshold."""
        return self.get_visibility(name) > threshold

    def get_center_of_mass(self) -> Optional[Tuple[float, float]]:
        """
        Estimate center of mass from keypoints.

        Simple approximation: weighted average of hip, shoulder, and head.
        """
        if not self.keypoints_2d:
            return None

        # Get key points for COM
        points = []
        weights = []

        # Hips (heavy)
        if 'left_hip' in self.keypoints_2d and self.is_visible('left_hip'):
            points.append(self.keypoints_2d['left_hip'])
            weights.append(0.25)
        if 'right_hip' in self.keypoints_2d and self.is_visible('right_hip'):
            points.append(self.keypoints_2d['right_hip'])
            weights.append(0.25)

        # Shoulders (medium)
        if 'left_shoulder' in self.keypoints_2d and self.is_visible('left_shoulder'):
            points.append(self.keypoints_2d['left_shoulder'])
            weights.append(0.15)
        if 'right_shoulder' in self.keypoints_2d and self.is_visible('right_shoulder'):
            points.append(self.keypoints_2d['right_shoulder'])
            weights.append(0.15)

        # Head (lighter)
        if 'nose' in self.keypoints_2d and self.is_visible('nose'):
            points.append(self.keypoints_2d['nose'])
            weights.append(0.10)

        if not points:
            return None

        # Weighted average
        points_array = np.array(points)
        weights_array = np.array(weights) / np.sum(weights)
        com = np.average(points_array, axis=0, weights=weights_array)

        return tuple(com)


class PoseEstimator:
    """
    Pose estimator using MediaPipe Pose.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initialize pose estimator.

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: 0 (lite), 1 (full), 2 (heavy)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            static_image_mode=False,
            smooth_landmarks=True
        )

        carb.log_info(f"[zs.pose] Initialized MediaPipe Pose estimator "
                     f"(complexity={model_complexity})")

    def estimate(self, frame: np.ndarray) -> Optional[PoseKeypoints]:
        """
        Estimate pose from frame.

        Args:
            frame: Input frame (H, W, 3) RGB uint8

        Returns:
            PoseKeypoints object, or None if no pose detected
        """
        # MediaPipe expects RGB
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Process frame
        results = self.pose.process(frame)

        if results.pose_landmarks is None:
            return None

        # Extract keypoints
        keypoints = PoseKeypoints(results.pose_landmarks.landmark)

        return keypoints

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: PoseKeypoints,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw keypoints on frame.

        Args:
            frame: Input frame (H, W, 3)
            keypoints: Pose keypoints
            draw_connections: Draw skeleton connections

        Returns:
            Frame with keypoints drawn
        """
        H, W = frame.shape[:2]
        output = frame.copy()

        # Draw keypoints
        for name, (x, y) in keypoints.keypoints_2d.items():
            if keypoints.is_visible(name):
                px = int(x * W)
                py = int(y * H)
                cv2.circle(output, (px, py), 5, (0, 255, 0), -1)

        # Draw connections
        if draw_connections:
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle'),
            ]

            for kp1, kp2 in connections:
                if keypoints.is_visible(kp1) and keypoints.is_visible(kp2):
                    p1 = keypoints.get_2d(kp1)
                    p2 = keypoints.get_2d(kp2)
                    px1, py1 = int(p1[0] * W), int(p1[1] * H)
                    px2, py2 = int(p2[0] * W), int(p2[1] * H)
                    cv2.line(output, (px1, py1), (px2, py2), (255, 0, 0), 2)

        return output

    def release(self):
        """Release resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


def test_pose_estimator():
    """Test pose estimator with synthetic frame."""
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not available, skipping test")
        return False

    # Create test frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Create estimator
    estimator = PoseEstimator()

    # Estimate (will likely return None on blank frame)
    keypoints = estimator.estimate(frame)

    print(f"Pose estimator test:")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Keypoints detected: {keypoints is not None}")

    estimator.release()

    print("  Result: PASS")
    return True


if __name__ == "__main__":
    test_pose_estimator()
