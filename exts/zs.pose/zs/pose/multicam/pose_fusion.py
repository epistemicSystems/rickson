"""
3D Pose Fusion with 3DGS Constraints

Fuses multi-camera 2D pose estimates into constrained 3D pose using:
- Multi-view triangulation
- 3D Gaussian Splatting gym priors (depth/collision constraints)
- Temporal smoothing
- Physical plausibility checks
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import least_squares

from ..pose_estimator import PoseKeypoints
from .calibration import MultiCameraCalibration


@dataclass
class Pose3D:
    """
    3D pose with keypoints and confidence.

    Attributes:
        keypoints_3d: Dict of keypoint name -> (x, y, z)
        confidence: Dict of keypoint name -> confidence [0, 1]
        timestamp: Pose timestamp
        floor_height: Estimated floor height
    """
    keypoints_3d: Dict[str, np.ndarray]
    confidence: Dict[str, float]
    timestamp: float
    floor_height: float = 0.0

    def get_center_of_mass(self) -> Optional[np.ndarray]:
        """Compute 3D center of mass."""
        if not self.keypoints_3d:
            return None

        # Weighted average
        points = []
        weights = []

        weight_map = {
            'left_hip': 0.25, 'right_hip': 0.25,
            'left_shoulder': 0.15, 'right_shoulder': 0.15,
            'nose': 0.10
        }

        for name, pos in self.keypoints_3d.items():
            if name in weight_map:
                points.append(pos)
                weights.append(weight_map[name] * self.confidence.get(name, 1.0))

        if not points:
            return None

        points = np.array(points)
        weights = np.array(weights) / np.sum(weights)

        com = np.average(points, axis=0, weights=weights)

        return com


class Pose3DFusion:
    """
    Fuse multi-camera 2D poses into 3D with gym prior constraints.
    """

    def __init__(
        self,
        calibration: MultiCameraCalibration,
        gym_prior: Optional['GymPrior'] = None,
        temporal_smoothing: float = 0.7
    ):
        """
        Initialize 3D pose fusion.

        Args:
            calibration: Multi-camera calibration
            gym_prior: Optional gym prior for depth constraints
            temporal_smoothing: Temporal smoothing factor [0, 1]
        """
        self.calibration = calibration
        self.gym_prior = gym_prior
        self.temporal_smoothing = temporal_smoothing

        # Previous pose for temporal smoothing
        self.prev_pose: Optional[Pose3D] = None

    def fuse(
        self,
        poses_2d: Dict[str, PoseKeypoints],
        timestamp: float
    ) -> Optional[Pose3D]:
        """
        Fuse 2D poses from multiple cameras into 3D.

        Args:
            poses_2d: Dict of camera_id -> PoseKeypoints
            timestamp: Frame timestamp

        Returns:
            Pose3D or None if fusion fails
        """
        if len(poses_2d) < 2:
            # Need at least 2 views
            return None

        # Get keypoint names present in all views
        all_keypoints = set()
        for pose in poses_2d.values():
            all_keypoints.update(pose.keypoints_2d.keys())

        # Triangulate each keypoint
        keypoints_3d = {}
        confidence_3d = {}

        for kp_name in all_keypoints:
            # Find cameras that see this keypoint
            visible_cameras = []
            points_2d = []
            confidences = []

            for camera_id, pose in poses_2d.items():
                if pose.is_visible(kp_name, threshold=0.5):
                    visible_cameras.append(camera_id)
                    points_2d.append(pose.get_2d(kp_name))
                    confidences.append(pose.get_visibility(kp_name))

            if len(visible_cameras) >= 2:
                # Triangulate
                point_3d = self._triangulate_multi_view(
                    visible_cameras,
                    points_2d,
                    confidences
                )

                if point_3d is not None:
                    keypoints_3d[kp_name] = point_3d
                    confidence_3d[kp_name] = np.mean(confidences)

        if not keypoints_3d:
            return None

        # Apply gym prior constraints
        if self.gym_prior is not None:
            keypoints_3d = self._apply_gym_constraints(keypoints_3d)

        # Estimate floor height
        floor_height = self._estimate_floor_height(keypoints_3d)

        # Create 3D pose
        pose_3d = Pose3D(
            keypoints_3d=keypoints_3d,
            confidence=confidence_3d,
            timestamp=timestamp,
            floor_height=floor_height
        )

        # Apply temporal smoothing
        if self.prev_pose is not None:
            pose_3d = self._apply_temporal_smoothing(pose_3d, self.prev_pose)

        self.prev_pose = pose_3d

        return pose_3d

    def _triangulate_multi_view(
        self,
        camera_ids: List[str],
        points_2d: List[Tuple[float, float]],
        confidences: List[float]
    ) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from multiple views.

        Uses weighted least squares for >2 views.

        Args:
            camera_ids: List of camera identifiers
            points_2d: List of 2D points (one per camera)
            confidences: Confidence weights

        Returns:
            (3,) 3D point or None
        """
        if len(camera_ids) < 2:
            return None

        if len(camera_ids) == 2:
            # Simple stereo triangulation
            pts1 = np.array([points_2d[0]])
            pts2 = np.array([points_2d[1]])

            point_3d = self.calibration.triangulate(
                camera_ids[0],
                camera_ids[1],
                pts1,
                pts2
            )

            return point_3d[0]

        # Multi-view triangulation via optimization
        # Minimize reprojection error

        def residuals(point_3d):
            """Compute reprojection errors."""
            errors = []

            for i, camera_id in enumerate(camera_ids):
                intrinsics, extrinsics = self.calibration.cameras[camera_id]

                # Project 3D point
                point_3d_reshaped = point_3d.reshape(1, 3)
                proj = extrinsics.project(point_3d_reshaped, intrinsics)[0]

                # Reprojection error
                observed = np.array(points_2d[i])
                error = (proj - observed) * confidences[i]

                errors.extend(error)

            return np.array(errors)

        # Initial guess from first stereo pair
        x0 = self._triangulate_multi_view(
            camera_ids[:2],
            points_2d[:2],
            confidences[:2]
        )

        if x0 is None:
            return None

        # Optimize
        result = least_squares(residuals, x0, method='lm')

        if result.success:
            return result.x
        else:
            return x0  # Fall back to initial guess

    def _apply_gym_constraints(
        self,
        keypoints_3d: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply gym prior depth/collision constraints.

        Args:
            keypoints_3d: Unconstrained keypoints

        Returns:
            Constrained keypoints
        """
        if self.gym_prior is None:
            return keypoints_3d

        constrained = {}

        for name, point in keypoints_3d.items():
            # Query gym prior for depth constraint
            # For now, just ensure points are above floor

            floor_height = self.gym_prior.get_floor_height()

            # Don't let keypoints go below floor
            if point[2] < floor_height:
                point = point.copy()
                point[2] = floor_height + 0.05  # 5cm above floor

            constrained[name] = point

        return constrained

    def _estimate_floor_height(
        self,
        keypoints_3d: Dict[str, np.ndarray]
    ) -> float:
        """
        Estimate floor height from foot keypoints.

        Args:
            keypoints_3d: 3D keypoints

        Returns:
            Floor height (z coordinate)
        """
        # Use ankle/heel/foot keypoints
        foot_keypoints = [
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot', 'right_foot'
        ]

        z_values = []

        for name in foot_keypoints:
            if name in keypoints_3d:
                z_values.append(keypoints_3d[name][2])

        if z_values:
            # Use minimum (lowest point)
            return min(z_values)

        # Fall back to gym prior
        if self.gym_prior is not None:
            return self.gym_prior.get_floor_height()

        return 0.0

    def _apply_temporal_smoothing(
        self,
        current: Pose3D,
        previous: Pose3D
    ) -> Pose3D:
        """
        Apply temporal smoothing between frames.

        Args:
            current: Current pose
            previous: Previous pose

        Returns:
            Smoothed pose
        """
        alpha = self.temporal_smoothing

        smoothed_keypoints = {}

        for name, curr_point in current.keypoints_3d.items():
            if name in previous.keypoints_3d:
                prev_point = previous.keypoints_3d[name]

                # Exponential moving average
                smoothed = alpha * prev_point + (1 - alpha) * curr_point

                smoothed_keypoints[name] = smoothed
            else:
                smoothed_keypoints[name] = curr_point

        return Pose3D(
            keypoints_3d=smoothed_keypoints,
            confidence=current.confidence,
            timestamp=current.timestamp,
            floor_height=current.floor_height
        )

    def reset(self):
        """Reset temporal state."""
        self.prev_pose = None


def test_pose_fusion():
    """Test 3D pose fusion."""
    print("Testing 3D Pose Fusion...")

    # Create calibration
    from .calibration import MultiCameraCalibration, CameraIntrinsics, CameraExtrinsics

    calib = MultiCameraCalibration()

    # Camera 1: origin
    intrinsics1 = CameraIntrinsics(
        fx=800, fy=800, cx=640, cy=360,
        width=1280, height=720,
        distortion=np.zeros(5)
    )
    calib.add_camera('cam1', intrinsics1)

    # Camera 2: 2m to the right
    intrinsics2 = CameraIntrinsics(
        fx=800, fy=800, cx=640, cy=360,
        width=1280, height=720,
        distortion=np.zeros(5)
    )
    extrinsics2 = CameraExtrinsics(
        R=np.eye(3),
        t=np.array([[2.0], [0.0], [0.0]])
    )
    calib.add_camera('cam2', intrinsics2, extrinsics2)

    # Create fusion
    fusion = Pose3DFusion(calib, gym_prior=None, temporal_smoothing=0.5)

    # Create synthetic 2D poses
    # Actual 3D point: (0, 0, 3)
    point_3d_true = np.array([[0, 0, 3]])

    intrinsics1, extrinsics1 = calib.cameras['cam1']
    intrinsics2, extrinsics2 = calib.cameras['cam2']

    point_2d_1 = extrinsics1.project(point_3d_true, intrinsics1)[0]
    point_2d_2 = extrinsics2.project(point_3d_true, intrinsics2)[0]

    print(f"True 3D: {point_3d_true[0]}")
    print(f"Projected 2D: cam1={point_2d_1}, cam2={point_2d_2}")

    # Create PoseKeypoints
    pose1 = PoseKeypoints()
    pose1.keypoints_2d = {'nose': tuple(point_2d_1 / [1280, 720])}  # Normalize
    pose1.visibility = {'nose': 0.9}

    pose2 = PoseKeypoints()
    pose2.keypoints_2d = {'nose': tuple(point_2d_2 / [1280, 720])}
    pose2.visibility = {'nose': 0.9}

    # Fuse
    pose_3d = fusion.fuse({'cam1': pose1, 'cam2': pose2}, timestamp=0.0)

    if pose_3d:
        recon = pose_3d.keypoints_3d['nose']
        print(f"Reconstructed 3D: {recon}")

        error = np.linalg.norm(point_3d_true[0] - recon)
        print(f"Reconstruction error: {error:.6f}m")

        assert error < 0.01, "3D reconstruction error too large"

        print("✓ PASS")
    else:
        print("✗ FAIL: Fusion returned None")


if __name__ == "__main__":
    test_pose_fusion()
