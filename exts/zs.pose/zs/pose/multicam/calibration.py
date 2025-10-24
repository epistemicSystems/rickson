"""
Multi-Camera Calibration

Handles intrinsic/extrinsic calibration for multi-camera setups.
Supports checkerboard, ChArUco, and LED-sync calibration methods.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    Attributes:
        fx, fy: Focal lengths (pixels)
        cx, cy: Principal point (pixels)
        width, height: Image dimensions
        distortion: Distortion coefficients [k1, k2, p1, p2, k3]
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray

    def matrix(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['distortion'] = self.distortion.tolist()
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'CameraIntrinsics':
        """Load from dictionary."""
        d['distortion'] = np.array(d['distortion'])
        return CameraIntrinsics(**d)


@dataclass
class CameraExtrinsics:
    """
    Camera extrinsic parameters (pose in world frame).

    Attributes:
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
    """
    R: np.ndarray
    t: np.ndarray

    def matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.flatten()
        return T

    def inverse(self) -> 'CameraExtrinsics':
        """Get inverse (camera to world)."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return CameraExtrinsics(R_inv, t_inv)

    def project(self, points_3d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        """
        Project 3D points to 2D.

        Args:
            points_3d: (N, 3) 3D points in world frame
            intrinsics: Camera intrinsics

        Returns:
            (N, 2) 2D image points
        """
        # Transform to camera frame
        points_cam = (self.R @ points_3d.T).T + self.t.T

        # Project
        K = intrinsics.matrix()
        points_2d_h = (K @ points_cam.T).T
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:]

        return points_2d

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'R': self.R.tolist(),
            't': self.t.tolist()
        }

    @staticmethod
    def from_dict(d: Dict) -> 'CameraExtrinsics':
        """Load from dictionary."""
        return CameraExtrinsics(
            R=np.array(d['R']),
            t=np.array(d['t'])
        )


class MultiCameraCalibration:
    """
    Multi-camera calibration system.

    Supports:
    - Intrinsic calibration (checkerboard/ChArUco)
    - Extrinsic calibration (relative poses)
    - Bundle adjustment
    """

    def __init__(self):
        """Initialize calibration system."""
        self.cameras: Dict[str, Tuple[CameraIntrinsics, CameraExtrinsics]] = {}

    def add_camera(
        self,
        camera_id: str,
        intrinsics: CameraIntrinsics,
        extrinsics: Optional[CameraExtrinsics] = None
    ):
        """
        Add camera to system.

        Args:
            camera_id: Unique camera identifier
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics (optional, can calibrate later)
        """
        if extrinsics is None:
            # Default to identity (world frame)
            extrinsics = CameraExtrinsics(
                R=np.eye(3),
                t=np.zeros((3, 1))
            )

        self.cameras[camera_id] = (intrinsics, extrinsics)

    def calibrate_intrinsics(
        self,
        camera_id: str,
        images: List[np.ndarray],
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025  # meters
    ) -> CameraIntrinsics:
        """
        Calibrate camera intrinsics using checkerboard.

        Args:
            camera_id: Camera identifier
            images: List of calibration images
            checkerboard_size: (cols, rows) of inner corners
            square_size: Physical size of squares (meters)

        Returns:
            Calibrated intrinsics
        """
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # Find corners in images
        obj_points = []  # 3D points
        img_points = []  # 2D points

        h, w = images[0].shape[:2]

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                obj_points.append(objp)
                img_points.append(corners)

        if len(obj_points) < 3:
            raise ValueError(f"Not enough valid images for calibration (found {len(obj_points)})")

        # Calibrate
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )

        intrinsics = CameraIntrinsics(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=w,
            height=h,
            distortion=dist.flatten()
        )

        # Update in system
        if camera_id in self.cameras:
            _, extrinsics = self.cameras[camera_id]
            self.cameras[camera_id] = (intrinsics, extrinsics)
        else:
            self.add_camera(camera_id, intrinsics)

        return intrinsics

    def calibrate_extrinsics_stereo(
        self,
        camera_id_1: str,
        camera_id_2: str,
        images_1: List[np.ndarray],
        images_2: List[np.ndarray],
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025
    ) -> Tuple[CameraExtrinsics, CameraExtrinsics]:
        """
        Calibrate stereo extrinsics between two cameras.

        Args:
            camera_id_1, camera_id_2: Camera identifiers
            images_1, images_2: Synchronized calibration images
            checkerboard_size: Checkerboard dimensions
            square_size: Square size in meters

        Returns:
            (extrinsics_1, extrinsics_2) relative to common frame
        """
        if camera_id_1 not in self.cameras or camera_id_2 not in self.cameras:
            raise ValueError("Cameras must be added with intrinsics first")

        intrinsics_1, _ = self.cameras[camera_id_1]
        intrinsics_2, _ = self.cameras[camera_id_2]

        # Find matching corners
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        obj_points = []
        img_points_1 = []
        img_points_2 = []

        for img1, img2 in zip(images_1, images_2):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

            ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size)
            ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size)

            if ret1 and ret2:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                obj_points.append(objp)
                img_points_1.append(corners1)
                img_points_2.append(corners2)

        if len(obj_points) < 3:
            raise ValueError("Not enough matching images for stereo calibration")

        # Stereo calibration
        K1 = intrinsics_1.matrix()
        K2 = intrinsics_2.matrix()
        D1 = intrinsics_1.distortion
        D2 = intrinsics_2.distortion

        flags = cv2.CALIB_FIX_INTRINSIC

        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_1,
            img_points_2,
            K1, D1, K2, D2,
            (intrinsics_1.width, intrinsics_1.height),
            flags=flags
        )

        # Camera 1 at origin
        extrinsics_1 = CameraExtrinsics(
            R=np.eye(3),
            t=np.zeros((3, 1))
        )

        # Camera 2 relative to camera 1
        extrinsics_2 = CameraExtrinsics(
            R=R,
            t=T
        )

        # Update system
        self.cameras[camera_id_1] = (intrinsics_1, extrinsics_1)
        self.cameras[camera_id_2] = (intrinsics_2, extrinsics_2)

        return extrinsics_1, extrinsics_2

    def triangulate(
        self,
        camera_id_1: str,
        camera_id_2: str,
        points_2d_1: np.ndarray,
        points_2d_2: np.ndarray
    ) -> np.ndarray:
        """
        Triangulate 3D points from 2D correspondences.

        Args:
            camera_id_1, camera_id_2: Camera identifiers
            points_2d_1, points_2d_2: (N, 2) 2D points in each camera

        Returns:
            (N, 3) 3D points in world frame
        """
        intrinsics_1, extrinsics_1 = self.cameras[camera_id_1]
        intrinsics_2, extrinsics_2 = self.cameras[camera_id_2]

        # Projection matrices
        K1 = intrinsics_1.matrix()
        K2 = intrinsics_2.matrix()

        T1 = extrinsics_1.matrix()
        T2 = extrinsics_2.matrix()

        P1 = K1 @ T1[:3, :]
        P2 = K2 @ T2[:3, :]

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, points_2d_1.T, points_2d_2.T)

        # Convert to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]

        return points_3d.T

    def save(self, path: str):
        """Save calibration to JSON file."""
        data = {}

        for camera_id, (intrinsics, extrinsics) in self.cameras.items():
            data[camera_id] = {
                'intrinsics': intrinsics.to_dict(),
                'extrinsics': extrinsics.to_dict()
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> 'MultiCameraCalibration':
        """Load calibration from JSON file."""
        calib = MultiCameraCalibration()

        with open(path) as f:
            data = json.load(f)

        for camera_id, cam_data in data.items():
            intrinsics = CameraIntrinsics.from_dict(cam_data['intrinsics'])
            extrinsics = CameraExtrinsics.from_dict(cam_data['extrinsics'])
            calib.add_camera(camera_id, intrinsics, extrinsics)

        return calib


def test_calibration():
    """Test calibration system."""
    print("Testing Multi-Camera Calibration...")

    # Create synthetic calibration
    calib = MultiCameraCalibration()

    # Camera 1: identity
    intrinsics_1 = CameraIntrinsics(
        fx=800, fy=800, cx=640, cy=360,
        width=1280, height=720,
        distortion=np.zeros(5)
    )

    # Camera 2: translated 1m to the right
    intrinsics_2 = CameraIntrinsics(
        fx=800, fy=800, cx=640, cy=360,
        width=1280, height=720,
        distortion=np.zeros(5)
    )

    extrinsics_2 = CameraExtrinsics(
        R=np.eye(3),
        t=np.array([[1.0], [0.0], [0.0]])
    )

    calib.add_camera('cam1', intrinsics_1)
    calib.add_camera('cam2', intrinsics_2, extrinsics_2)

    # Test triangulation
    # 3D point at (0, 0, 5)
    point_3d = np.array([[0, 0, 5]])

    # Project to each camera
    intrinsics_1, extrinsics_1 = calib.cameras['cam1']
    intrinsics_2, extrinsics_2 = calib.cameras['cam2']

    point_2d_1 = extrinsics_1.project(point_3d, intrinsics_1)
    point_2d_2 = extrinsics_2.project(point_3d, intrinsics_2)

    print(f"2D projections: {point_2d_1}, {point_2d_2}")

    # Triangulate back
    point_3d_recon = calib.triangulate('cam1', 'cam2', point_2d_1, point_2d_2)

    print(f"Original 3D: {point_3d[0]}")
    print(f"Reconstructed: {point_3d_recon[0]}")

    error = np.linalg.norm(point_3d - point_3d_recon)
    print(f"Reconstruction error: {error:.6f}")

    assert error < 0.001, "Triangulation error too large"

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        temp_path = f.name

    calib.save(temp_path)
    loaded_calib = MultiCameraCalibration.load(temp_path)

    assert len(loaded_calib.cameras) == 2, "Wrong number of cameras loaded"

    import os
    os.unlink(temp_path)

    print("✓ PASS")


if __name__ == "__main__":
    test_calibration()
