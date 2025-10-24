"""
Multi-Camera 3D Pose Estimation

Calibration, synchronization, and 3D pose fusion from multiple cameras.
"""

from .calibration import CameraIntrinsics, CameraExtrinsics, MultiCameraCalibration
from .synchronization import FrameSynchronizer
from .pose_fusion import Pose3DFusion

__all__ = [
    'CameraIntrinsics',
    'CameraExtrinsics',
    'MultiCameraCalibration',
    'FrameSynchronizer',
    'Pose3DFusion'
]
