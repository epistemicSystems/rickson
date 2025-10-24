"""
Rickson Pose Estimation Extension

Provides pose estimation, keypoint tracking, and balance analysis.
"""

import omni.ext
import carb


class RicksonPoseExtension(omni.ext.IExt):
    """Pose estimation extension for Rickson training assistant."""

    def on_startup(self, ext_id):
        """Called when the extension starts up."""
        carb.log_info("[zs.pose] Rickson Pose Extension starting...")
        carb.log_info("[zs.pose] Rickson Pose Extension started successfully")

    def on_shutdown(self):
        """Called when the extension is shutting down."""
        carb.log_info("[zs.pose] Rickson Pose Extension shutting down...")
        carb.log_info("[zs.pose] Rickson Pose Extension shut down")
