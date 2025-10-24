"""
Rickson Core Extension

Provides event logging, state management, and insights engine.
"""

import omni.ext
import carb


class RicksonCoreExtension(omni.ext.IExt):
    """Core extension for Rickson training assistant."""

    def on_startup(self, ext_id):
        """Called when the extension starts up."""
        carb.log_info("[zs.core] Rickson Core Extension starting...")
        carb.log_info("[zs.core] Rickson Core Extension started successfully")

    def on_shutdown(self):
        """Called when the extension is shutting down."""
        carb.log_info("[zs.core] Rickson Core Extension shutting down...")
        carb.log_info("[zs.core] Rickson Core Extension shut down")
