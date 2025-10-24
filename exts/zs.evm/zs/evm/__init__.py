"""
Rickson EVM (Eulerian Video Magnification) Extension

Provides OmniGraph compute nodes for breath analysis via
temporal band-pass filtering over spatial pyramids.
"""

import omni.ext
import carb


class RicksonEVMExtension(omni.ext.IExt):
    """EVM compute extension for Rickson training assistant."""

    def on_startup(self, ext_id):
        """Called when the extension starts up."""
        carb.log_info("[zs.evm] Rickson EVM Extension starting...")

        # Register OmniGraph nodes
        # Note: OmniGraph nodes are auto-discovered from the nodes/ directory
        # when following the proper .ogn structure

        carb.log_info("[zs.evm] Rickson EVM Extension started successfully")

    def on_shutdown(self):
        """Called when the extension is shutting down."""
        carb.log_info("[zs.evm] Rickson EVM Extension shutting down...")
        carb.log_info("[zs.evm] Rickson EVM Extension shut down")
