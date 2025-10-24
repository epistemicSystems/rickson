"""
Rickson UI Extension

Provides the main UI panel with scrubbable EVM parameters,
breath metrics, and training insights display.
"""

import omni.ext
import omni.ui as ui
import carb

from .ui_panel import RicksonUIPanel


class RicksonUIExtension(omni.ext.IExt):
    """Main UI extension for Rickson training assistant."""

    def on_startup(self, ext_id):
        """Called when the extension starts up."""
        carb.log_info("[zs.ui] Rickson UI Extension starting...")

        # Create the main UI panel
        self._window = ui.Window("Rickson Training Assistant", width=400, height=600)
        with self._window.frame:
            self._panel = RicksonUIPanel()

        carb.log_info("[zs.ui] Rickson UI Extension started successfully")

    def on_shutdown(self):
        """Called when the extension is shutting down."""
        carb.log_info("[zs.ui] Rickson UI Extension shutting down...")

        if hasattr(self, '_panel'):
            self._panel.destroy()

        if hasattr(self, '_window'):
            self._window.destroy()
            self._window = None

        carb.log_info("[zs.ui] Rickson UI Extension shut down")
