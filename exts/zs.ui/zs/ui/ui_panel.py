"""
Rickson UI Panel

Main control panel with scrubbable EVM parameters, breath metrics,
and training insights following Bret Victor's principles.
"""

import omni.ui as ui
import carb


class RicksonUIPanel:
    """Main UI panel for Rickson training assistant."""

    def __init__(self):
        """Initialize the UI panel."""
        self._evm_alpha = 10.0  # Amplification factor
        self._evm_low_freq = 0.1  # Hz - lower breathing band
        self._evm_high_freq = 0.7  # Hz - upper breathing band
        self._evm_levels = 4  # Spatial pyramid levels

        self._breath_rate = 0.0  # Breaths per minute
        self._balance_score = 0.0  # 0-100

        self._build_ui()

    def _build_ui(self):
        """Build the UI layout."""
        with ui.VStack(spacing=10, height=0):
            # Header
            ui.Label(
                "Rickson Training Assistant",
                height=30,
                style={"font_size": 18, "color": 0xFF00FF00}
            )

            ui.Spacer(height=5)
            ui.Separator()
            ui.Spacer(height=10)

            # EVM Parameters Section
            with ui.CollapsableFrame("EVM Parameters (Breath Analysis)", height=0, collapsed=False):
                with ui.VStack(spacing=8, height=0):
                    self._build_slider(
                        "Alpha (Amplification)",
                        self._evm_alpha,
                        0.0, 50.0,
                        lambda v: self._on_alpha_changed(v)
                    )

                    self._build_slider(
                        "Low Freq (Hz)",
                        self._evm_low_freq,
                        0.05, 2.0,
                        lambda v: self._on_low_freq_changed(v)
                    )

                    self._build_slider(
                        "High Freq (Hz)",
                        self._evm_high_freq,
                        0.1, 3.0,
                        lambda v: self._on_high_freq_changed(v)
                    )

                    self._build_int_slider(
                        "Pyramid Levels",
                        self._evm_levels,
                        2, 8,
                        lambda v: self._on_levels_changed(v)
                    )

            ui.Spacer(height=10)

            # Metrics Section
            with ui.CollapsableFrame("Live Metrics", height=0, collapsed=False):
                with ui.VStack(spacing=8, height=0):
                    with ui.HStack(height=0):
                        ui.Label("Breath Rate:", width=120)
                        self._breath_label = ui.Label(
                            f"{self._breath_rate:.1f} BPM",
                            style={"color": 0xFF00AAFF}
                        )

                    with ui.HStack(height=0):
                        ui.Label("Balance Score:", width=120)
                        self._balance_label = ui.Label(
                            f"{self._balance_score:.1f}%",
                            style={"color": 0xFF00AAFF}
                        )

            ui.Spacer(height=10)

            # Insights Section
            with ui.CollapsableFrame("Insights", height=0, collapsed=False):
                with ui.VStack(spacing=5, height=0):
                    self._insights_label = ui.Label(
                        "Waiting for data...",
                        word_wrap=True,
                        style={"color": 0xFFAAAAAA}
                    )

                    ui.Spacer(height=5)

                    ui.Button(
                        "Explain This Alert",
                        height=30,
                        clicked_fn=self._on_explain_clicked
                    )

            ui.Spacer(height=10)

            # Controls
            with ui.HStack(spacing=5, height=0):
                ui.Button(
                    "Start Capture",
                    height=40,
                    clicked_fn=self._on_start_clicked
                )
                ui.Button(
                    "Pause",
                    height=40,
                    clicked_fn=self._on_pause_clicked
                )
                ui.Button(
                    "Reset",
                    height=40,
                    clicked_fn=self._on_reset_clicked
                )

    def _build_slider(self, label, value, min_val, max_val, callback):
        """Build a labeled float slider."""
        with ui.VStack(spacing=3, height=0):
            with ui.HStack(height=0):
                ui.Label(label, width=150)
                value_label = ui.Label(f"{value:.2f}", width=50)

            slider = ui.FloatSlider(
                min=min_val,
                max=max_val,
                height=20
            )
            slider.model.set_value(value)

            def on_value_changed(model):
                v = model.get_value_as_float()
                value_label.text = f"{v:.2f}"
                callback(v)

            slider.model.add_value_changed_fn(on_value_changed)

    def _build_int_slider(self, label, value, min_val, max_val, callback):
        """Build a labeled integer slider."""
        with ui.VStack(spacing=3, height=0):
            with ui.HStack(height=0):
                ui.Label(label, width=150)
                value_label = ui.Label(f"{value}", width=50)

            slider = ui.IntSlider(
                min=min_val,
                max=max_val,
                height=20
            )
            slider.model.set_value(value)

            def on_value_changed(model):
                v = model.get_value_as_int()
                value_label.text = f"{v}"
                callback(v)

            slider.model.add_value_changed_fn(on_value_changed)

    # Parameter change callbacks
    def _on_alpha_changed(self, value):
        """Handle alpha parameter change."""
        self._evm_alpha = value
        carb.log_info(f"[zs.ui] EVM Alpha changed to {value:.2f}")
        # TODO: Send to EVM compute node via OmniGraph

    def _on_low_freq_changed(self, value):
        """Handle low frequency change."""
        self._evm_low_freq = value
        carb.log_info(f"[zs.ui] EVM Low Freq changed to {value:.2f} Hz")
        # TODO: Send to EVM compute node via OmniGraph

    def _on_high_freq_changed(self, value):
        """Handle high frequency change."""
        self._evm_high_freq = value
        carb.log_info(f"[zs.ui] EVM High Freq changed to {value:.2f} Hz")
        # TODO: Send to EVM compute node via OmniGraph

    def _on_levels_changed(self, value):
        """Handle pyramid levels change."""
        self._evm_levels = value
        carb.log_info(f"[zs.ui] EVM Pyramid Levels changed to {value}")
        # TODO: Send to EVM compute node via OmniGraph

    # Button callbacks
    def _on_start_clicked(self):
        """Handle start button click."""
        carb.log_info("[zs.ui] Start capture clicked")
        self._insights_label.text = "Capturing... analyzing breath patterns..."
        # TODO: Start capture pipeline

    def _on_pause_clicked(self):
        """Handle pause button click."""
        carb.log_info("[zs.ui] Pause clicked")
        self._insights_label.text = "Paused - adjust parameters and resume"
        # TODO: Pause capture pipeline

    def _on_reset_clicked(self):
        """Handle reset button click."""
        carb.log_info("[zs.ui] Reset clicked")
        self._breath_rate = 0.0
        self._balance_score = 0.0
        self._breath_label.text = f"{self._breath_rate:.1f} BPM"
        self._balance_label.text = f"{self._balance_score:.1f}%"
        self._insights_label.text = "Reset - ready to capture"
        # TODO: Reset pipeline state

    def _on_explain_clicked(self):
        """Handle explain button click - show derivation graph."""
        carb.log_info("[zs.ui] Explain clicked")
        self._insights_label.text = (
            "Derivation: Breath rate derived from EVM temporal band-pass "
            f"({self._evm_low_freq:.2f}-{self._evm_high_freq:.2f} Hz) "
            f"with alpha={self._evm_alpha:.1f} over {self._evm_levels} pyramid levels."
        )
        # TODO: Open OmniGraph visualization window

    def destroy(self):
        """Clean up resources."""
        carb.log_info("[zs.ui] Panel destroyed")
