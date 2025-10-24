"""
Rickson UI Panel

Main control panel with scrubbable EVM parameters, breath metrics,
and training insights following Bret Victor's principles.
"""

import omni.ui as ui
import carb
import asyncio
from typing import Optional

# Import EVM pipeline and video manager
# Note: These will be available once zs.evm extension is loaded
try:
    from zs.evm.core.evm_pipeline import EVMPipeline
    from zs.evm.video_input import get_video_manager
    EVM_AVAILABLE = True
except ImportError:
    carb.log_warn("[zs.ui] EVM modules not available yet")
    EVM_AVAILABLE = False
    EVMPipeline = None
    get_video_manager = None


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

        # Pipeline state
        self._pipeline: Optional[EVMPipeline] = None
        self._video_manager = None
        self._processing = False
        self._update_task = None

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

        # Update pipeline if running
        if self._pipeline is not None:
            self._pipeline.update_params(alpha=value)

    def _on_low_freq_changed(self, value):
        """Handle low frequency change."""
        self._evm_low_freq = value
        carb.log_info(f"[zs.ui] EVM Low Freq changed to {value:.2f} Hz")

        # Update pipeline if running
        if self._pipeline is not None:
            self._pipeline.update_params(low_freq=value)

    def _on_high_freq_changed(self, value):
        """Handle high frequency change."""
        self._evm_high_freq = value
        carb.log_info(f"[zs.ui] EVM High Freq changed to {value:.2f} Hz")

        # Update pipeline if running
        if self._pipeline is not None:
            self._pipeline.update_params(high_freq=value)

    def _on_levels_changed(self, value):
        """Handle pyramid levels change."""
        self._evm_levels = value
        carb.log_info(f"[zs.ui] EVM Pyramid Levels changed to {value}")

        # Note: Changing levels requires recreating pipeline
        if self._pipeline is not None and not self._processing:
            self._create_pipeline()

    # Button callbacks
    def _on_start_clicked(self):
        """Handle start button click."""
        carb.log_info("[zs.ui] Start capture clicked")

        if not EVM_AVAILABLE:
            self._insights_label.text = "Error: EVM modules not available"
            return

        if self._processing:
            carb.log_warn("[zs.ui] Already processing")
            return

        # Initialize pipeline
        self._create_pipeline()

        # Initialize video source (synthetic for now)
        self._video_manager = get_video_manager()
        self._video_manager.open_synthetic(
            width=640,
            height=480,
            fps=30.0,
            duration=60.0,
            breath_freq=0.3  # 18 BPM
        )

        # Start processing
        self._processing = True
        self._video_manager.play()
        self._insights_label.text = "Capturing... analyzing breath patterns..."

        # Start update loop
        self._start_update_loop()

    def _on_pause_clicked(self):
        """Handle pause button click."""
        carb.log_info("[zs.ui] Pause clicked")

        if self._video_manager is not None:
            self._video_manager.pause()

        self._processing = False
        self._insights_label.text = "Paused - adjust parameters and resume"

    def _on_reset_clicked(self):
        """Handle reset button click."""
        carb.log_info("[zs.ui] Reset clicked")

        # Stop processing
        self._processing = False

        # Reset metrics
        self._breath_rate = 0.0
        self._balance_score = 0.0
        self._breath_label.text = f"{self._breath_rate:.1f} BPM"
        self._balance_label.text = f"{self._balance_score:.1f}%"
        self._insights_label.text = "Reset - ready to capture"

        # Reset pipeline
        if self._pipeline is not None:
            self._pipeline.reset()

        # Reset video
        if self._video_manager is not None:
            self._video_manager.reset()

    def _on_explain_clicked(self):
        """Handle explain button click - show derivation graph."""
        carb.log_info("[zs.ui] Explain clicked")
        self._insights_label.text = (
            "Derivation: Breath rate derived from EVM temporal band-pass "
            f"({self._evm_low_freq:.2f}-{self._evm_high_freq:.2f} Hz) "
            f"with alpha={self._evm_alpha:.1f} over {self._evm_levels} pyramid levels."
        )
        # TODO: Open OmniGraph visualization window

    def _create_pipeline(self):
        """Create or recreate EVM pipeline with current parameters."""
        if not EVM_AVAILABLE:
            return

        fps = 30.0  # Default FPS
        if self._video_manager is not None:
            fps = self._video_manager.get_fps()

        self._pipeline = EVMPipeline(
            fps=fps,
            low_freq=self._evm_low_freq,
            high_freq=self._evm_high_freq,
            alpha=self._evm_alpha,
            pyramid_levels=self._evm_levels,
            buffer_seconds=10.0,
            wavelength_attenuation=True
        )

        carb.log_info(f"[zs.ui] Created EVM pipeline: "
                     f"alpha={self._evm_alpha}, "
                     f"band=[{self._evm_low_freq}, {self._evm_high_freq}] Hz, "
                     f"levels={self._evm_levels}")

    def _start_update_loop(self):
        """Start async update loop to process frames."""
        if self._update_task is not None:
            return  # Already running

        async def update_loop():
            """Process frames in loop."""
            carb.log_info("[zs.ui] Update loop started")

            while self._processing:
                # Read frame
                if self._video_manager is None:
                    break

                frame = self._video_manager.read_frame()

                if frame is None:
                    # End of video
                    carb.log_info("[zs.ui] End of video")
                    self._processing = False
                    self._insights_label.text = "Video completed. Click Start to replay."
                    break

                # Process through EVM pipeline
                if self._pipeline is not None:
                    try:
                        amplified_frame, metrics = self._pipeline.process_frame(frame)

                        # Update UI metrics
                        self._breath_rate = metrics['breath_rate_bpm']
                        self._breath_label.text = f"{self._breath_rate:.1f} BPM"

                        # Update confidence indicator
                        conf = metrics.get('breath_confidence', 0.0)
                        if conf > 0.8:
                            status = "High confidence"
                        elif conf > 0.5:
                            status = "Medium confidence"
                        elif conf > 0.2:
                            status = "Low confidence"
                        else:
                            status = "Warming up..."

                        if self._breath_rate > 0:
                            self._insights_label.text = (
                                f"Breathing at {self._breath_rate:.1f} BPM. {status}. "
                                f"Frame {metrics['frame_count']}"
                            )
                    except Exception as e:
                        carb.log_error(f"[zs.ui] Error processing frame: {e}")
                        self._processing = False
                        self._insights_label.text = f"Error: {e}"
                        break

                # Yield control (run at ~30fps)
                await asyncio.sleep(0.033)

            carb.log_info("[zs.ui] Update loop stopped")
            self._update_task = None

        # Start task
        self._update_task = asyncio.ensure_future(update_loop())

    def destroy(self):
        """Clean up resources."""
        carb.log_info("[zs.ui] Panel destroyed")

        # Stop processing
        self._processing = False

        # Release video
        if self._video_manager is not None:
            self._video_manager.release()

        # Cancel update task
        if self._update_task is not None:
            self._update_task.cancel()
