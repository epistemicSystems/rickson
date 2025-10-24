"""
USD Timeline Replay

Replay recorded sessions with scrubbing, annotations, and event playback.
Integrates with USD timeline for frame-accurate replay.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from .event_log import EventLog, Event


@dataclass
class TimelineMarker:
    """
    Timeline marker for notable events.

    Attributes:
        time: Time in seconds
        label: Marker label
        event_type: Event type
        color: Marker color (RGB)
        metadata: Additional data
    """
    time: float
    label: str
    event_type: str
    color: tuple = (1.0, 0.5, 0.0)
    metadata: Dict[str, Any] = None


class TimelinePlayer:
    """
    Timeline player for session replay.

    Features:
    - Frame-accurate playback
    - Speed control (0.1x - 10x)
    - Scrubbing
    - Event markers
    - Annotation overlay
    """

    def __init__(self, event_log: EventLog):
        """
        Initialize timeline player.

        Args:
            event_log: EventLog to replay
        """
        self.event_log = event_log

        # Playback state
        self.current_time = 0.0
        self.is_playing = False
        self.playback_speed = 1.0

        # Timeline bounds
        self.start_time = 0.0
        self.end_time = self._compute_duration()

        # Markers
        self.markers: List[TimelineMarker] = []
        self._build_markers()

        # Frame callbacks
        self.frame_callbacks: List[Callable] = []

    def _compute_duration(self) -> float:
        """Compute total duration from events."""
        if not self.event_log.events:
            return 0.0

        # Find session end
        for event in reversed(self.event_log.events):
            if event.event_type == 'session_ended':
                return event.timestamp

        # Use last event timestamp
        return self.event_log.events[-1].timestamp

    def _build_markers(self):
        """Build timeline markers from events."""
        self.markers.clear()

        for event in self.event_log.events:
            # Add markers for significant events
            if event.event_type in ['alert_balance_edge', 'breath_hold_detected']:
                marker = TimelineMarker(
                    time=event.timestamp,
                    label=event.event_type.replace('_', ' ').title(),
                    event_type=event.event_type,
                    color=(1.0, 0.2, 0.2) if 'alert' in event.event_type else (0.2, 0.8, 1.0),
                    metadata=event.data
                )
                self.markers.append(marker)

    def play(self):
        """Start playback."""
        self.is_playing = True

    def pause(self):
        """Pause playback."""
        self.is_playing = False

    def stop(self):
        """Stop and reset."""
        self.is_playing = False
        self.current_time = 0.0

    def seek(self, time: float):
        """
        Seek to specific time.

        Args:
            time: Time in seconds
        """
        self.current_time = np.clip(time, self.start_time, self.end_time)

    def set_speed(self, speed: float):
        """
        Set playback speed.

        Args:
            speed: Speed multiplier (0.1 - 10.0)
        """
        self.playback_speed = np.clip(speed, 0.1, 10.0)

    def update(self, dt: float) -> List[Event]:
        """
        Update playback for frame.

        Args:
            dt: Delta time (seconds)

        Returns:
            List of events that occurred this frame
        """
        if not self.is_playing:
            return []

        # Advance time
        prev_time = self.current_time
        self.current_time += dt * self.playback_speed

        # Clamp to bounds
        if self.current_time >= self.end_time:
            self.current_time = self.end_time
            self.is_playing = False

        # Get events in time range
        events = self._get_events_in_range(prev_time, self.current_time)

        # Trigger callbacks
        for callback in self.frame_callbacks:
            callback(self.current_time, events)

        return events

    def _get_events_in_range(self, start: float, end: float) -> List[Event]:
        """Get events between start and end time."""
        events = []

        for event in self.event_log.events:
            if start <= event.timestamp <= end:
                events.append(event)

        return events

    def get_state_at_time(self, time: float) -> Dict[str, Any]:
        """
        Get session state at specific time.

        Reconstructs state by replaying events up to time.

        Args:
            time: Time in seconds

        Returns:
            State dictionary
        """
        state = {}

        # Replay events up to time
        for event in self.event_log.events:
            if event.timestamp > time:
                break

            # Update state based on event
            if event.event_type == 'frame_processed':
                state['frame_number'] = event.data.get('frame_number', 0)

            elif event.event_type == 'breath_rate_estimated':
                state['breath_rate'] = event.data.get('breath_rate_bpm', 0.0)
                state['breath_confidence'] = event.data.get('confidence', 0.0)

            elif event.event_type == 'pose_estimated':
                state['pose_keypoints'] = event.data.get('keypoints', {})
                state['balance_score'] = event.data.get('balance_score', 0.0)

            elif event.event_type == 'insight_generated':
                if 'insights' not in state:
                    state['insights'] = []
                state['insights'].append(event.data)

        return state

    def get_markers_in_range(self, start: float, end: float) -> List[TimelineMarker]:
        """Get markers between start and end time."""
        return [m for m in self.markers if start <= m.time <= end]

    def add_annotation(
        self,
        time: float,
        label: str,
        annotation_type: str = 'user',
        color: tuple = (0.0, 1.0, 0.0)
    ):
        """
        Add user annotation to timeline.

        Args:
            time: Time in seconds
            label: Annotation text
            annotation_type: Annotation type
            color: Marker color
        """
        marker = TimelineMarker(
            time=time,
            label=label,
            event_type=annotation_type,
            color=color
        )

        self.markers.append(marker)
        self.markers.sort(key=lambda m: m.time)

    def export_annotations(self, path: str):
        """Export annotations to JSON."""
        annotations = []

        for marker in self.markers:
            annotations.append({
                'time': marker.time,
                'label': marker.label,
                'event_type': marker.event_type,
                'color': marker.color,
                'metadata': marker.metadata
            })

        with open(path, 'w') as f:
            json.dump(annotations, f, indent=2)

    def load_annotations(self, path: str):
        """Load annotations from JSON."""
        with open(path) as f:
            annotations = json.load(f)

        for ann in annotations:
            marker = TimelineMarker(
                time=ann['time'],
                label=ann['label'],
                event_type=ann['event_type'],
                color=tuple(ann['color']),
                metadata=ann.get('metadata')
            )
            self.markers.append(marker)

        self.markers.sort(key=lambda m: m.time)

    def get_progress(self) -> float:
        """Get playback progress [0, 1]."""
        if self.end_time <= 0:
            return 0.0

        return self.current_time / self.end_time


def test_timeline_player():
    """Test timeline player."""
    print("Testing Timeline Player...")

    # Create event log
    log = EventLog()

    log.log_event('session_started', {'session_id': 'test_001'}, timestamp=0.0)

    for i in range(100):
        t = i * 0.033  # 30fps

        log.log_event('frame_processed', {'frame_number': i}, timestamp=t)

        if i % 30 == 0:
            log.log_event('breath_rate_estimated', {
                'breath_rate_bpm': 18.0,
                'confidence': 0.9
            }, timestamp=t)

        if i == 50:
            log.log_event('alert_balance_edge', {
                'balance_score': 45,
                'message': 'Balance edge detected'
            }, timestamp=t)

    log.log_event('session_ended', {}, timestamp=3.3)

    # Create player
    player = TimelinePlayer(log)

    print(f"Duration: {player.end_time:.2f}s")
    print(f"Markers: {len(player.markers)}")

    # Test playback
    player.play()
    player.set_speed(2.0)

    while player.is_playing:
        events = player.update(0.033)

        if events:
            print(f"  t={player.current_time:.2f}s: {len(events)} events")

    print(f"Playback complete at t={player.current_time:.2f}s")

    # Test seeking
    player.seek(1.5)
    state = player.get_state_at_time(1.5)

    print(f"\nState at t=1.5s:")
    print(f"  Frame: {state.get('frame_number', 'N/A')}")
    print(f"  Breath rate: {state.get('breath_rate', 'N/A')} BPM")

    # Test annotations
    player.add_annotation(1.0, "Good technique here", "user")
    player.add_annotation(2.0, "Watch footwork", "coach")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        temp_path = f.name

    player.export_annotations(temp_path)

    new_player = TimelinePlayer(log)
    new_player.load_annotations(temp_path)

    print(f"\nAnnotations loaded: {len(new_player.markers)}")

    import os
    os.unlink(temp_path)

    print("✓ PASS")


if __name__ == "__main__":
    test_timeline_player()
