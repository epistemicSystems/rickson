"""
Timeline Replay System

Replay training sessions from event logs with USD timeline integration.
Enables scrubbing through sessions, adding annotations, and analyzing
specific moments.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import carb

from .event_log import EventLog, Event, EventTypes


@dataclass
class TimelineFrame:
    """Single frame in timeline with all associated data."""

    frame_number: int
    timestamp: float
    breath_rate_bpm: float
    breath_confidence: float
    balance_score: float
    stance_type: str
    com_position: Optional[Tuple[float, float]]
    support_polygon: Optional[np.ndarray]
    alerts: List[str]
    annotations: List[Dict[str, Any]]


class TimelineReplay:
    """
    Replay training sessions from event logs.

    Provides timeline navigation, scrubbing, and annotation capabilities.
    """

    def __init__(self, event_log: EventLog):
        """
        Initialize timeline replay from event log.

        Args:
            event_log: EventLog to replay
        """
        self.event_log = event_log
        self.frames: List[TimelineFrame] = []
        self.current_frame_idx = 0

        self._build_timeline()

    def _build_timeline(self):
        """Build timeline from events by reducing state."""
        carb.log_info("[TimelineReplay] Building timeline from event log...")

        # Group events by frame
        frame_events: Dict[int, List[Event]] = {}

        for event in self.event_log.events:
            if event.event_type == EventTypes.FRAME_INGESTED:
                frame_num = event.data.get('frame', 0)
                if frame_num not in frame_events:
                    frame_events[frame_num] = []
                frame_events[frame_num].append(event)
            elif event.event_type in [
                EventTypes.POSE_ESTIMATED,
                EventTypes.EVM_BREATH_CYCLE,
                EventTypes.ALERT_BALANCE_EDGE,
                EventTypes.ALERT_BREATH_HOLD,
                EventTypes.ANNOTATION_ADDED
            ]:
                frame_num = event.data.get('frame', 0)
                if frame_num not in frame_events:
                    frame_events[frame_num] = []
                frame_events[frame_num].append(event)

        # Build frames
        for frame_num in sorted(frame_events.keys()):
            events = frame_events[frame_num]

            # Extract data from events
            breath_rate = 0.0
            breath_conf = 0.0
            balance_score = 0.0
            stance_type = "unknown"
            com_pos = None
            support_poly = None
            alerts = []
            annotations = []
            timestamp = events[0].timestamp if events else 0.0

            for event in events:
                if event.event_type == EventTypes.EVM_BREATH_CYCLE:
                    breath_rate = event.data.get('breath_rate_bpm', 0.0)
                    breath_conf = event.data.get('confidence', 0.0)

                elif event.event_type == EventTypes.POSE_ESTIMATED:
                    balance_score = event.data.get('balance_score', 0.0)
                    stance_type = event.data.get('stance_type', 'unknown')
                    if 'com_position' in event.data:
                        com_pos = tuple(event.data['com_position'])
                    if 'support_polygon' in event.data:
                        support_poly = np.array(event.data['support_polygon'])

                elif event.event_type == EventTypes.ALERT_BALANCE_EDGE:
                    alerts.append(f"Balance edge: score={event.data.get('balance_score', 0):.0f}")

                elif event.event_type == EventTypes.ALERT_BREATH_HOLD:
                    alerts.append("Breath hold detected")

                elif event.event_type == EventTypes.ANNOTATION_ADDED:
                    annotations.append(event.data)

            # Create frame
            frame = TimelineFrame(
                frame_number=frame_num,
                timestamp=timestamp,
                breath_rate_bpm=breath_rate,
                breath_confidence=breath_conf,
                balance_score=balance_score,
                stance_type=stance_type,
                com_position=com_pos,
                support_polygon=support_poly,
                alerts=alerts,
                annotations=annotations
            )

            self.frames.append(frame)

        carb.log_info(f"[TimelineReplay] Built timeline with {len(self.frames)} frames")

    def get_frame(self, index: int) -> Optional[TimelineFrame]:
        """Get frame by index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def get_current_frame(self) -> Optional[TimelineFrame]:
        """Get current frame."""
        return self.get_frame(self.current_frame_idx)

    def seek(self, index: int):
        """Seek to specific frame index."""
        if 0 <= index < len(self.frames):
            self.current_frame_idx = index
            carb.log_info(f"[TimelineReplay] Seeked to frame {index}")

    def seek_time(self, timestamp: float):
        """Seek to specific timestamp."""
        # Find closest frame by timestamp
        if not self.frames:
            return

        closest_idx = 0
        min_diff = abs(self.frames[0].timestamp - timestamp)

        for i, frame in enumerate(self.frames):
            diff = abs(frame.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        self.seek(closest_idx)

    def next_frame(self) -> Optional[TimelineFrame]:
        """Advance to next frame."""
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            return self.get_current_frame()
        return None

    def prev_frame(self) -> Optional[TimelineFrame]:
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            return self.get_current_frame()
        return None

    def find_alerts(self) -> List[Tuple[int, TimelineFrame]]:
        """Find all frames with alerts."""
        alert_frames = []
        for i, frame in enumerate(self.frames):
            if frame.alerts:
                alert_frames.append((i, frame))
        return alert_frames

    def find_annotations(self) -> List[Tuple[int, TimelineFrame]]:
        """Find all frames with annotations."""
        annotated_frames = []
        for i, frame in enumerate(self.frames):
            if frame.annotations:
                annotated_frames.append((i, frame))
        return annotated_frames

    def get_summary(self) -> Dict[str, Any]:
        """Get timeline summary statistics."""
        if not self.frames:
            return {}

        breath_rates = [f.breath_rate_bpm for f in self.frames if f.breath_rate_bpm > 0]
        balance_scores = [f.balance_score for f in self.frames if f.balance_score > 0]

        alert_frames = self.find_alerts()
        annotated_frames = self.find_annotations()

        summary = {
            'total_frames': len(self.frames),
            'duration_seconds': self.frames[-1].timestamp - self.frames[0].timestamp if self.frames else 0,
            'breath_analysis': {
                'mean_bpm': np.mean(breath_rates) if breath_rates else 0,
                'std_bpm': np.std(breath_rates) if breath_rates else 0,
                'min_bpm': np.min(breath_rates) if breath_rates else 0,
                'max_bpm': np.max(breath_rates) if breath_rates else 0,
            },
            'balance_analysis': {
                'mean_score': np.mean(balance_scores) if balance_scores else 0,
                'std_score': np.std(balance_scores) if balance_scores else 0,
                'min_score': np.min(balance_scores) if balance_scores else 0,
                'max_score': np.max(balance_scores) if balance_scores else 0,
            },
            'alerts_count': len(alert_frames),
            'annotations_count': len(annotated_frames),
        }

        return summary

    def export_to_usd_timeline(self, output_path: Path):
        """
        Export timeline to USD format.

        Creates USD stage with timecoded attributes for breath rate,
        balance score, and other metrics.

        Args:
            output_path: Path to output USD file
        """
        # This would require omni.usd to be available
        # For now, export as JSON with USD-compatible structure

        timeline_data = {
            'frames': [],
            'summary': self.get_summary()
        }

        for frame in self.frames:
            frame_data = {
                'frame': frame.frame_number,
                'time': frame.timestamp,
                'breath_rate_bpm': frame.breath_rate_bpm,
                'breath_confidence': frame.breath_confidence,
                'balance_score': frame.balance_score,
                'stance_type': frame.stance_type,
                'com_position': list(frame.com_position) if frame.com_position else None,
                'support_polygon': frame.support_polygon.tolist() if frame.support_polygon is not None else None,
                'alerts': frame.alerts,
                'annotations': frame.annotations
            }
            timeline_data['frames'].append(frame_data)

        import json
        with open(output_path, 'w') as f:
            json.dump(timeline_data, f, indent=2)

        carb.log_info(f"[TimelineReplay] Exported timeline to {output_path}")


class TimelineAnnotator:
    """
    Add annotations to timeline frames.

    Annotations are stored as events in the log for immutability.
    """

    def __init__(self, event_log: EventLog):
        """Initialize annotator."""
        self.event_log = event_log

    def add_annotation(
        self,
        frame_number: int,
        annotation_type: str,
        text: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add annotation to specific frame.

        Args:
            frame_number: Frame to annotate
            annotation_type: Type of annotation ('technique', 'error', 'note', etc.)
            text: Annotation text
            metadata: Optional additional metadata
        """
        annotation_data = {
            'frame': frame_number,
            'type': annotation_type,
            'text': text,
        }

        if metadata:
            annotation_data.update(metadata)

        self.event_log.append_event(EventTypes.ANNOTATION_ADDED, annotation_data)
        carb.log_info(f"[TimelineAnnotator] Added annotation to frame {frame_number}: {text}")

    def get_annotations(self, frame_number: Optional[int] = None) -> List[Dict]:
        """
        Get annotations, optionally filtered by frame.

        Args:
            frame_number: If specified, only return annotations for this frame

        Returns:
            List of annotation data dicts
        """
        annotation_events = self.event_log.get_events(EventTypes.ANNOTATION_ADDED)

        annotations = []
        for event in annotation_events:
            if frame_number is None or event.data.get('frame') == frame_number:
                annotations.append(event.data)

        return annotations


def test_timeline_replay():
    """Test timeline replay with synthetic event log."""
    print("\nTesting Timeline Replay...")
    print("="*70)

    # Create synthetic event log
    from .event_log import EventLog

    log = EventLog()

    # Session start
    log.append_event(EventTypes.SESSION_STARTED, {
        'duration': 10.0,
        'fps': 30.0
    })

    # Generate frame events
    for frame_num in range(100):
        timestamp = frame_num / 30.0

        # Frame ingested
        log.append_event(EventTypes.FRAME_INGESTED, {
            'frame': frame_num,
            'timestamp': timestamp
        })

        # Breath cycle (every 10 frames)
        if frame_num % 10 == 0:
            log.append_event(EventTypes.EVM_BREATH_CYCLE, {
                'frame': frame_num,
                'breath_rate_bpm': 18.0 + np.random.randn() * 2,
                'confidence': 0.9
            })

        # Pose estimated (every frame)
        log.append_event(EventTypes.POSE_ESTIMATED, {
            'frame': frame_num,
            'balance_score': 75.0 + np.random.randn() * 10,
            'stance_type': 'parallel' if frame_num < 50 else 'staggered',
            'com_position': [0.5, 0.6]
        })

        # Alert (occasionally)
        if frame_num in [25, 75]:
            log.append_event(EventTypes.ALERT_BALANCE_EDGE, {
                'frame': frame_num,
                'balance_score': 35.0
            })

    # Session end
    log.append_event(EventTypes.SESSION_ENDED, {
        'frames_processed': 100
    })

    # Create timeline
    replay = TimelineReplay(log)

    print(f"✓ Created timeline with {len(replay.frames)} frames")

    # Test navigation
    replay.seek(50)
    frame = replay.get_current_frame()
    print(f"✓ Seek to frame 50: breath={frame.breath_rate_bpm:.1f} BPM, balance={frame.balance_score:.0f}")

    # Test alerts
    alert_frames = replay.find_alerts()
    print(f"✓ Found {len(alert_frames)} frames with alerts")

    # Test summary
    summary = replay.get_summary()
    print(f"✓ Summary: {summary['total_frames']} frames, {summary['duration_seconds']:.1f}s")
    print(f"  Mean breath: {summary['breath_analysis']['mean_bpm']:.1f} BPM")
    print(f"  Mean balance: {summary['balance_analysis']['mean_score']:.0f}/100")

    # Test annotations
    annotator = TimelineAnnotator(log)
    annotator.add_annotation(50, 'technique', 'Good transition to staggered stance')
    annotator.add_annotation(25, 'error', 'Balance edge - widen stance')

    annotations = annotator.get_annotations()
    print(f"✓ Added {len(annotations)} annotations")

    # Export
    output_path = Path("/tmp/timeline_test.json")
    replay.export_to_usd_timeline(output_path)
    print(f"✓ Exported timeline to {output_path}")

    print("\n" + "="*70)
    print("Timeline Replay Test: PASS")
    return True


if __name__ == "__main__":
    test_timeline_replay()
