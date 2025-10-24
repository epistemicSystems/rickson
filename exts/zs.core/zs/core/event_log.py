"""
Event Log System

Immutable append-only event log following Rich Hickey's principles.
State is derived by reducing events.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import carb


class Event:
    """Immutable event."""

    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: Optional[float] = None):
        """
        Create event.

        Args:
            event_type: Event type (e.g., 'session_started', 'frame_processed')
            data: Event data
            timestamp: Unix timestamp (auto-generated if None)
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp if timestamp is not None else datetime.now().timestamp()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'data': self.data
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(d: Dict) -> 'Event':
        """Create event from dictionary."""
        return Event(d['event_type'], d['data'], d['timestamp'])

    @staticmethod
    def from_json(s: str) -> 'Event':
        """Create event from JSON string."""
        d = json.loads(s)
        return Event.from_dict(d)


class EventLog:
    """
    Append-only event log.

    Following Rich Hickey's principles:
    - Events are immutable
    - Log is append-only
    - State is derived by reducing events
    """

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize event log.

        Args:
            log_file: Optional file path to persist log
        """
        self.events: List[Event] = []
        self.log_file = log_file

        if log_file is not None and log_file.exists():
            self._load_from_file()

    def append(self, event: Event):
        """Append event to log."""
        self.events.append(event)

        # Persist if log file specified
        if self.log_file is not None:
            self._append_to_file(event)

        carb.log_info(f"[EventLog] {event.event_type}: {event.data}")

    def append_event(self, event_type: str, data: Dict[str, Any]):
        """Convenience method to create and append event."""
        event = Event(event_type, data)
        self.append(event)

    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Event]:
        """
        Query events with filters.

        Args:
            event_type: Filter by event type
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp

        Returns:
            List of matching events
        """
        filtered = self.events

        if event_type is not None:
            filtered = [e for e in filtered if e.event_type == event_type]

        if start_time is not None:
            filtered = [e for e in filtered if e.timestamp >= start_time]

        if end_time is not None:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        return filtered

    def reduce_state(self, reducer_fn) -> Any:
        """
        Derive state by reducing events.

        Args:
            reducer_fn: Function (state, event) -> new_state

        Returns:
            Final state
        """
        state = None
        for event in self.events:
            state = reducer_fn(state, event)
        return state

    def _load_from_file(self):
        """Load events from file."""
        if self.log_file is None or not self.log_file.exists():
            return

        with open(self.log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        event = Event.from_json(line)
                        self.events.append(event)
                    except Exception as e:
                        carb.log_warn(f"[EventLog] Failed to parse event: {e}")

        carb.log_info(f"[EventLog] Loaded {len(self.events)} events from {self.log_file}")

    def _append_to_file(self, event: Event):
        """Append event to file."""
        if self.log_file is None:
            return

        # Create parent directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, 'a') as f:
            f.write(event.to_json() + '\n')

    def export_json(self, output_path: Path):
        """Export entire log as JSON array."""
        with open(output_path, 'w') as f:
            events_dict = [e.to_dict() for e in self.events]
            json.dump(events_dict, f, indent=2)

        carb.log_info(f"[EventLog] Exported {len(self.events)} events to {output_path}")

    def clear(self):
        """Clear log (for testing only - normally append-only!)."""
        self.events.clear()


# Global event log instance
_event_log: Optional[EventLog] = None


def get_event_log() -> EventLog:
    """Get global event log instance."""
    global _event_log
    if _event_log is None:
        # Default log file in user's data directory
        log_dir = Path.home() / ".rickson" / "logs"
        log_file = log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        _event_log = EventLog(log_file)
    return _event_log


def log_event(event_type: str, data: Dict[str, Any]):
    """Convenience function to log event to global log."""
    get_event_log().append_event(event_type, data)


# Event type constants
class EventTypes:
    """Standard event types."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    FRAME_INGESTED = "frame_ingested"
    POSE_ESTIMATED = "pose_estimated"
    EVM_BREATH_CYCLE = "evm_breath_cycle"
    ALERT_BALANCE_EDGE = "alert_balance_edge"
    ALERT_BREATH_HOLD = "alert_breath_hold"
    ANNOTATION_ADDED = "annotation_added"
    PARAMETER_CHANGED = "parameter_changed"
