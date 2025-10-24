"""
Opponent Feature Extraction

Analyzes opponent footage to extract:
- Stance preferences (orthodox/southpaw, weight distribution)
- Movement patterns (footwork, pressure style)
- Attack frequencies (guard pulls, takedowns, strikes)
- Defensive tendencies (distance management, counter timing)
- Breath patterns (conditioning, fatigue onset)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json


@dataclass
class OpponentProfile:
    """
    Opponent fighting profile extracted from footage.

    Attributes:
        name: Opponent identifier
        stance: Primary stance ('orthodox', 'southpaw', 'switch')
        stance_distribution: % time in each stance
        pressure_style: 'aggressive', 'counter', 'balanced'
        movement_speed: Average movement speed (normalized)
        guard_pull_rate: Guard pulls per minute
        takedown_rate: Takedown attempts per minute
        strike_rate: Strikes per minute
        breath_rate_mean: Average breath rate (BPM)
        fatigue_onset: Time to fatigue (seconds)
        defensive_distance: Preferred defensive distance (meters)
        tendencies: Dict of specific patterns/tells
    """
    name: str
    stance: str = 'unknown'
    stance_distribution: Dict[str, float] = field(default_factory=dict)
    pressure_style: str = 'balanced'
    movement_speed: float = 0.0
    guard_pull_rate: float = 0.0
    takedown_rate: float = 0.0
    strike_rate: float = 0.0
    breath_rate_mean: float = 0.0
    fatigue_onset: float = 0.0
    defensive_distance: float = 0.0
    tendencies: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            'name': self.name,
            'stance': self.stance,
            'stance_distribution': self.stance_distribution,
            'pressure_style': self.pressure_style,
            'movement_speed': self.movement_speed,
            'guard_pull_rate': self.guard_pull_rate,
            'takedown_rate': self.takedown_rate,
            'strike_rate': self.strike_rate,
            'breath_rate_mean': self.breath_rate_mean,
            'fatigue_onset': self.fatigue_onset,
            'defensive_distance': self.defensive_distance,
            'tendencies': self.tendencies
        }

    def save(self, path: str):
        """Save profile to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> 'OpponentProfile':
        """Load profile from JSON."""
        with open(path) as f:
            data = json.load(f)
        return OpponentProfile(**data)


class OpponentFeatureExtractor:
    """
    Extract opponent features from footage.

    Analyzes pose, movement, and EVM breath data to build OpponentProfile.
    """

    def __init__(self):
        """Initialize feature extractor."""
        # Buffers for accumulating data
        self.pose_history: List[Dict] = []
        self.breath_history: List[float] = []
        self.timestamps: List[float] = []

        # Event counters
        self.events: Counter = Counter()

    def add_frame_data(
        self,
        timestamp: float,
        pose_keypoints: Optional[Dict] = None,
        breath_rate: Optional[float] = None,
        stance_type: Optional[str] = None
    ):
        """
        Add frame analysis data.

        Args:
            timestamp: Frame timestamp
            pose_keypoints: Pose keypoints (2D or 3D)
            breath_rate: Breath rate (BPM)
            stance_type: Detected stance
        """
        self.timestamps.append(timestamp)

        if pose_keypoints:
            self.pose_history.append({
                'timestamp': timestamp,
                'keypoints': pose_keypoints,
                'stance': stance_type
            })

        if breath_rate:
            self.breath_history.append(breath_rate)

    def add_event(self, event_type: str):
        """
        Add discrete event.

        Args:
            event_type: 'guard_pull', 'takedown', 'strike', etc.
        """
        self.events[event_type] += 1

    def extract_profile(self, opponent_name: str) -> OpponentProfile:
        """
        Extract complete opponent profile from accumulated data.

        Args:
            opponent_name: Opponent identifier

        Returns:
            OpponentProfile
        """
        if not self.pose_history:
            return OpponentProfile(name=opponent_name)

        duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 1.0
        duration_minutes = duration / 60.0

        # Stance analysis
        stances = [frame.get('stance') for frame in self.pose_history if frame.get('stance')]
        stance_counts = Counter(stances)

        if stance_counts:
            primary_stance = stance_counts.most_common(1)[0][0]
            total = sum(stance_counts.values())
            stance_dist = {k: v / total for k, v in stance_counts.items()}
        else:
            primary_stance = 'unknown'
            stance_dist = {}

        # Movement speed (from pose deltas)
        movement_speed = self._compute_movement_speed()

        # Pressure style (from distance and aggression)
        pressure_style = self._classify_pressure_style()

        # Event rates
        guard_pull_rate = self.events['guard_pull'] / duration_minutes if duration_minutes > 0 else 0
        takedown_rate = self.events['takedown'] / duration_minutes if duration_minutes > 0 else 0
        strike_rate = self.events['strike'] / duration_minutes if duration_minutes > 0 else 0

        # Breath analysis
        breath_mean = np.mean(self.breath_history) if self.breath_history else 0.0

        # Fatigue detection (when breath rate increases by 20%)
        fatigue_onset = self._detect_fatigue_onset()

        # Defensive distance
        defensive_distance = self._compute_defensive_distance()

        # Tendencies
        tendencies = self._extract_tendencies()

        return OpponentProfile(
            name=opponent_name,
            stance=primary_stance,
            stance_distribution=stance_dist,
            pressure_style=pressure_style,
            movement_speed=movement_speed,
            guard_pull_rate=guard_pull_rate,
            takedown_rate=takedown_rate,
            strike_rate=strike_rate,
            breath_rate_mean=breath_mean,
            fatigue_onset=fatigue_onset,
            defensive_distance=defensive_distance,
            tendencies=tendencies
        )

    def _compute_movement_speed(self) -> float:
        """Compute average movement speed from pose history."""
        if len(self.pose_history) < 2:
            return 0.0

        speeds = []

        for i in range(1, len(self.pose_history)):
            prev = self.pose_history[i - 1]
            curr = self.pose_history[i]

            # Use COM or hip midpoint
            prev_pos = self._get_center_position(prev['keypoints'])
            curr_pos = self._get_center_position(curr['keypoints'])

            if prev_pos is not None and curr_pos is not None:
                dt = curr['timestamp'] - prev['timestamp']
                if dt > 0:
                    dist = np.linalg.norm(curr_pos - prev_pos)
                    speeds.append(dist / dt)

        return np.mean(speeds) if speeds else 0.0

    def _get_center_position(self, keypoints: Dict) -> Optional[np.ndarray]:
        """Get center position from keypoints."""
        # Try hips first
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            left_hip = np.array(keypoints['left_hip'])
            right_hip = np.array(keypoints['right_hip'])
            return (left_hip + right_hip) / 2

        # Fall back to any available keypoint
        if keypoints:
            return np.array(list(keypoints.values())[0])

        return None

    def _classify_pressure_style(self) -> str:
        """Classify pressure style from movement patterns."""
        if not self.pose_history:
            return 'balanced'

        # Simple heuristic: high movement speed = aggressive
        speed = self._compute_movement_speed()

        if speed > 1.0:  # meters/second
            return 'aggressive'
        elif speed < 0.5:
            return 'counter'
        else:
            return 'balanced'

    def _detect_fatigue_onset(self) -> float:
        """Detect when fatigue sets in (breath rate increase)."""
        if len(self.breath_history) < 30:
            return 0.0

        # Find when breath rate increases by 20% from baseline
        baseline = np.mean(self.breath_history[:30])

        for i in range(30, len(self.breath_history)):
            if self.breath_history[i] > baseline * 1.2:
                return self.timestamps[i]

        return 0.0  # No fatigue detected

    def _compute_defensive_distance(self) -> float:
        """Compute preferred defensive distance."""
        # Placeholder: would need opponent-relative positions
        return 1.5  # meters (typical BJJ/MT range)

    def _extract_tendencies(self) -> Dict[str, float]:
        """Extract specific behavioral tendencies."""
        tendencies = {}

        # Example tendencies
        if self.events['guard_pull'] > self.events['takedown']:
            tendencies['prefers_guard_pull'] = 0.8

        if self.events['strike'] > 0:
            tendencies['uses_strikes'] = min(1.0, self.events['strike'] / 10)

        return tendencies

    def reset(self):
        """Reset extractor state."""
        self.pose_history.clear()
        self.breath_history.clear()
        self.timestamps.clear()
        self.events.clear()


def test_feature_extractor():
    """Test opponent feature extraction."""
    print("Testing Opponent Feature Extractor...")

    extractor = OpponentFeatureExtractor()

    # Simulate 2 minutes of footage
    fps = 30
    duration = 120  # seconds

    for i in range(fps * duration):
        t = i / fps

        # Simulate pose with movement
        x = np.sin(t * 0.5) * 2.0  # 2m amplitude
        y = 0.0
        z = 1.5  # standing height

        keypoints = {
            'left_hip': np.array([x - 0.2, y, z]),
            'right_hip': np.array([x + 0.2, y, z])
        }

        # Simulate breath (starts at 18 BPM, increases to 24 after 60s)
        breath_rate = 18 + (6 * (t / 60)) if t < 60 else 24

        extractor.add_frame_data(
            timestamp=t,
            pose_keypoints=keypoints,
            breath_rate=breath_rate,
            stance_type='orthodox'
        )

        # Simulate events
        if i % 300 == 0:  # Every 10 seconds
            extractor.add_event('strike')

        if i % 900 == 0:  # Every 30 seconds
            extractor.add_event('guard_pull')

    # Extract profile
    profile = extractor.extract_profile('Opponent_A')

    print(f"\nOpponent Profile: {profile.name}")
    print(f"  Stance: {profile.stance}")
    print(f"  Pressure style: {profile.pressure_style}")
    print(f"  Movement speed: {profile.movement_speed:.2f} m/s")
    print(f"  Guard pull rate: {profile.guard_pull_rate:.2f} /min")
    print(f"  Strike rate: {profile.strike_rate:.2f} /min")
    print(f"  Breath rate: {profile.breath_rate_mean:.1f} BPM")
    print(f"  Fatigue onset: {profile.fatigue_onset:.1f}s")

    assert profile.stance == 'orthodox'
    assert profile.strike_rate > 0
    assert profile.breath_rate_mean > 18

    print("✓ PASS")


if __name__ == "__main__":
    test_feature_extractor()
