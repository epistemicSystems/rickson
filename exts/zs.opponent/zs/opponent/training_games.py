"""
Training Game Recommender

Suggests specific training games/drills based on opponent profile analysis.
Helps athletes prepare strategically for upcoming matches.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .feature_extractor import OpponentProfile


@dataclass
class TrainingGame:
    """
    Training drill/game recommendation.

    Attributes:
        name: Game name
        description: What to practice
        rationale: Why this game is recommended
        priority: Priority score [0, 1]
        duration_minutes: Recommended duration
        intensity: 'low', 'medium', 'high'
        category: 'offense', 'defense', 'conditioning', 'strategy'
    """
    name: str
    description: str
    rationale: str
    priority: float
    duration_minutes: int
    intensity: str = 'medium'
    category: str = 'strategy'


class TrainingGameRecommender:
    """
    Recommend training games based on opponent profile.

    Uses rule-based system to match opponent tendencies with
    appropriate counter-strategies and drills.
    """

    def __init__(self):
        """Initialize recommender."""
        # Game templates
        self.game_templates = self._build_game_templates()

    def _build_game_templates(self) -> Dict[str, Dict]:
        """Build library of training game templates."""
        return {
            'guard_pull_defense': {
                'name': 'Guard Pull Defense Drill',
                'description': 'Partner attempts guard pulls, you defend and maintain top position',
                'category': 'defense',
                'intensity': 'medium',
                'duration_minutes': 15,
                'triggers': {
                    'guard_pull_rate': (0.5, float('inf'))  # >0.5 per minute
                }
            },
            'pressure_passing': {
                'name': 'Pressure Passing Game',
                'description': 'Start in opponent\'s closed guard, focus on breaking and passing with pressure',
                'category': 'offense',
                'intensity': 'high',
                'duration_minutes': 10,
                'triggers': {
                    'pressure_style': ['counter', 'balanced']
                }
            },
            'distance_management': {
                'name': 'Distance Control Sparring',
                'description': 'Practice maintaining optimal range, prevent opponent from entering preferred distance',
                'category': 'strategy',
                'intensity': 'medium',
                'duration_minutes': 20,
                'triggers': {
                    'pressure_style': ['aggressive']
                }
            },
            'takedown_defense': {
                'name': 'Takedown Defense Rounds',
                'description': 'Defend takedown attempts, emphasize sprawl and footwork',
                'category': 'defense',
                'intensity': 'high',
                'duration_minutes': 15,
                'triggers': {
                    'takedown_rate': (1.0, float('inf'))
                }
            },
            'orthodox_vs_southpaw': {
                'name': 'Orthodox vs Southpaw Drilling',
                'description': 'Practice stance-specific strategies and angles',
                'category': 'strategy',
                'intensity': 'medium',
                'duration_minutes': 15,
                'triggers': {
                    'stance': ['southpaw']
                }
            },
            'conditioning_pace': {
                'name': 'High-Pace Conditioning',
                'description': 'Maintain high intensity for opponent\'s fatigue onset time',
                'category': 'conditioning',
                'intensity': 'high',
                'duration_minutes': 20,
                'triggers': {
                    'fatigue_onset': (60, 180)  # If they fatigue between 1-3 minutes
                }
            },
            'breath_control': {
                'name': 'Breath Control Under Pressure',
                'description': 'Maintain low breath rate while partner applies pressure',
                'category': 'conditioning',
                'intensity': 'medium',
                'duration_minutes': 10,
                'triggers': {
                    'breath_rate_mean': (20, float('inf'))  # If opponent breath rate is high
                }
            },
            'strike_defense': {
                'name': 'Strike Defense Drill',
                'description': 'Defend strikes while entering for clinch/takedown',
                'category': 'defense',
                'intensity': 'medium',
                'duration_minutes': 10,
                'triggers': {
                    'strike_rate': (3.0, float('inf'))  # >3 strikes per minute
                }
            },
            'counter_timing': {
                'name': 'Counter Timing Game',
                'description': 'Bait and counter opponent attacks',
                'category': 'offense',
                'intensity': 'medium',
                'duration_minutes': 15,
                'triggers': {
                    'pressure_style': ['counter']
                }
            }
        }

    def recommend(
        self,
        opponent_profile: OpponentProfile,
        max_recommendations: int = 5
    ) -> List[TrainingGame]:
        """
        Recommend training games for opponent.

        Args:
            opponent_profile: Opponent analysis
            max_recommendations: Maximum number of games to return

        Returns:
            List of TrainingGame sorted by priority
        """
        recommendations = []

        for game_id, template in self.game_templates.items():
            # Check if game is triggered by profile
            priority = self._compute_priority(template, opponent_profile)

            if priority > 0.1:  # Minimum threshold
                # Generate rationale
                rationale = self._generate_rationale(template, opponent_profile)

                game = TrainingGame(
                    name=template['name'],
                    description=template['description'],
                    rationale=rationale,
                    priority=priority,
                    duration_minutes=template['duration_minutes'],
                    intensity=template['intensity'],
                    category=template['category']
                )

                recommendations.append(game)

        # Sort by priority
        recommendations.sort(key=lambda g: g.priority, reverse=True)

        return recommendations[:max_recommendations]

    def _compute_priority(
        self,
        template: Dict,
        profile: OpponentProfile
    ) -> float:
        """Compute priority score for a game based on profile match."""
        priority = 0.0
        num_triggers = len(template['triggers'])
        matches = 0

        for attr, condition in template['triggers'].items():
            value = getattr(profile, attr, None)

            if value is None:
                continue

            if isinstance(condition, tuple):
                # Numeric range
                min_val, max_val = condition
                if min_val <= value <= max_val:
                    matches += 1
            elif isinstance(condition, list):
                # Categorical match
                if value in condition:
                    matches += 1

        if num_triggers > 0:
            priority = matches / num_triggers

        return priority

    def _generate_rationale(
        self,
        template: Dict,
        profile: OpponentProfile
    ) -> str:
        """Generate human-readable rationale for recommendation."""
        name = profile.name

        # Extract key triggers
        rationale_parts = [f"Based on {name}'s profile:"]

        if 'guard_pull_rate' in template['triggers']:
            if profile.guard_pull_rate > 0.5:
                rationale_parts.append(
                    f"- Frequent guard pulls ({profile.guard_pull_rate:.1f}/min)"
                )

        if 'takedown_rate' in template['triggers']:
            if profile.takedown_rate > 1.0:
                rationale_parts.append(
                    f"- High takedown rate ({profile.takedown_rate:.1f}/min)"
                )

        if 'pressure_style' in template['triggers']:
            rationale_parts.append(f"- {profile.pressure_style.capitalize()} pressure style")

        if 'stance' in template['triggers']:
            rationale_parts.append(f"- {profile.stance.capitalize()} stance")

        if 'strike_rate' in template['triggers']:
            if profile.strike_rate > 3.0:
                rationale_parts.append(
                    f"- Strikes frequently ({profile.strike_rate:.1f}/min)"
                )

        if 'fatigue_onset' in template['triggers']:
            if profile.fatigue_onset > 0:
                fatigue_min = profile.fatigue_onset / 60
                rationale_parts.append(
                    f"- Fatigue onset at {fatigue_min:.1f} minutes"
                )

        return '\n'.join(rationale_parts)

    def generate_training_plan(
        self,
        opponent_profile: OpponentProfile,
        available_time: int = 60  # minutes
    ) -> Dict:
        """
        Generate complete training plan for session.

        Args:
            opponent_profile: Opponent profile
            available_time: Available training time (minutes)

        Returns:
            Training plan with schedule
        """
        recommendations = self.recommend(opponent_profile, max_recommendations=10)

        # Allocate time by priority
        plan = {
            'opponent': opponent_profile.name,
            'total_time': available_time,
            'games': [],
            'schedule': []
        }

        remaining_time = available_time
        cumulative_time = 0

        for game in recommendations:
            if remaining_time <= 0:
                break

            # Allocate time (cap at recommended duration)
            allocated_time = min(game.duration_minutes, remaining_time)

            plan['games'].append({
                'name': game.name,
                'duration': allocated_time,
                'intensity': game.intensity,
                'category': game.category,
                'rationale': game.rationale
            })

            plan['schedule'].append({
                'start': cumulative_time,
                'end': cumulative_time + allocated_time,
                'game': game.name
            })

            remaining_time -= allocated_time
            cumulative_time += allocated_time

        return plan


def test_recommender():
    """Test training game recommender."""
    print("Testing Training Game Recommender...")

    # Create opponent profile
    profile = OpponentProfile(
        name='Opponent_A',
        stance='southpaw',
        pressure_style='aggressive',
        guard_pull_rate=1.2,
        takedown_rate=0.5,
        strike_rate=4.0,
        breath_rate_mean=22.0,
        fatigue_onset=90.0
    )

    # Get recommendations
    recommender = TrainingGameRecommender()
    games = recommender.recommend(profile, max_recommendations=5)

    print(f"\nTop Training Games for {profile.name}:")
    for i, game in enumerate(games, 1):
        print(f"\n{i}. {game.name} (Priority: {game.priority:.2f})")
        print(f"   Duration: {game.duration_minutes} min, Intensity: {game.intensity}")
        print(f"   Category: {game.category}")
        print(f"   {game.rationale}")

    assert len(games) > 0, "No games recommended"
    assert games[0].priority >= games[-1].priority, "Games not sorted by priority"

    # Generate training plan
    plan = recommender.generate_training_plan(profile, available_time=60)

    print(f"\n\nTraining Plan for {profile.name} ({plan['total_time']} min):")
    for item in plan['schedule']:
        print(f"  {item['start']:02d}-{item['end']:02d} min: {item['game']}")

    print("\n✓ PASS")


if __name__ == "__main__":
    test_recommender()
