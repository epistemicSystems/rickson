"""
Opponent Analysis (Milestone 6)

Analyze opponent footage to extract patterns, tendencies, and suggest
training games/drills to prepare for specific opponents.

This is a prototype implementation focusing on:
1. Movement pattern recognition (stance, transitions)
2. Attack frequency analysis (strikes, takedowns, submissions)
3. Training game generation based on observed patterns
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import carb


@dataclass
class OpponentPattern:
    """Detected opponent pattern."""

    pattern_type: str  # 'stance', 'attack', 'transition', 'defense'
    description: str
    frequency: float  # 0-1, how often this pattern occurs
    confidence: float  # 0-1, confidence in detection
    context: str  # When/where this pattern appears
    counter_strategy: str  # Suggested counter


@dataclass
class TrainingGame:
    """Suggested training game/drill."""

    name: str
    objective: str
    description: str
    opponent_pattern_addressed: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    duration_minutes: int
    success_criteria: str


class FeatureExtractor:
    """
    Extract features from opponent video footage.

    Features include:
    - Stance preferences (orthodox, southpaw, switching)
    - Attack patterns (frequency, combinations)
    - Movement style (aggressive, defensive, counter)
    - Position preferences (guard, top, etc.)
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.features = {
            'stance_distribution': Counter(),
            'attack_frequency': {},
            'movement_patterns': [],
            'position_time': Counter(),
        }

    def process_frame_data(
        self,
        stance: str,
        attack_detected: Optional[str],
        position: Optional[str]
    ):
        """
        Process single frame of opponent data.

        Args:
            stance: Current stance ('orthodox', 'southpaw', 'parallel', etc.)
            attack_detected: Type of attack if detected
            position: BJJ position if applicable
        """
        # Track stance
        self.features['stance_distribution'][stance] += 1

        # Track attacks
        if attack_detected:
            if attack_detected not in self.features['attack_frequency']:
                self.features['attack_frequency'][attack_detected] = 0
            self.features['attack_frequency'][attack_detected] += 1

        # Track position
        if position:
            self.features['position_time'][position] += 1

    def analyze_stance_preference(self) -> Tuple[str, float]:
        """
        Analyze opponent's stance preference.

        Returns:
            (preferred_stance, preference_strength)
        """
        if not self.features['stance_distribution']:
            return ('unknown', 0.0)

        total = sum(self.features['stance_distribution'].values())
        most_common = self.features['stance_distribution'].most_common(1)[0]
        stance = most_common[0]
        count = most_common[1]

        preference_strength = count / total

        return (stance, preference_strength)

    def analyze_attack_patterns(self) -> List[Tuple[str, float]]:
        """
        Analyze attack frequency.

        Returns:
            List of (attack_type, normalized_frequency)
        """
        if not self.features['attack_frequency']:
            return []

        total_attacks = sum(self.features['attack_frequency'].values())

        patterns = []
        for attack, count in self.features['attack_frequency'].items():
            frequency = count / total_attacks
            patterns.append((attack, frequency))

        # Sort by frequency
        patterns.sort(key=lambda x: x[1], reverse=True)

        return patterns

    def get_summary(self) -> Dict[str, Any]:
        """Get feature extraction summary."""
        stance, stance_strength = self.analyze_stance_preference()
        attack_patterns = self.analyze_attack_patterns()

        return {
            'stance': {
                'preferred': stance,
                'strength': stance_strength,
                'distribution': dict(self.features['stance_distribution'])
            },
            'attacks': {
                'patterns': attack_patterns,
                'total_observed': sum(self.features['attack_frequency'].values())
            },
            'positions': dict(self.features['position_time'])
        }


class OpponentAnalyzer:
    """
    Analyze opponent based on extracted features and generate insights.
    """

    def __init__(self):
        """Initialize opponent analyzer."""
        self.patterns: List[OpponentPattern] = []

    def analyze_features(self, features: Dict[str, Any]) -> List[OpponentPattern]:
        """
        Analyze extracted features to identify patterns.

        Args:
            features: Feature dictionary from FeatureExtractor

        Returns:
            List of detected patterns
        """
        patterns = []

        # Analyze stance
        stance_info = features.get('stance', {})
        preferred_stance = stance_info.get('preferred', 'unknown')
        stance_strength = stance_info.get('strength', 0.0)

        if stance_strength > 0.7:
            # Strong stance preference
            patterns.append(OpponentPattern(
                pattern_type='stance',
                description=f"Strongly prefers {preferred_stance} stance ({stance_strength*100:.0f}% of time)",
                frequency=stance_strength,
                confidence=0.9,
                context="Throughout match",
                counter_strategy=f"Train against {preferred_stance} stance exclusively. "
                                f"Exploit angles specific to this stance."
            ))
        elif stance_strength < 0.4:
            # Stance switcher
            patterns.append(OpponentPattern(
                pattern_type='stance',
                description="Frequently switches stance (no strong preference)",
                frequency=1.0 - stance_strength,
                confidence=0.8,
                context="Throughout match",
                counter_strategy="Practice adapting to stance switches mid-exchange. "
                                "Don't commit to stance-specific game plans."
            ))

        # Analyze attacks
        attack_info = features.get('attacks', {})
        attack_patterns = attack_info.get('patterns', [])

        if attack_patterns:
            # Identify favorite attacks
            top_attack = attack_patterns[0]
            attack_type, frequency = top_attack

            patterns.append(OpponentPattern(
                pattern_type='attack',
                description=f"Favors {attack_type} ({frequency*100:.0f}% of attacks)",
                frequency=frequency,
                confidence=0.85,
                context="Primary offensive tool",
                counter_strategy=f"Drill defense against {attack_type}. "
                                f"Create training scenarios focused on this specific attack."
            ))

            # Check for attack diversity
            if len(attack_patterns) > 3 and attack_patterns[0][1] < 0.4:
                patterns.append(OpponentPattern(
                    pattern_type='attack',
                    description="Diverse attack selection (no dominant pattern)",
                    frequency=1.0,
                    confidence=0.7,
                    context="Well-rounded offensive game",
                    counter_strategy="Prepare for varied attacks. Focus on fundamental defensive position."
                ))

        self.patterns = patterns
        return patterns


class TrainingGameGenerator:
    """
    Generate training games/drills based on opponent analysis.
    """

    def __init__(self):
        """Initialize training game generator."""
        self.games: List[TrainingGame] = []

    def generate_games(self, patterns: List[OpponentPattern]) -> List[TrainingGame]:
        """
        Generate training games from opponent patterns.

        Args:
            patterns: List of detected opponent patterns

        Returns:
            List of suggested training games
        """
        games = []

        for pattern in patterns:
            if pattern.pattern_type == 'stance':
                # Generate stance-specific games
                if 'switches stance' in pattern.description.lower():
                    games.append(TrainingGame(
                        name="Stance Switch Adaptation Drill",
                        objective="Develop ability to adapt to mid-exchange stance switches",
                        description="Partner switches stance every 10-15 seconds. "
                                  "Maintain pressure and adapt angles accordingly. "
                                  "Practice entering on both orthodox and southpaw.",
                        opponent_pattern_addressed=pattern.description,
                        difficulty='intermediate',
                        duration_minutes=10,
                        success_criteria="Successfully land 3+ combinations despite stance switches"
                    ))
                elif 'prefers' in pattern.description.lower():
                    stance_name = pattern.description.split('prefers ')[1].split(' stance')[0]
                    games.append(TrainingGame(
                        name=f"{stance_name.title()} Stance Exploitation Drill",
                        objective=f"Exploit angles and openings specific to {stance_name} stance",
                        description=f"Partner holds {stance_name} stance exclusively. "
                                  f"Practice stance-specific combinations and angles. "
                                  f"Focus on attacks that target {stance_name} weaknesses.",
                        opponent_pattern_addressed=pattern.description,
                        difficulty='beginner',
                        duration_minutes=15,
                        success_criteria=f"Execute 5+ clean {stance_name}-specific techniques"
                    ))

            elif pattern.pattern_type == 'attack':
                # Generate attack-defense games
                if 'favors' in pattern.description.lower():
                    attack_type = pattern.description.split('Favors ')[1].split(' (')[0]
                    games.append(TrainingGame(
                        name=f"{attack_type} Defense Drill",
                        objective=f"Master defense against {attack_type}",
                        description=f"Partner throws {attack_type} at 50-70% speed/power. "
                                  f"Practice defense, counter, and recovery. "
                                  f"Gradually increase speed as proficiency improves.",
                        opponent_pattern_addressed=pattern.description,
                        difficulty='intermediate',
                        duration_minutes=12,
                        success_criteria=f"Successfully defend 8/10 {attack_type} attempts"
                    ))

                    # Counter game
                    games.append(TrainingGame(
                        name=f"{attack_type} Counter Game",
                        objective=f"Counter {attack_type} with immediate offense",
                        description=f"Partner attacks with {attack_type}. "
                                  f"Defend and immediately counter with your best technique. "
                                  f"Focus on timing and reaction speed.",
                        opponent_pattern_addressed=pattern.description,
                        difficulty='advanced',
                        duration_minutes=8,
                        success_criteria=f"Land clean counter on 6/10 {attack_type} attempts"
                    ))

        self.games = games
        return games

    def format_game_plan(self, games: List[TrainingGame]) -> str:
        """
        Format training games as readable training plan.

        Args:
            games: List of training games

        Returns:
            Formatted string
        """
        if not games:
            return "No training games generated."

        lines = ["Opponent-Specific Training Plan", "="*70, ""]

        for i, game in enumerate(games, 1):
            lines.append(f"{i}. {game.name} [{game.difficulty.upper()}] ({game.duration_minutes} min)")
            lines.append(f"   Objective: {game.objective}")
            lines.append(f"   Addresses: {game.opponent_pattern_addressed}")
            lines.append(f"   ")
            lines.append(f"   {game.description}")
            lines.append(f"   ")
            lines.append(f"   Success: {game.success_criteria}")
            lines.append("")

        total_time = sum(g.duration_minutes for g in games)
        lines.append(f"Total Training Time: {total_time} minutes")

        return '\n'.join(lines)


def analyze_opponent_video(
    video_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete opponent analysis pipeline.

    Args:
        video_path: Path to opponent video
        output_path: Optional path to save training plan

    Returns:
        Analysis results with training games
    """
    carb.log_info(f"[OpponentAnalysis] Analyzing opponent video: {video_path}")

    # Feature extraction
    extractor = FeatureExtractor()

    # In real implementation, would process video frames
    # For this prototype, simulate with synthetic data
    for _ in range(300):  # 10 seconds @ 30fps
        # Simulate detected data
        stance = np.random.choice(
            ['orthodox', 'southpaw', 'parallel'],
            p=[0.7, 0.2, 0.1]  # Strong orthodox preference
        )

        attack = None
        if np.random.rand() < 0.1:  # 10% attack frequency
            attack = np.random.choice(
                ['jab', 'cross', 'hook', 'low_kick', 'takedown'],
                p=[0.4, 0.3, 0.15, 0.1, 0.05]  # Jab-heavy
            )

        position = np.random.choice(['standing', 'guard', 'top'], p=[0.7, 0.2, 0.1])

        extractor.process_frame_data(stance, attack, position)

    # Get features
    features = extractor.get_summary()

    # Analyze patterns
    analyzer = OpponentAnalyzer()
    patterns = analyzer.analyze_features(features)

    # Generate training games
    generator = TrainingGameGenerator()
    games = generator.generate_games(patterns)

    # Results
    results = {
        'features': features,
        'patterns': [
            {
                'type': p.pattern_type,
                'description': p.description,
                'frequency': p.frequency,
                'confidence': p.confidence,
                'counter_strategy': p.counter_strategy
            }
            for p in patterns
        ],
        'training_games': [
            {
                'name': g.name,
                'objective': g.objective,
                'description': g.description,
                'difficulty': g.difficulty,
                'duration_minutes': g.duration_minutes
            }
            for g in games
        ]
    }

    # Print training plan
    training_plan = generator.format_game_plan(games)
    print("\n" + training_plan)

    # Save if requested
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        carb.log_info(f"[OpponentAnalysis] Saved results to {output_path}")

    return results


def test_opponent_analysis():
    """Test opponent analysis with synthetic data."""
    print("\nTesting Opponent Analysis...")
    print("="*70)

    # Run analysis
    results = analyze_opponent_video('synthetic_opponent.mp4')

    print("\n[Feature Summary]")
    print(f"  Stance: {results['features']['stance']['preferred']} "
          f"({results['features']['stance']['strength']*100:.0f}% preference)")
    print(f"  Attacks: {results['features']['attacks']['total_observed']} total observed")

    print(f"\n[Patterns Detected]: {len(results['patterns'])}")
    for pattern in results['patterns']:
        print(f"  - [{pattern['type']}] {pattern['description']}")

    print(f"\n[Training Games Generated]: {len(results['training_games'])}")
    for game in results['training_games']:
        print(f"  - {game['name']} ({game['difficulty']}, {game['duration_minutes']}min)")

    print("\n" + "="*70)
    print("Opponent Analysis Test: PASS")
    return True


if __name__ == "__main__":
    test_opponent_analysis()
