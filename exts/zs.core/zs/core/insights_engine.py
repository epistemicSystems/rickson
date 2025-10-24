"""
Insights Engine

Analyzes training sessions and provides actionable insights
for BJJ/Muay Thai athletes.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class TrainingInsight:
    """Single training insight."""

    category: str  # 'breath', 'balance', 'technique', 'conditioning'
    severity: str  # 'info', 'warning', 'alert'
    title: str
    description: str
    recommendation: str
    confidence: float  # 0-1


class InsightsEngine:
    """
    Generates training insights from EVM, pose, and balance data.
    """

    def __init__(self):
        """Initialize insights engine."""
        self.insights_history: List[TrainingInsight] = []

    def analyze_breath_pattern(
        self,
        breath_rate_history: List[float],
        time_history: List[float],
        fps: float = 30.0
    ) -> List[TrainingInsight]:
        """
        Analyze breathing patterns for insights.

        Args:
            breath_rate_history: Breath rates (BPM) over time
            time_history: Timestamps for each measurement
            fps: Video framerate

        Returns:
            List of insights
        """
        insights = []

        if len(breath_rate_history) < 30:
            return insights

        # Convert to numpy
        rates = np.array(breath_rate_history)
        rates = rates[rates > 0]  # Filter out invalid

        if len(rates) == 0:
            return insights

        # Compute statistics
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        min_rate = np.min(rates)
        max_rate = np.max(rates)

        # Insight 1: Average breathing rate
        if mean_rate < 12:
            insights.append(TrainingInsight(
                category='breath',
                severity='info',
                title='Low Breathing Rate',
                description=f'Average breathing rate of {mean_rate:.1f} BPM indicates good breath control.',
                recommendation='Maintain this controlled breathing during high-intensity drills.',
                confidence=0.9
            ))
        elif mean_rate > 24:
            insights.append(TrainingInsight(
                category='breath',
                severity='warning',
                title='Elevated Breathing Rate',
                description=f'Average breathing rate of {mean_rate:.1f} BPM suggests high exertion or stress.',
                recommendation='Focus on slower, deeper breaths between rounds. Practice box breathing: 4s in, 4s hold, 4s out, 4s hold.',
                confidence=0.85
            ))

        # Insight 2: Breath variability
        if std_rate > 6:
            insights.append(TrainingInsight(
                category='breath',
                severity='warning',
                title='Inconsistent Breathing',
                description=f'High breath rate variability (std={std_rate:.1f}) indicates irregular rhythm.',
                recommendation='Work on maintaining steady breath rhythm even during position transitions.',
                confidence=0.8
            ))

        # Insight 3: Breath holds (very low rates)
        breath_holds = np.sum(rates < 8)
        if breath_holds > len(rates) * 0.1:
            insights.append(TrainingInsight(
                category='breath',
                severity='alert',
                title='Frequent Breath Holding',
                description=f'Detected breath holding in {breath_holds}/{len(rates)} measurements.',
                recommendation='Avoid holding breath during exertion. Exhale during technique execution (e.g., strikes, submissions).',
                confidence=0.75
            ))

        return insights

    def analyze_balance_stability(
        self,
        balance_score_history: List[float],
        stance_type_history: List[str]
    ) -> List[TrainingInsight]:
        """
        Analyze balance stability for insights.

        Args:
            balance_score_history: Balance scores (0-100) over time
            stance_type_history: Stance types over time

        Returns:
            List of insights
        """
        insights = []

        if len(balance_score_history) < 10:
            return insights

        scores = np.array(balance_score_history)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)

        # Insight 1: Average balance
        if mean_score > 75:
            insights.append(TrainingInsight(
                category='balance',
                severity='info',
                title='Strong Balance',
                description=f'Average balance score of {mean_score:.0f}/100 shows solid stability.',
                recommendation='Challenge yourself with dynamic footwork drills and single-leg techniques.',
                confidence=0.9
            ))
        elif mean_score < 50:
            insights.append(TrainingInsight(
                category='balance',
                severity='warning',
                title='Balance Needs Work',
                description=f'Average balance score of {mean_score:.0f}/100 suggests instability.',
                recommendation='Practice static stances (horse, cat, front). Focus on keeping COM over base of support.',
                confidence=0.85
            ))

        # Insight 2: Balance variability
        if std_score > 20:
            insights.append(TrainingInsight(
                category='balance',
                severity='warning',
                title='Unstable Balance',
                description=f'High balance variability (std={std_score:.1f}) indicates inconsistent stability.',
                recommendation='Work on smooth weight transitions. Practice slow-motion shadow boxing.',
                confidence=0.8
            ))

        # Insight 3: Low balance moments
        low_balance_count = np.sum(scores < 30)
        if low_balance_count > len(scores) * 0.05:
            insights.append(TrainingInsight(
                category='balance',
                severity='alert',
                title='Balance Edge Moments',
                description=f'Detected {low_balance_count} moments near loss of balance.',
                recommendation='Widen stance slightly. Keep knees bent and weight on balls of feet.',
                confidence=0.75
            ))

        # Insight 4: Stance preference
        if len(stance_type_history) > 0:
            from collections import Counter
            stance_counts = Counter(stance_type_history)
            most_common = stance_counts.most_common(1)[0]

            if most_common[0] == 'parallel' and most_common[1] > len(stance_type_history) * 0.7:
                insights.append(TrainingInsight(
                    category='balance',
                    severity='info',
                    title='Parallel Stance Dominant',
                    description='You favor parallel stance (70%+ of time).',
                    recommendation='Practice staggered/fighting stance more for better BJJ/Muay Thai applicability.',
                    confidence=0.7
                ))

        return insights

    def analyze_combined(
        self,
        breath_rate_history: List[float],
        balance_score_history: List[float]
    ) -> List[TrainingInsight]:
        """
        Analyze correlations between breath and balance.

        Args:
            breath_rate_history: Breath rates over time
            balance_score_history: Balance scores over time

        Returns:
            List of insights
        """
        insights = []

        if len(breath_rate_history) < 30 or len(balance_score_history) < 30:
            return insights

        # Align lengths
        min_len = min(len(breath_rate_history), len(balance_score_history))
        breath = np.array(breath_rate_history[-min_len:])
        balance = np.array(balance_score_history[-min_len:])

        # Filter valid data
        valid_mask = (breath > 0) & (balance > 0)
        if np.sum(valid_mask) < 10:
            return insights

        breath = breath[valid_mask]
        balance = balance[valid_mask]

        # Compute correlation
        if len(breath) > 10:
            corr = np.corrcoef(breath, balance)[0, 1]

            if corr < -0.3:
                insights.append(TrainingInsight(
                    category='conditioning',
                    severity='warning',
                    title='Breath Affects Balance',
                    description=f'Negative correlation ({corr:.2f}) between breathing rate and balance.',
                    recommendation='High breathing rate degrades balance. Practice breathing drills to improve endurance.',
                    confidence=0.7
                ))

        return insights

    def generate_session_summary(
        self,
        breath_rate_history: List[float],
        balance_score_history: List[float],
        stance_type_history: List[str],
        duration_seconds: float
    ) -> Dict:
        """
        Generate comprehensive session summary.

        Args:
            breath_rate_history: Breath rates
            balance_score_history: Balance scores
            stance_type_history: Stance types
            duration_seconds: Session duration

        Returns:
            Summary dictionary
        """
        summary = {
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_seconds / 60.0,
            'breath': {},
            'balance': {},
            'insights': []
        }

        # Breath statistics
        if breath_rate_history:
            valid_breath = [r for r in breath_rate_history if r > 0]
            if valid_breath:
                summary['breath'] = {
                    'mean_bpm': np.mean(valid_breath),
                    'std_bpm': np.std(valid_breath),
                    'min_bpm': np.min(valid_breath),
                    'max_bpm': np.max(valid_breath),
                    'measurements': len(valid_breath)
                }

        # Balance statistics
        if balance_score_history:
            valid_balance = [s for s in balance_score_history if s > 0]
            if valid_balance:
                summary['balance'] = {
                    'mean_score': np.mean(valid_balance),
                    'std_score': np.std(valid_balance),
                    'min_score': np.min(valid_balance),
                    'max_score': np.max(valid_balance),
                    'measurements': len(valid_balance)
                }

        # Generate insights
        insights = []
        insights.extend(self.analyze_breath_pattern(
            breath_rate_history, [], fps=30.0
        ))
        insights.extend(self.analyze_balance_stability(
            balance_score_history, stance_type_history
        ))
        insights.extend(self.analyze_combined(
            breath_rate_history, balance_score_history
        ))

        summary['insights'] = [
            {
                'category': i.category,
                'severity': i.severity,
                'title': i.title,
                'description': i.description,
                'recommendation': i.recommendation,
                'confidence': i.confidence
            }
            for i in insights
        ]

        return summary

    def format_insights_text(self, insights: List[TrainingInsight]) -> str:
        """
        Format insights as human-readable text.

        Args:
            insights: List of insights

        Returns:
            Formatted string
        """
        if not insights:
            return "No insights available yet. Keep training!"

        lines = ["Training Insights:\n"]

        for i, insight in enumerate(insights, 1):
            emoji = {
                'info': '\u2139',  # ℹ️
                'warning': '\u26a0',  # ⚠️
                'alert': '\u26a1'  # ⚡
            }.get(insight.severity, '')

            lines.append(f"{i}. {emoji} {insight.title}")
            lines.append(f"   {insight.description}")
            lines.append(f"   → {insight.recommendation}\n")

        return '\n'.join(lines)


def test_insights_engine():
    """Test insights engine with synthetic data."""
    print("Testing Insights Engine...")

    engine = InsightsEngine()

    # Synthetic data
    breath_rates = [15.0 + np.random.randn() * 3 for _ in range(100)]
    balance_scores = [70.0 + np.random.randn() * 10 for _ in range(100)]
    stance_types = ['parallel'] * 70 + ['staggered'] * 30

    # Analyze
    breath_insights = engine.analyze_breath_pattern(breath_rates, [], 30.0)
    balance_insights = engine.analyze_balance_stability(balance_scores, stance_types)
    combined_insights = engine.analyze_combined(breath_rates, balance_scores)

    all_insights = breath_insights + balance_insights + combined_insights

    print(f"  Generated {len(all_insights)} insights")
    for insight in all_insights:
        print(f"    - [{insight.severity}] {insight.title}")

    # Session summary
    summary = engine.generate_session_summary(
        breath_rates, balance_scores, stance_types, 60.0
    )

    print(f"  Session summary: {summary['duration_minutes']:.1f} min")
    print(f"  Mean breath: {summary['breath'].get('mean_bpm', 0):.1f} BPM")
    print(f"  Mean balance: {summary['balance'].get('mean_score', 0):.0f}/100")

    print("  Result: PASS")
    return True


if __name__ == "__main__":
    test_insights_engine()
