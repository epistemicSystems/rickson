"""
Opponent Analysis Extension

Extracts features from opponent footage and suggests training games.
"""

from .feature_extractor import OpponentFeatureExtractor, OpponentProfile
from .training_games import TrainingGameRecommender, TrainingGame

__all__ = [
    'OpponentFeatureExtractor',
    'OpponentProfile',
    'TrainingGameRecommender',
    'TrainingGame'
]
