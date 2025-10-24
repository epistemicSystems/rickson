"""
3D Gaussian Splatting Extension

Loads gym/athlete priors as 3DGS point clouds for constrained 3D pose estimation.
Integrates with USD scene graph for rendering and spatial queries.
"""

from .splat_loader import GaussianSplatLoader, GaussianSplat
from .usd_integration import create_splat_prim, query_splat_depth
from .gym_prior import GymPrior

__all__ = [
    'GaussianSplatLoader',
    'GaussianSplat',
    'create_splat_prim',
    'query_splat_depth',
    'GymPrior'
]
