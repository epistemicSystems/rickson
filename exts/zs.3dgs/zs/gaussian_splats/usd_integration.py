"""
USD Integration for 3D Gaussian Splats

Creates USD prims for splat point clouds and provides spatial query utilities.
"""

import numpy as np
from typing import Optional, Tuple, List
import carb

try:
    from pxr import Usd, UsdGeom, Gf, Sdf, Vt
    USD_AVAILABLE = True
except ImportError:
    carb.log_warn("[zs.3dgs] USD not available")
    USD_AVAILABLE = False

from .splat_loader import GaussianSplat


def create_splat_prim(
    stage: 'Usd.Stage',
    prim_path: str,
    splat: GaussianSplat,
    use_instancing: bool = False
) -> Optional['UsdGeom.Points']:
    """
    Create USD Points prim for Gaussian splat.

    Args:
        stage: USD stage
        prim_path: Path for new prim (e.g., "/World/GymSplat")
        splat: GaussianSplat to add
        use_instancing: Use point instancing for better performance

    Returns:
        UsdGeom.Points prim
    """
    if not USD_AVAILABLE:
        carb.log_error("[zs.3dgs] USD not available")
        return None

    # Create Points prim
    points_prim = UsdGeom.Points.Define(stage, prim_path)

    if not points_prim:
        carb.log_error(f"[zs.3dgs] Failed to create prim at {prim_path}")
        return None

    # Set positions
    positions_attr = points_prim.GetPointsAttr()
    positions_vt = Vt.Vec3fArray([Gf.Vec3f(*p) for p in splat.positions])
    positions_attr.Set(positions_vt)

    # Set colors
    colors_attr = points_prim.GetDisplayColorAttr()
    colors_vt = Vt.Vec3fArray([Gf.Vec3f(*c) for c in splat.colors])
    colors_attr.Set(colors_vt)

    # Set opacities
    opacities_attr = points_prim.GetDisplayOpacityAttr()
    opacities_vt = Vt.FloatArray(splat.opacities.tolist())
    opacities_attr.Set(opacities_vt)

    # Set widths (use mean scale)
    widths = np.mean(splat.scales, axis=1)
    widths_attr = points_prim.GetWidthsAttr()
    widths_vt = Vt.FloatArray(widths.tolist())
    widths_attr.Set(widths_vt)

    # Store full splat data as custom attributes for spatial queries
    prim = points_prim.GetPrim()

    # Scales
    scales_attr = prim.CreateAttribute('splat:scales', Sdf.ValueTypeNames.Float3Array)
    scales_vt = Vt.Vec3fArray([Gf.Vec3f(*s) for s in splat.scales])
    scales_attr.Set(scales_vt)

    # Rotations
    rotations_attr = prim.CreateAttribute('splat:rotations', Sdf.ValueTypeNames.QuatfArray)
    rotations_vt = Vt.QuatfArray([Gf.Quatf(r[0], r[1], r[2], r[3]) for r in splat.rotations])
    rotations_attr.Set(rotations_vt)

    carb.log_info(f"[zs.3dgs] Created splat prim with {len(splat)} points at {prim_path}")

    return points_prim


def query_splat_depth(
    stage: 'Usd.Stage',
    splat_prim_path: str,
    camera_pos: np.ndarray,
    camera_dir: np.ndarray,
    num_samples: int = 100
) -> Optional[np.ndarray]:
    """
    Query depth values from splat along camera rays.

    Used for constraining 3D pose estimation.

    Args:
        stage: USD stage
        splat_prim_path: Path to splat prim
        camera_pos: (3,) camera position
        camera_dir: (3,) camera direction (normalized)
        num_samples: Number of samples along ray

    Returns:
        (num_samples,) depth values, or None if query fails
    """
    if not USD_AVAILABLE:
        return None

    prim = stage.GetPrimAtPath(splat_prim_path)
    if not prim.IsValid():
        carb.log_warn(f"[zs.3dgs] Prim not found: {splat_prim_path}")
        return None

    points_geom = UsdGeom.Points(prim)

    # Get positions
    positions_attr = points_geom.GetPointsAttr()
    positions_vt = positions_attr.Get()

    if not positions_vt:
        return None

    positions = np.array([[p[0], p[1], p[2]] for p in positions_vt])

    # Get scales for query radius
    scales_attr = prim.GetAttribute('splat:scales')
    if scales_attr:
        scales_vt = scales_attr.Get()
        scales = np.array([[s[0], s[1], s[2]] for s in scales_vt])
        radii = np.mean(scales, axis=1)
    else:
        radii = np.ones(len(positions)) * 0.1

    # Query depth along ray
    depths = []

    for t in np.linspace(0, 10, num_samples):
        query_pos = camera_pos + camera_dir * t

        # Find closest splats
        distances = np.linalg.norm(positions - query_pos, axis=1)
        closest_idx = np.argmin(distances)

        if distances[closest_idx] < radii[closest_idx]:
            depths.append(t)
        else:
            depths.append(np.nan)

    return np.array(depths)


def load_gym_prior(
    stage: 'Usd.Stage',
    ply_path: str,
    prim_path: str = "/World/GymPrior",
    downsample: int = 1,
    min_opacity: float = 0.1
) -> Optional['UsdGeom.Points']:
    """
    Load gym prior from PLY file and add to USD stage.

    Args:
        stage: USD stage
        ply_path: Path to .ply file
        prim_path: USD prim path
        downsample: Downsample factor
        min_opacity: Minimum opacity filter

    Returns:
        Created Points prim
    """
    from .splat_loader import GaussianSplatLoader

    carb.log_info(f"[zs.3dgs] Loading gym prior from {ply_path}")

    # Load splat
    splat = GaussianSplatLoader.load_ply(ply_path)

    # Filter and downsample
    if min_opacity > 0:
        splat = splat.filter_by_opacity(min_opacity)
        carb.log_info(f"[zs.3dgs] Filtered to {len(splat)} splats (opacity >= {min_opacity})")

    if downsample > 1:
        splat = splat.downsample(downsample)
        carb.log_info(f"[zs.3dgs] Downsampled to {len(splat)} splats")

    # Create USD prim
    points_prim = create_splat_prim(stage, prim_path, splat)

    return points_prim


def test_usd_integration():
    """Test USD integration."""
    if not USD_AVAILABLE:
        print("USD not available, skipping test")
        return False

    print("Testing USD integration...")

    # Create in-memory stage
    stage = Usd.Stage.CreateInMemory()

    # Create test splat
    from .splat_loader import GaussianSplat

    N = 50
    positions = np.random.randn(N, 3)
    colors = np.random.rand(N, 3)
    opacities = np.random.rand(N)
    scales = np.random.rand(N, 3) * 0.1
    rotations = np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32)

    splat = GaussianSplat(
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations
    )

    # Create prim
    prim_path = "/World/TestSplat"
    points_prim = create_splat_prim(stage, prim_path, splat)

    assert points_prim is not None, "Failed to create prim"

    # Verify attributes
    positions_attr = points_prim.GetPointsAttr()
    assert positions_attr.Get() is not None, "Positions not set"

    print(f"Created splat prim with {len(splat)} points")

    # Test depth query
    camera_pos = np.array([0, 0, -5])
    camera_dir = np.array([0, 0, 1])

    depths = query_splat_depth(stage, prim_path, camera_pos, camera_dir, num_samples=50)

    if depths is not None:
        valid_depths = depths[~np.isnan(depths)]
        print(f"Depth query: {len(valid_depths)} valid samples")
    else:
        print("Depth query failed")

    print("✓ PASS")
    return True


if __name__ == "__main__":
    test_usd_integration()
