"""
Gym Environment Priors

Manages 3DGS gym/space priors for constrained 3D pose estimation.
Provides spatial queries for depth, collision, and semantic regions.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

from .splat_loader import GaussianSplat, GaussianSplatLoader


class GymPrior:
    """
    Gym environment prior from 3D Gaussian Splatting.

    Provides:
    - Depth constraints for pose estimation
    - Semantic regions (mat area, walls, equipment)
    - Collision detection
    - Floor plane estimation
    """

    def __init__(self, splat: GaussianSplat, metadata: Optional[Dict] = None):
        """
        Initialize gym prior.

        Args:
            splat: 3D Gaussian splat of gym
            metadata: Optional metadata (calibration, semantics, etc.)
        """
        self.splat = splat
        self.metadata = metadata or {}

        # Precompute useful structures
        self._build_spatial_index()
        self._estimate_floor_plane()

    def _build_spatial_index(self):
        """Build spatial index for fast queries."""
        # Simple grid-based index
        bounds_min, bounds_max = self.splat.bounds()
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

        # Create voxel grid (for fast nearest-neighbor)
        self.voxel_size = 0.5  # 50cm voxels
        grid_size = np.ceil((bounds_max - bounds_min) / self.voxel_size).astype(int)

        self.grid_size = grid_size
        self.voxel_map = {}

        # Assign splats to voxels
        for i, pos in enumerate(self.splat.positions):
            voxel_idx = tuple(((pos - bounds_min) / self.voxel_size).astype(int))

            if voxel_idx not in self.voxel_map:
                self.voxel_map[voxel_idx] = []

            self.voxel_map[voxel_idx].append(i)

    def _estimate_floor_plane(self):
        """Estimate floor plane from lowest splats."""
        # Find lowest points
        z_values = self.splat.positions[:, 2]
        floor_threshold = np.percentile(z_values, 5)

        floor_points = self.splat.positions[z_values < floor_threshold]

        if len(floor_points) < 3:
            # Default floor at z=0
            self.floor_plane = np.array([0, 0, 1, 0])  # ax + by + cz + d = 0
            return

        # Fit plane with RANSAC (simplified)
        # For now, just use mean z
        mean_z = np.mean(floor_points[:, 2])
        self.floor_plane = np.array([0, 0, 1, -mean_z])

        # Store floor height
        self.floor_height = mean_z

    def query_depth(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        max_distance: float = 10.0,
        num_samples: int = 100
    ) -> Optional[float]:
        """
        Query depth along ray.

        Args:
            ray_origin: (3,) ray origin
            ray_direction: (3,) ray direction (normalized)
            max_distance: Maximum query distance
            num_samples: Number of samples along ray

        Returns:
            Depth to first intersection, or None
        """
        # Sample along ray
        t_values = np.linspace(0, max_distance, num_samples)

        for t in t_values:
            query_pos = ray_origin + ray_direction * t

            # Find nearest splats
            nearest_indices = self._query_nearest(query_pos, radius=0.5)

            if len(nearest_indices) > 0:
                # Check if any splat intersects
                for idx in nearest_indices:
                    splat_pos = self.splat.positions[idx]
                    splat_radius = np.mean(self.splat.scales[idx])

                    dist = np.linalg.norm(query_pos - splat_pos)

                    if dist < splat_radius * 2.0:  # Hit
                        return t

        return None

    def _query_nearest(
        self,
        position: np.ndarray,
        radius: float = 1.0
    ) -> List[int]:
        """
        Query splats within radius of position.

        Args:
            position: (3,) query position
            radius: Query radius

        Returns:
            List of splat indices
        """
        # Get voxel coords
        voxel_coord = ((position - self.bounds_min) / self.voxel_size).astype(int)

        # Check neighboring voxels
        radius_voxels = int(np.ceil(radius / self.voxel_size))

        indices = []

        for dx in range(-radius_voxels, radius_voxels + 1):
            for dy in range(-radius_voxels, radius_voxels + 1):
                for dz in range(-radius_voxels, radius_voxels + 1):
                    check_coord = tuple(voxel_coord + np.array([dx, dy, dz]))

                    if check_coord in self.voxel_map:
                        # Filter by actual distance
                        for idx in self.voxel_map[check_coord]:
                            dist = np.linalg.norm(self.splat.positions[idx] - position)

                            if dist < radius:
                                indices.append(idx)

        return indices

    def project_to_floor(self, position: np.ndarray) -> np.ndarray:
        """
        Project position to floor plane.

        Args:
            position: (3,) 3D position

        Returns:
            (3,) projected position on floor
        """
        # Simple z-projection for now
        projected = position.copy()
        projected[2] = self.floor_height

        return projected

    def is_on_mat(self, position: np.ndarray) -> bool:
        """
        Check if position is on mat area.

        Args:
            position: (3,) 3D position

        Returns:
            True if on mat
        """
        # For now, assume mat is central region
        # In real implementation, use semantic labels

        mat_bounds = self.metadata.get('mat_bounds', None)

        if mat_bounds is None:
            # Default: central 6m x 6m
            center = (self.bounds_min + self.bounds_max) / 2
            mat_bounds = {
                'min': center - np.array([3, 3, 0]),
                'max': center + np.array([3, 3, 2])
            }

        in_x = mat_bounds['min'][0] <= position[0] <= mat_bounds['max'][0]
        in_y = mat_bounds['min'][1] <= position[1] <= mat_bounds['max'][1]
        in_z = mat_bounds['min'][2] <= position[2] <= mat_bounds['max'][2]

        return in_x and in_y and in_z

    def get_floor_height(self) -> float:
        """Get floor height."""
        return self.floor_height

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get spatial bounds."""
        return self.bounds_min, self.bounds_max

    @staticmethod
    def from_file(ply_path: str, metadata_path: Optional[str] = None) -> 'GymPrior':
        """
        Load gym prior from file.

        Args:
            ply_path: Path to .ply file
            metadata_path: Optional path to JSON metadata

        Returns:
            GymPrior instance
        """
        splat = GaussianSplatLoader.load_ply(ply_path)

        metadata = {}
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        return GymPrior(splat, metadata)

    def save_metadata(self, path: str):
        """Save metadata to JSON."""
        metadata = {
            'floor_height': float(self.floor_height),
            'bounds_min': self.bounds_min.tolist(),
            'bounds_max': self.bounds_max.tolist(),
            'num_splats': len(self.splat),
            **self.metadata
        }

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)


def test_gym_prior():
    """Test gym prior."""
    print("Testing Gym Prior...")

    # Create synthetic gym splat
    # Simulate a simple gym: floor + walls
    N = 500

    # Floor (z=0, 10m x 10m)
    floor_points = []
    for x in np.linspace(-5, 5, 20):
        for y in np.linspace(-5, 5, 20):
            floor_points.append([x, y, 0])

    floor_points = np.array(floor_points)

    # Add some noise
    floor_points += np.random.randn(*floor_points.shape) * 0.05

    positions = floor_points
    colors = np.ones((len(positions), 3)) * 0.8
    opacities = np.ones(len(positions)) * 0.9
    scales = np.ones((len(positions), 3)) * 0.2
    rotations = np.tile([1, 0, 0, 0], (len(positions), 1)).astype(np.float32)

    splat = GaussianSplat(
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations
    )

    # Create gym prior
    gym_prior = GymPrior(splat)

    print(f"Gym prior: {len(splat)} splats")
    print(f"Floor height: {gym_prior.get_floor_height():.2f}")
    print(f"Bounds: {gym_prior.get_bounds()}")

    # Test depth query
    ray_origin = np.array([0, 0, 2])
    ray_direction = np.array([0, 0, -1])  # Down

    depth = gym_prior.query_depth(ray_origin, ray_direction, max_distance=5.0)

    if depth:
        print(f"Depth query: {depth:.2f}m")
        hit_point = ray_origin + ray_direction * depth
        print(f"Hit point: {hit_point}")
    else:
        print("No intersection")

    # Test mat query
    test_pos = np.array([0, 0, 0.5])
    on_mat = gym_prior.is_on_mat(test_pos)
    print(f"Position {test_pos} on mat: {on_mat}")

    print("✓ PASS")


if __name__ == "__main__":
    test_gym_prior()
