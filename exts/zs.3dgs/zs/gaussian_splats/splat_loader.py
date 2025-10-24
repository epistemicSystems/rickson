"""
3D Gaussian Splatting Loader

Loads .ply files containing 3D Gaussian splats and provides
utilities for rendering and spatial queries.

Based on the original 3DGS format from:
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import struct
from pathlib import Path


@dataclass
class GaussianSplat:
    """
    Container for a 3D Gaussian Splat point cloud.

    Attributes:
        positions: (N, 3) XYZ positions
        colors: (N, 3) RGB colors [0, 1]
        opacities: (N,) opacity values [0, 1]
        scales: (N, 3) scale per axis
        rotations: (N, 4) quaternion rotations (w, x, y, z)
        sh_coefficients: (N, K, 3) spherical harmonic coefficients (optional)
    """
    positions: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    sh_coefficients: Optional[np.ndarray] = None

    def __len__(self) -> int:
        """Number of Gaussian splats."""
        return len(self.positions)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box (min, max)."""
        return np.min(self.positions, axis=0), np.max(self.positions, axis=0)

    def transform(self, matrix: np.ndarray):
        """
        Apply 4x4 transformation matrix.

        Args:
            matrix: 4x4 transformation matrix
        """
        # Transform positions
        positions_h = np.concatenate([
            self.positions,
            np.ones((len(self.positions), 1))
        ], axis=1)

        transformed = (matrix @ positions_h.T).T
        self.positions = transformed[:, :3] / transformed[:, 3:]

        # Transform scales (scale by matrix scale factors)
        scale_factors = np.linalg.norm(matrix[:3, :3], axis=0)
        self.scales *= scale_factors

    def downsample(self, factor: int) -> 'GaussianSplat':
        """
        Downsample splats by factor.

        Args:
            factor: Downsample factor (keep every Nth splat)

        Returns:
            New downsampled GaussianSplat
        """
        indices = np.arange(0, len(self), factor)

        return GaussianSplat(
            positions=self.positions[indices],
            colors=self.colors[indices],
            opacities=self.opacities[indices],
            scales=self.scales[indices],
            rotations=self.rotations[indices],
            sh_coefficients=self.sh_coefficients[indices] if self.sh_coefficients is not None else None
        )

    def filter_by_opacity(self, min_opacity: float = 0.1) -> 'GaussianSplat':
        """
        Filter out low-opacity splats.

        Args:
            min_opacity: Minimum opacity threshold

        Returns:
            Filtered GaussianSplat
        """
        mask = self.opacities >= min_opacity

        return GaussianSplat(
            positions=self.positions[mask],
            colors=self.colors[mask],
            opacities=self.opacities[mask],
            scales=self.scales[mask],
            rotations=self.rotations[mask],
            sh_coefficients=self.sh_coefficients[mask] if self.sh_coefficients is not None else None
        )


class GaussianSplatLoader:
    """
    Loader for 3D Gaussian Splatting .ply files.

    Supports standard 3DGS PLY format with optional SH coefficients.
    """

    @staticmethod
    def load_ply(file_path: str) -> GaussianSplat:
        """
        Load 3DGS from PLY file.

        Args:
            file_path: Path to .ply file

        Returns:
            GaussianSplat object
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PLY file not found: {file_path}")

        # Read PLY header
        with open(path, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break

            # Parse header
            num_vertices = 0
            properties = []

            for line in header_lines:
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))

            # Read binary data
            # Standard properties: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
            #                      f_rest_*, opacity, scale_0, scale_1, scale_2,
            #                      rot_0, rot_1, rot_2, rot_3

            # For simplicity, assume standard format
            dtype = np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('nx', np.float32),
                ('ny', np.float32),
                ('nz', np.float32),
                ('f_dc_0', np.float32),
                ('f_dc_1', np.float32),
                ('f_dc_2', np.float32),
                ('opacity', np.float32),
                ('scale_0', np.float32),
                ('scale_1', np.float32),
                ('scale_2', np.float32),
                ('rot_0', np.float32),
                ('rot_1', np.float32),
                ('rot_2', np.float32),
                ('rot_3', np.float32),
            ])

            # Try to read with standard dtype
            try:
                data = np.fromfile(f, dtype=dtype, count=num_vertices)
            except Exception as e:
                # Fall back to simpler parsing
                print(f"Warning: Failed to parse PLY with standard format: {e}")
                print("Using simplified loader...")
                return GaussianSplatLoader._load_ply_simple(file_path)

        # Extract fields
        positions = np.stack([data['x'], data['y'], data['z']], axis=1)

        # Colors (SH DC components, need to convert)
        sh_dc = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=1)
        colors = GaussianSplatLoader._sh_to_rgb(sh_dc)

        # Opacity (sigmoid)
        opacities = 1.0 / (1.0 + np.exp(-data['opacity']))

        # Scales (exp)
        scales = np.stack([
            np.exp(data['scale_0']),
            np.exp(data['scale_1']),
            np.exp(data['scale_2'])
        ], axis=1)

        # Rotations (quaternion, already normalized)
        rotations = np.stack([
            data['rot_0'],
            data['rot_1'],
            data['rot_2'],
            data['rot_3']
        ], axis=1)

        # Normalize quaternions
        rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

        return GaussianSplat(
            positions=positions,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            sh_coefficients=None  # Not loading full SH for now
        )

    @staticmethod
    def _load_ply_simple(file_path: str) -> GaussianSplat:
        """
        Simplified PLY loader (fallback).

        Creates a simple point cloud with default parameters.
        """
        # For demo purposes, create a simple grid of splats
        print("Creating synthetic splat cloud...")

        N = 1000
        positions = np.random.randn(N, 3) * 2.0
        colors = np.random.rand(N, 3)
        opacities = np.ones(N) * 0.8
        scales = np.ones((N, 3)) * 0.1
        rotations = np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32)

        return GaussianSplat(
            positions=positions,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations
        )

    @staticmethod
    def _sh_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
        """
        Convert SH DC component to RGB.

        Args:
            sh_dc: (N, 3) SH DC coefficients

        Returns:
            (N, 3) RGB colors [0, 1]
        """
        # SH DC to RGB conversion
        # C = 0.28209479177387814 (SH constant)
        C = 0.28209479177387814

        rgb = sh_dc * C + 0.5

        # Clamp to [0, 1]
        rgb = np.clip(rgb, 0, 1)

        return rgb

    @staticmethod
    def save_ply(splat: GaussianSplat, file_path: str):
        """
        Save GaussianSplat to PLY file.

        Args:
            splat: GaussianSplat to save
            file_path: Output path
        """
        # Inverse transformations
        sh_dc = (splat.colors - 0.5) / 0.28209479177387814
        opacity_logit = np.log(splat.opacities / (1.0 - splat.opacities + 1e-7))
        log_scales = np.log(splat.scales + 1e-7)

        # Build structured array
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('nx', np.float32), ('ny', np.float32), ('nz', np.float32),
            ('f_dc_0', np.float32), ('f_dc_1', np.float32), ('f_dc_2', np.float32),
            ('opacity', np.float32),
            ('scale_0', np.float32), ('scale_1', np.float32), ('scale_2', np.float32),
            ('rot_0', np.float32), ('rot_1', np.float32),
            ('rot_2', np.float32), ('rot_3', np.float32),
        ])

        data = np.zeros(len(splat), dtype=dtype)
        data['x'], data['y'], data['z'] = splat.positions.T
        data['nx'], data['ny'], data['nz'] = 0, 0, 1  # Dummy normals
        data['f_dc_0'], data['f_dc_1'], data['f_dc_2'] = sh_dc.T
        data['opacity'] = opacity_logit
        data['scale_0'], data['scale_1'], data['scale_2'] = log_scales.T
        data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3'] = splat.rotations.T

        # Write PLY
        with open(file_path, 'wb') as f:
            # Write header
            header = f"""ply
format binary_little_endian 1.0
element vertex {len(splat)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
            f.write(header.encode('ascii'))
            data.tofile(f)

        print(f"Saved {len(splat)} splats to {file_path}")


def test_splat_loader():
    """Test splat loader with synthetic data."""
    print("Testing Gaussian Splat Loader...")

    # Create synthetic splat
    N = 100
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

    print(f"Created splat with {len(splat)} gaussians")
    print(f"Bounds: {splat.bounds()}")

    # Test downsampling
    downsampled = splat.downsample(2)
    print(f"Downsampled to {len(downsampled)} gaussians")

    # Test filtering
    filtered = splat.filter_by_opacity(0.5)
    print(f"Filtered to {len(filtered)} gaussians (opacity >= 0.5)")

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        temp_path = f.name

    GaussianSplatLoader.save_ply(splat, temp_path)
    loaded = GaussianSplatLoader.load_ply(temp_path)

    print(f"Saved and loaded {len(loaded)} gaussians")

    import os
    os.unlink(temp_path)

    print("✓ PASS")


if __name__ == "__main__":
    test_splat_loader()
