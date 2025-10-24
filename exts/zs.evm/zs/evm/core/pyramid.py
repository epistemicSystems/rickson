"""
Spatial Pyramid Construction

Builds Gaussian and Laplacian pyramids for multi-scale EVM processing.
Pure functions following Rich Hickey principles.
"""

import numpy as np
import cv2
from typing import List, Tuple


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Gaussian pyramid by iterative blur and downsample.

    Args:
        image: Input image (H, W, C)
        levels: Number of pyramid levels

    Returns:
        List of images from finest to coarsest [L0, L1, ..., LN]
    """
    pyramid = [image]
    current = image

    for i in range(levels - 1):
        # Gaussian blur with 5x5 kernel (sigma ~ 0.83)
        blurred = cv2.GaussianBlur(current, (5, 5), 0)
        # Downsample by factor of 2
        downsampled = cv2.pyrDown(blurred)
        pyramid.append(downsampled)
        current = downsampled

    return pyramid


def build_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """
    Build Laplacian pyramid from Gaussian pyramid.

    Each level is: L[i] = G[i] - upsample(G[i+1])

    Args:
        gaussian_pyramid: Gaussian pyramid from build_gaussian_pyramid

    Returns:
        Laplacian pyramid (same length as input)
    """
    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):
        # Upsample next level
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1])

        # Ensure same size (handle odd dimensions)
        h, w = gaussian_pyramid[i].shape[:2]
        upsampled = cv2.resize(upsampled, (w, h))

        # Laplacian = current - upsampled_next
        laplacian = gaussian_pyramid[i].astype(np.float32) - upsampled.astype(np.float32)
        laplacian_pyramid.append(laplacian)

    # Last level is just the coarsest Gaussian
    laplacian_pyramid.append(gaussian_pyramid[-1].astype(np.float32))

    return laplacian_pyramid


def collapse_laplacian_pyramid(laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct image from Laplacian pyramid.

    Args:
        laplacian_pyramid: Laplacian pyramid to collapse

    Returns:
        Reconstructed image
    """
    # Start from coarsest level
    current = laplacian_pyramid[-1]

    # Work from coarse to fine
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # Upsample current
        upsampled = cv2.pyrUp(current)

        # Match size
        h, w = laplacian_pyramid[i].shape[:2]
        upsampled = cv2.resize(upsampled, (w, h))

        # Add Laplacian details
        current = upsampled + laplacian_pyramid[i]

    return current


def amplify_spatial_frequencies(
    laplacian_pyramid: List[np.ndarray],
    alpha: float,
    wavelength_attenuation: bool = True
) -> List[np.ndarray]:
    """
    Amplify each level of the Laplacian pyramid.

    Args:
        laplacian_pyramid: Input pyramid
        alpha: Base amplification factor
        wavelength_attenuation: If True, attenuate higher frequencies

    Returns:
        Amplified pyramid
    """
    amplified = []

    for level_idx, level in enumerate(laplacian_pyramid):
        if wavelength_attenuation:
            # Attenuate finer scales to reduce noise
            # alpha_level = alpha * (level_idx + 1) / len(laplacian_pyramid)
            # More aggressive: higher levels get less amplification
            alpha_level = alpha * np.power(0.7, level_idx)
        else:
            alpha_level = alpha

        amplified.append(level * alpha_level)

    return amplified


def test_pyramid_reconstruction():
    """Test that pyramid build/collapse is near-lossless."""
    # Create test image
    test_img = np.random.rand(256, 256, 3).astype(np.float32) * 255

    # Build and collapse
    g_pyr = build_gaussian_pyramid(test_img, levels=4)
    l_pyr = build_laplacian_pyramid(g_pyr)
    reconstructed = collapse_laplacian_pyramid(l_pyr)

    # Check reconstruction error
    error = np.abs(test_img - reconstructed).mean()
    print(f"Pyramid reconstruction error: {error:.4f} (should be < 1.0)")

    return error < 1.0


if __name__ == "__main__":
    # Run self-test
    success = test_pyramid_reconstruction()
    print(f"Pyramid test: {'PASS' if success else 'FAIL'}")
