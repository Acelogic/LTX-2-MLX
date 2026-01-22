"""Noise injection strategies for LTX-2 diffusion sampling."""

from typing import Optional, Protocol

import mlx.core as mx

from LTX_2_MLX.types import LatentState


class NoiserProtocol(Protocol):
    """Protocol for adding noise to a latent state during diffusion."""

    def __call__(self, latent_state: LatentState, noise_scale: float) -> LatentState:
        """Add noise to the latent state."""
        ...


class GaussianNoiser:
    """
    Adds Gaussian noise to a latent state, scaled by the denoise mask.

    The noise is blended with the original latent based on the denoise mask,
    where mask=1 means full noise and mask=0 means original latent preserved.
    """

    def __init__(self, key: Optional[mx.array] = None):
        """
        Initialize the Gaussian noiser.

        Args:
            key: Optional MLX random key for reproducibility. If None, uses
                 the global random state.
        """
        self.key = key

    def __call__(
        self, latent_state: LatentState, noise_scale: float = 1.0
    ) -> LatentState:
        """
        Add Gaussian noise to the latent state.

        Args:
            latent_state: The current latent state to add noise to.
            noise_scale: Scale factor for the noise (typically sigma value).

        Returns:
            New LatentState with noise added according to the denoise mask.
        """
        # Generate Gaussian noise
        if self.key is not None:
            noise = mx.random.normal(
                shape=latent_state.latent.shape,
                dtype=latent_state.latent.dtype,
                key=self.key,
            )
            # Update key for next call
            self.key = mx.random.split(self.key)[0]
        else:
            noise = mx.random.normal(
                shape=latent_state.latent.shape,
                dtype=latent_state.latent.dtype,
            )

        # Scale the denoise mask by noise_scale
        # Handle both 2D (B, T) and 3D (B, T, 1) masks from patchification
        mask = latent_state.denoise_mask
        if mask.ndim == 2:
            # Expand mask to broadcast with latent: (B, T) -> (B, T, 1)
            scaled_mask = mx.expand_dims(mask, axis=-1) * noise_scale
        else:
            # Mask is already 3D (B, T, 1), use directly
            scaled_mask = mask * noise_scale

        # Blend noise with original latent based on mask
        # Where mask=1: use noise, where mask=0: keep original
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)

        return latent_state.replace(latent=latent.astype(latent_state.latent.dtype))


class DeterministicNoiser:
    """
    A noiser that uses a fixed noise pattern for deterministic generation.

    Useful for reproducible results when debugging or testing.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize with a seed.

        Args:
            seed: Random seed for generating the fixed noise pattern.
        """
        self.seed = seed

    def __call__(
        self, latent_state: LatentState, noise_scale: float = 1.0
    ) -> LatentState:
        """
        Add deterministic noise to the latent state.

        Args:
            latent_state: The current latent state to add noise to.
            noise_scale: Scale factor for the noise.

        Returns:
            New LatentState with deterministic noise added.
        """
        key = mx.random.key(self.seed)
        noise = mx.random.normal(
            shape=latent_state.latent.shape,
            dtype=latent_state.latent.dtype,
            key=key,
        )

        # Handle both 2D (B, T) and 3D (B, T, 1) masks from patchification
        mask = latent_state.denoise_mask
        if mask.ndim == 2:
            scaled_mask = mx.expand_dims(mask, axis=-1) * noise_scale
        else:
            scaled_mask = mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)

        return latent_state.replace(latent=latent.astype(latent_state.latent.dtype))
