"""Latent tools for building and managing latent states."""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from LTX_2_MLX.types import (
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)
from LTX_2_MLX.components.patchifiers import (
    VideoLatentPatchifier,
    get_pixel_coords,
)


DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


@dataclass(frozen=True)
class VideoLatentTools:
    """
    Tools for building video latent states.

    Provides utilities for creating initial states, patchifying/unpatchifying,
    and clearing conditioning tokens.
    """

    patchifier: VideoLatentPatchifier
    target_shape: VideoLatentShape
    fps: float
    scale_factors: SpatioTemporalScaleFactors = DEFAULT_SCALE_FACTORS
    causal_fix: bool = True

    def create_initial_state(
        self,
        dtype: mx.Dtype = mx.float32,
        initial_latent: Optional[mx.array] = None,
    ) -> LatentState:
        """
        Create an initial latent state.

        Args:
            dtype: Data type for the tensors.
            initial_latent: Optional initial latent tensor. If None, creates zeros.

        Returns:
            Patchified LatentState ready for diffusion.
        """
        if initial_latent is not None:
            if initial_latent.shape != self.target_shape.to_tuple():
                raise ValueError(
                    f"Initial latent shape {initial_latent.shape} does not match "
                    f"target shape {self.target_shape.to_tuple()}"
                )
        else:
            initial_latent = mx.zeros(self.target_shape.to_tuple(), dtype=dtype)

        clean_latent = initial_latent

        denoise_mask = mx.ones(self.target_shape.mask_shape().to_tuple(), dtype=mx.float32)

        latent_coords = self.patchifier.get_patch_grid_bounds(output_shape=self.target_shape)

        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.scale_factors,
            causal_fix=self.causal_fix,
        ).astype(mx.float32)

        # Convert temporal positions to time in seconds
        temporal_positions = positions[:, 0:1, ...] / self.fps
        other_positions = positions[:, 1:, ...]
        positions = mx.concatenate([temporal_positions, other_positions], axis=1)

        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=positions,
                clean_latent=clean_latent,
            )
        )

    def patchify(self, latent_state: LatentState) -> LatentState:
        """
        Patchify the latent state.

        Converts 5D latents (B, C, F, H, W) to 3D patch sequences (B, N, D).

        Args:
            latent_state: State with unpatchified latents.

        Returns:
            State with patchified latents.
        """
        latent = self.patchifier.patchify(latent_state.latent)
        clean_latent = self.patchifier.patchify(latent_state.clean_latent)
        denoise_mask = self.patchifier.patchify(latent_state.denoise_mask)

        return latent_state.replace(
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
        )

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        """
        Unpatchify the latent state.

        Converts 3D patch sequences (B, N, D) back to 5D latents (B, C, F, H, W).

        Args:
            latent_state: State with patchified latents.

        Returns:
            State with unpatchified latents.
        """
        latent = self.patchifier.unpatchify(
            latent_state.latent, output_shape=self.target_shape
        )
        clean_latent = self.patchifier.unpatchify(
            latent_state.clean_latent, output_shape=self.target_shape
        )
        denoise_mask = self.patchifier.unpatchify(
            latent_state.denoise_mask, output_shape=self.target_shape.mask_shape()
        )

        return latent_state.replace(
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        """
        Clear the conditioning from the latent state.

        This method removes extra tokens from the end of the latent.
        Therefore, conditioning items should add extra tokens ONLY to the end.

        Args:
            latent_state: State with potential conditioning tokens appended.

        Returns:
            State with only the original tokens.
        """
        num_tokens = self.patchifier.get_token_count(self.target_shape)

        latent = latent_state.latent[:, :num_tokens]
        clean_latent = latent_state.clean_latent[:, :num_tokens]
        denoise_mask = mx.ones_like(latent_state.denoise_mask)[:, :num_tokens]
        positions = latent_state.positions[:, :, :num_tokens]

        return LatentState(
            latent=latent,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=clean_latent,
        )
