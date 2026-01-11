"""Latent-index based conditioning for video generation."""

import mlx.core as mx

from LTX_2_MLX.types import LatentState
from LTX_2_MLX.conditioning.tools import VideoLatentTools


class ConditioningError(Exception):
    """Error raised when conditioning cannot be applied."""

    pass


class VideoConditionByLatentIndex:
    """
    Conditions video generation by injecting latents at a specific latent frame index.

    Replaces tokens in the latent state at positions corresponding to latent_idx,
    and sets denoise strength according to the strength parameter.

    This is used for image-to-video or video-to-video generation where specific
    frames are replaced with encoded content.
    """

    def __init__(self, latent: mx.array, strength: float, latent_idx: int):
        """
        Initialize latent index conditioning.

        Args:
            latent: Encoded latent to inject (B, C, F_cond, H, W).
            strength: Denoising strength (0 = keep clean, 1 = full denoise).
            latent_idx: Starting latent frame index for injection.
        """
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """
        Apply latent conditioning by replacing tokens at specified indices.

        Args:
            latent_state: Current patchified latent state.
            latent_tools: Tools for patchification and position computation.

        Returns:
            Modified latent state with tokens replaced at specified positions.
        """
        cond_shape = self.latent.shape
        cond_batch, cond_channels, _, cond_height, cond_width = cond_shape

        tgt_shape = latent_tools.target_shape

        # Validate shapes match (except frames dimension)
        if (cond_batch, cond_channels, cond_height, cond_width) != (
            tgt_shape.batch,
            tgt_shape.channels,
            tgt_shape.height,
            tgt_shape.width,
        ):
            raise ConditioningError(
                f"Cannot apply image conditioning item to latent with shape {tgt_shape}. "
                f"Expected shape is ({tgt_shape.batch}, {tgt_shape.channels}, _, "
                f"{tgt_shape.height}, {tgt_shape.width}). "
                "Make sure the image and latent have the same spatial shape."
            )

        # Patchify the conditioning latent
        tokens = latent_tools.patchifier.patchify(self.latent)

        # Calculate token range for replacement
        # Get token count up to latent_idx
        start_shape = tgt_shape._replace(frames=self.latent_idx)
        start_token = latent_tools.patchifier.get_token_count(start_shape)
        stop_token = start_token + tokens.shape[1]

        # Validate bounds
        max_tokens = latent_tools.patchifier.get_token_count(tgt_shape)
        if stop_token > max_tokens:
            raise ValueError(
                f"Conditioning tokens exceed latent sequence length: "
                f"stop_token={stop_token} > max_tokens={max_tokens}. "
                f"latent_idx={self.latent_idx}, tokens.shape={tokens.shape}"
            )

        # Create new arrays with conditioning applied
        # Replace latent tokens
        latent_before = latent_state.latent[:, :start_token]
        latent_after = latent_state.latent[:, stop_token:]
        new_latent = mx.concatenate([latent_before, tokens, latent_after], axis=1)

        # Replace clean_latent tokens
        clean_before = latent_state.clean_latent[:, :start_token]
        clean_after = latent_state.clean_latent[:, stop_token:]
        new_clean_latent = mx.concatenate([clean_before, tokens, clean_after], axis=1)

        # Update denoise_mask
        mask_before = latent_state.denoise_mask[:, :start_token]
        mask_after = latent_state.denoise_mask[:, stop_token:]
        new_mask = mx.full(
            (tokens.shape[0], tokens.shape[1], 1),
            1.0 - self.strength,
            dtype=latent_state.denoise_mask.dtype,
        )
        new_denoise_mask = mx.concatenate([mask_before, new_mask, mask_after], axis=1)

        return LatentState(
            latent=new_latent,
            denoise_mask=new_denoise_mask,
            positions=latent_state.positions,  # Positions unchanged
            clean_latent=new_clean_latent,
        )
