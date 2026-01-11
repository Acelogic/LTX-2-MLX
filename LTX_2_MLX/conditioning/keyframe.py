"""Keyframe-based conditioning for video generation."""

import mlx.core as mx

from LTX_2_MLX.types import LatentState, VideoLatentShape
from LTX_2_MLX.components.patchifiers import get_pixel_coords
from LTX_2_MLX.conditioning.tools import VideoLatentTools


class VideoConditionByKeyframeIndex:
    """
    Conditions video generation on keyframe latents at a specific frame index.

    Appends keyframe tokens to the latent state with positions offset by frame_idx,
    and sets denoise strength according to the strength parameter.

    This is used for image-to-video generation where keyframes guide the output.
    """

    def __init__(self, keyframes: mx.array, frame_idx: int, strength: float):
        """
        Initialize keyframe conditioning.

        Args:
            keyframes: Encoded keyframe latents (B, C, 1, H, W).
            frame_idx: Frame index where the keyframe should be placed.
            strength: Denoising strength (0 = keep clean, 1 = full denoise).
        """
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """
        Apply keyframe conditioning to the latent state.

        Appends keyframe tokens to the end of the latent sequence with
        appropriate position offsets.

        Args:
            latent_state: Current patchified latent state.
            latent_tools: Tools for patchification and position computation.

        Returns:
            Modified latent state with keyframe tokens appended.
        """
        # Patchify keyframe: (B, C, 1, H, W) -> (B, N_kf, D)
        tokens = latent_tools.patchifier.patchify(self.keyframes)

        # Get patch grid bounds for keyframe
        keyframe_shape = VideoLatentShape.from_shape(self.keyframes.shape)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=keyframe_shape
        )

        # Convert to pixel coordinates
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            # Only apply causal fix if this is the first frame
            causal_fix=latent_tools.causal_fix if self.frame_idx == 0 else False,
        )

        # Offset temporal positions by frame_idx and convert to time in seconds
        positions = positions.astype(mx.float32)
        temporal_positions = (positions[:, 0:1, ...] + self.frame_idx) / latent_tools.fps
        other_positions = positions[:, 1:, ...]
        positions = mx.concatenate([temporal_positions, other_positions], axis=1)

        # Create denoise mask (1 - strength means mostly keep clean)
        denoise_mask = mx.full(
            shape=(tokens.shape[0], tokens.shape[1], 1),
            vals=1.0 - self.strength,
            dtype=self.keyframes.dtype,
        )

        # Concatenate to existing state
        return LatentState(
            latent=mx.concatenate([latent_state.latent, tokens], axis=1),
            denoise_mask=mx.concatenate([latent_state.denoise_mask, denoise_mask], axis=1),
            positions=mx.concatenate([latent_state.positions, positions], axis=2),
            clean_latent=mx.concatenate([latent_state.clean_latent, tokens], axis=1),
        )
