"""Latent upsampler for LTX-2."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .pixel_shuffle import PixelShuffleND, pixel_shuffle_2d
from .res_block import GroupNorm, ResBlock


class LatentUpsampler(nn.Module):
    """
    Latent upsampler for 2x spatial upscaling.

    Architecture:
    1. Initial conv -> GroupNorm -> SiLU
    2. Stack of ResBlocks (pre-upsample)
    3. Conv to expand channels -> PixelShuffle (2x)
    4. Stack of ResBlocks (post-upsample)
    5. Final conv to output channels

    This upsampler operates on video latents in [B, C, F, H, W] format,
    processing each frame independently with 2D convolutions.
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        spatial_scale: int = 2,
        num_groups: int = 32,
    ):
        """
        Initialize latent upsampler.

        Args:
            in_channels: Number of input/output channels (128 for LTX VAE).
            mid_channels: Number of intermediate channels (512).
            num_blocks_per_stage: Number of ResBlocks per stage (4).
            spatial_scale: Spatial upscale factor (2).
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.spatial_scale = spatial_scale

        # Initial projection
        self.initial_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = GroupNorm(num_groups, mid_channels)

        # Pre-upsample ResBlocks
        self.res_blocks = [
            ResBlock(mid_channels, num_groups=num_groups)
            for _ in range(num_blocks_per_stage)
        ]

        # Upsampler: expand channels then pixel shuffle
        # For 2x spatial upscale: need 4x channels (2*2)
        expand_channels = mid_channels * (spatial_scale * spatial_scale)
        self.upsample_conv = nn.Conv2d(mid_channels, expand_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffleND(dims=2, upscale_factors=(spatial_scale, spatial_scale, 1))

        # Post-upsample ResBlocks
        self.post_upsample_res_blocks = [
            ResBlock(mid_channels, num_groups=num_groups)
            for _ in range(num_blocks_per_stage)
        ]

        # Final projection back to input channels
        self.final_conv = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, latent: mx.array) -> mx.array:
        """
        Upsample latent by 2x spatially.

        Args:
            latent: Input latent of shape [B, C, F, H, W].

        Returns:
            Upsampled latent of shape [B, C, F, H*2, W*2].
        """
        b, c, f, h, w = latent.shape

        # Reshape to process frames as batch: [B, C, F, H, W] -> [B*F, C, H, W]
        x = latent.transpose(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(b * f, c, h, w)

        # Initial projection
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)

        # Pre-upsample ResBlocks
        for block in self.res_blocks:
            x = block(x)

        # Upsample: expand channels then pixel shuffle
        x = self.upsample_conv(x)
        x = pixel_shuffle_2d(x, (self.spatial_scale, self.spatial_scale))

        # Post-upsample ResBlocks
        for block in self.post_upsample_res_blocks:
            x = block(x)

        # Final projection
        x = self.final_conv(x)

        # Reshape back: [B*F, C, H*2, W*2] -> [B, C, F, H*2, W*2]
        new_h = h * self.spatial_scale
        new_w = w * self.spatial_scale
        x = x.reshape(b, f, self.in_channels, new_h, new_w)
        x = x.transpose(0, 2, 1, 3, 4)  # [B, C, F, H*2, W*2]

        return x


def upsample_latent(
    latent: mx.array,
    upsampler: LatentUpsampler,
    per_channel_mean: Optional[mx.array] = None,
    per_channel_std: Optional[mx.array] = None,
) -> mx.array:
    """
    Upsample latent with proper normalization handling.

    The upsampler works on un-normalized latents, so we need to:
    1. Un-normalize input latent
    2. Apply upsampler
    3. Re-normalize output

    Args:
        latent: Normalized input latent [B, C, F, H, W].
        upsampler: LatentUpsampler instance.
        per_channel_mean: Per-channel mean for normalization.
        per_channel_std: Per-channel std for normalization.

    Returns:
        Upsampled and normalized latent.
    """
    if per_channel_mean is not None and per_channel_std is not None:
        # Un-normalize: x = x * std + mean
        shape = (1, -1, 1, 1, 1)  # Broadcast shape for [B, C, F, H, W]
        mean = per_channel_mean.reshape(shape)
        std = per_channel_std.reshape(shape)

        latent = latent * std + mean

    # Apply upsampler
    latent = upsampler(latent)

    if per_channel_mean is not None and per_channel_std is not None:
        # Re-normalize: x = (x - mean) / std
        latent = (latent - mean) / std

    return latent


def create_latent_upsampler(
    in_channels: int = 128,
    mid_channels: int = 512,
    num_blocks: int = 4,
    scale: int = 2,
) -> LatentUpsampler:
    """
    Create a latent upsampler with default LTX-2 configuration.

    Args:
        in_channels: Input/output channels (128).
        mid_channels: Intermediate channels (512).
        num_blocks: ResBlocks per stage (4).
        scale: Spatial upscale factor (2).

    Returns:
        Configured LatentUpsampler.
    """
    return LatentUpsampler(
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_blocks_per_stage=num_blocks,
        spatial_scale=scale,
    )
