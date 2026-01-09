"""Pixel shuffle operations for upsampling in MLX."""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def pixel_shuffle_2d(
    x: mx.array,
    upscale_factors: Tuple[int, int] = (2, 2),
) -> mx.array:
    """
    2D pixel shuffle (depth to space) for spatial upsampling.

    Rearranges data from channels to spatial dimensions.
    Input:  [B, C*p1*p2, H, W]
    Output: [B, C, H*p1, W*p2]

    Args:
        x: Input tensor of shape [B, C*p1*p2, H, W].
        upscale_factors: Tuple of (height_factor, width_factor).

    Returns:
        Upsampled tensor of shape [B, C, H*p1, W*p2].
    """
    p1, p2 = upscale_factors
    b, c_in, h, w = x.shape

    # Calculate output channels
    c_out = c_in // (p1 * p2)

    # Reshape: [B, C, p1, p2, H, W]
    x = x.reshape(b, c_out, p1, p2, h, w)

    # Permute: [B, C, H, p1, W, p2]
    x = x.transpose(0, 1, 4, 2, 5, 3)

    # Reshape: [B, C, H*p1, W*p2]
    x = x.reshape(b, c_out, h * p1, w * p2)

    return x


def pixel_shuffle_3d(
    x: mx.array,
    upscale_factors: Tuple[int, int, int] = (2, 2, 2),
) -> mx.array:
    """
    3D pixel shuffle (depth to space) for spatiotemporal upsampling.

    Rearranges data from channels to spatiotemporal dimensions.
    Input:  [B, C*p1*p2*p3, D, H, W]
    Output: [B, C, D*p1, H*p2, W*p3]

    Args:
        x: Input tensor of shape [B, C*p1*p2*p3, D, H, W].
        upscale_factors: Tuple of (depth_factor, height_factor, width_factor).

    Returns:
        Upsampled tensor of shape [B, C, D*p1, H*p2, W*p3].
    """
    p1, p2, p3 = upscale_factors
    b, c_in, d, h, w = x.shape

    # Calculate output channels
    c_out = c_in // (p1 * p2 * p3)

    # Reshape: [B, C, p1, p2, p3, D, H, W]
    x = x.reshape(b, c_out, p1, p2, p3, d, h, w)

    # Permute: [B, C, D, p1, H, p2, W, p3]
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)

    # Reshape: [B, C, D*p1, H*p2, W*p3]
    x = x.reshape(b, c_out, d * p1, h * p2, w * p3)

    return x


def pixel_shuffle_1d(
    x: mx.array,
    upscale_factor: int = 2,
) -> mx.array:
    """
    1D pixel shuffle for temporal upsampling.

    Rearranges data from channels to temporal dimension.
    Input:  [B, C*p, F, H, W]
    Output: [B, C, F*p, H, W]

    Args:
        x: Input tensor of shape [B, C*p, F, H, W].
        upscale_factor: Temporal upscale factor.

    Returns:
        Upsampled tensor of shape [B, C, F*p, H, W].
    """
    p = upscale_factor
    b, c_in, f, h, w = x.shape

    # Calculate output channels
    c_out = c_in // p

    # Reshape: [B, C, p, F, H, W]
    x = x.reshape(b, c_out, p, f, h, w)

    # Permute: [B, C, F, p, H, W]
    x = x.transpose(0, 1, 3, 2, 4, 5)

    # Reshape: [B, C, F*p, H, W]
    x = x.reshape(b, c_out, f * p, h, w)

    return x


class PixelShuffleND(nn.Module):
    """
    N-dimensional pixel shuffle for upsampling.

    Args:
        dims: Number of dimensions to upsample (1, 2, or 3).
        upscale_factors: Upscale factors for each dimension.
    """

    def __init__(
        self,
        dims: int,
        upscale_factors: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        if dims not in [1, 2, 3]:
            raise ValueError("dims must be 1, 2, or 3")

        self.dims = dims
        self.upscale_factors = upscale_factors

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply pixel shuffle.

        Args:
            x: Input tensor.

        Returns:
            Upsampled tensor.
        """
        if self.dims == 3:
            return pixel_shuffle_3d(x, self.upscale_factors)
        elif self.dims == 2:
            return pixel_shuffle_2d(x, self.upscale_factors[:2])
        elif self.dims == 1:
            return pixel_shuffle_1d(x, self.upscale_factors[0])
        else:
            raise ValueError(f"Unsupported dims: {self.dims}")
