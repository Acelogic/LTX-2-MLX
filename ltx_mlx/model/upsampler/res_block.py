"""Residual blocks for upsampler."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class GroupNorm(nn.Module):
    """
    Group normalization layer.

    Divides channels into groups and normalizes within each group.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        # Learnable parameters
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply group normalization.

        Args:
            x: Input tensor of shape [B, C, ...] where ... can be H, W or D, H, W.

        Returns:
            Normalized tensor.
        """
        # Get shape info
        original_shape = x.shape
        b = x.shape[0]
        c = x.shape[1]
        spatial_shape = x.shape[2:]

        # Reshape to [B, G, C//G, ...]
        g = self.num_groups
        x = x.reshape(b, g, c // g, *spatial_shape)

        # Compute mean and variance over channels within each group
        # Axes: [2, ...] = channel and spatial dimensions within each group
        axes = tuple(range(2, x.ndim))
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back to [B, C, ...]
        x = x.reshape(original_shape)

        # Apply affine transformation
        # Reshape weight and bias for broadcasting
        shape = [1, c] + [1] * len(spatial_shape)
        weight = self.weight.reshape(shape)
        bias = self.bias.reshape(shape)

        return x * weight + bias


class ResBlock(nn.Module):
    """
    Residual block with two convolutions and group normalization.

    Architecture:
    1. Conv -> GroupNorm -> SiLU
    2. Conv -> GroupNorm
    3. Add residual -> SiLU
    """

    def __init__(
        self,
        channels: int,
        mid_channels: Optional[int] = None,
        dims: int = 2,
        num_groups: int = 32,
    ):
        """
        Initialize residual block.

        Args:
            channels: Number of input/output channels.
            mid_channels: Number of intermediate channels (defaults to channels).
            dims: Dimensionality (2 for 2D, 3 for 3D).
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()

        if mid_channels is None:
            mid_channels = channels

        self.dims = dims

        # For MLX, we'll use 2D convolutions even for 3D data
        # by processing frames separately
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm(num_groups, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(num_groups, channels)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W] for 2D or processed [B*F, C, H, W].

        Returns:
            Output tensor with same shape.
        """
        residual = x

        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)

        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)

        # Add residual and activate
        x = nn.silu(x + residual)

        return x


class ResBlock3D(nn.Module):
    """
    3D Residual block that processes spatial dimensions per-frame.

    For video data in [B, C, F, H, W] format, this block:
    1. Reshapes to [B*F, C, H, W]
    2. Applies 2D convolutions
    3. Reshapes back to [B, C, F, H, W]
    """

    def __init__(
        self,
        channels: int,
        mid_channels: Optional[int] = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.block = ResBlock(
            channels=channels,
            mid_channels=mid_channels,
            dims=2,
            num_groups=num_groups,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for 5D tensor.

        Args:
            x: Input tensor [B, C, F, H, W].

        Returns:
            Output tensor [B, C, F, H, W].
        """
        b, c, f, h, w = x.shape

        # Reshape to [B*F, C, H, W]
        x = x.transpose(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(b * f, c, h, w)

        # Apply 2D ResBlock
        x = self.block(x)

        # Reshape back to [B, C, F, H, W]
        x = x.reshape(b, f, c, h, w)
        x = x.transpose(0, 2, 1, 3, 4)

        return x
