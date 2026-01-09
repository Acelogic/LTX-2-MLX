"""Spatial up/downsampling layers for Video VAE."""

import math
from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .convolution import make_conv_nd, PaddingModeType


def space_to_depth(
    x: mx.array,
    stride: Tuple[int, int, int],
) -> mx.array:
    """
    Space-to-depth operation for 5D tensors.

    Rearranges spatial/temporal dimensions into channels:
    (B, C, D, H, W) -> (B, C*p1*p2*p3, D/p1, H/p2, W/p3)

    Args:
        x: Input tensor of shape (B, C, D, H, W).
        stride: Downsampling factors (p1, p2, p3) for depth, height, width.

    Returns:
        Downsampled tensor with increased channels.
    """
    b, c, d, h, w = x.shape
    p1, p2, p3 = stride

    # Reshape: (B, C, D, H, W) -> (B, C, D/p1, p1, H/p2, p2, W/p3, p3)
    x = x.reshape(b, c, d // p1, p1, h // p2, p2, w // p3, p3)
    # Transpose: -> (B, C, p1, p2, p3, D/p1, H/p2, W/p3)
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    # Reshape: -> (B, C*p1*p2*p3, D/p1, H/p2, W/p3)
    x = x.reshape(b, c * p1 * p2 * p3, d // p1, h // p2, w // p3)

    return x


def depth_to_space(
    x: mx.array,
    stride: Tuple[int, int, int],
) -> mx.array:
    """
    Depth-to-space operation for 5D tensors.

    Rearranges channels into spatial/temporal dimensions:
    (B, C*p1*p2*p3, D, H, W) -> (B, C, D*p1, H*p2, W*p3)

    Args:
        x: Input tensor of shape (B, C*p1*p2*p3, D, H, W).
        stride: Upsampling factors (p1, p2, p3) for depth, height, width.

    Returns:
        Upsampled tensor with reduced channels.
    """
    b, c_packed, d, h, w = x.shape
    p1, p2, p3 = stride
    c = c_packed // (p1 * p2 * p3)

    # Reshape: (B, C*p1*p2*p3, D, H, W) -> (B, C, p1, p2, p3, D, H, W)
    x = x.reshape(b, c, p1, p2, p3, d, h, w)
    # Transpose: -> (B, C, D, p1, H, p2, W, p3)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    # Reshape: -> (B, C, D*p1, H*p2, W*p3)
    x = x.reshape(b, c, d * p1, h * p2, w * p3)

    return x


class SpaceToDepthDownsample(nn.Module):
    """
    Downsampling layer using space-to-depth with a residual connection.

    Combines a convolutional path with a direct space-to-depth shortcut
    for better gradient flow.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """
        Initialize SpaceToDepthDownsample.

        Args:
            dims: Dimension specification for convolutions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Downsampling factors (temporal, height, width).
            spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()

        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels

        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """
        Apply downsampling.

        Args:
            x: Input tensor of shape (B, C, D, H, W).
            causal: Whether to use causal convolution.

        Returns:
            Downsampled tensor.
        """
        # Temporal padding if stride[0] == 2
        if self.stride[0] == 2:
            # Duplicate first frame for padding
            first_frame = x[:, :, :1, :, :]
            x = mx.concatenate([first_frame, x], axis=2)

        # Skip connection: direct space-to-depth with channel averaging
        x_in = space_to_depth(x, self.stride)

        # Group and average channels
        b, c_packed, d, h, w = x_in.shape
        g = self.group_size
        c_out = c_packed // g

        # Reshape: (B, C*g, D, H, W) -> (B, C, g, D, H, W)
        x_in = x_in.reshape(b, c_out, g, d, h, w)
        # Average over groups
        x_in = mx.mean(x_in, axis=2)

        # Convolutional path
        x = self.conv(x, causal=causal)
        x = space_to_depth(x, self.stride)

        # Residual addition
        return x + x_in


class DepthToSpaceUpsample(nn.Module):
    """
    Upsampling layer using depth-to-space with optional residual connection.

    Combines a convolutional path with a direct depth-to-space shortcut
    for better gradient flow.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        stride: Tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """
        Initialize DepthToSpaceUpsample.

        Args:
            dims: Dimension specification for convolutions.
            in_channels: Number of input channels.
            stride: Upsampling factors (temporal, height, width).
            residual: Whether to add a residual connection.
            out_channels_reduction_factor: Factor to reduce output channels.
            spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()

        self.stride = stride
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor

        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """
        Apply upsampling.

        Args:
            x: Input tensor of shape (B, C, D, H, W).
            causal: Whether to use causal convolution.

        Returns:
            Upsampled tensor.
        """
        if self.residual:
            # Direct depth-to-space for skip connection
            x_in = depth_to_space(x, self.stride)

            # Repeat channels to match output
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = mx.repeat(x_in, num_repeat, axis=1)

            # Remove first temporal frame if stride[0] == 2 (to undo padding)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]

        # Convolutional path
        x = self.conv(x, causal=causal)
        x = depth_to_space(x, self.stride)

        # Remove first temporal frame if stride[0] == 2
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]

        # Residual addition
        if self.residual:
            x = x + x_in

        return x
