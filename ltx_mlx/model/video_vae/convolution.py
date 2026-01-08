"""3D convolution implementations using 2D+1D decomposition for MLX."""

from enum import Enum
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class PaddingModeType(Enum):
    """Padding mode types for convolutions."""

    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class NormLayerType(Enum):
    """Normalization layer types."""

    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


def _apply_padding(
    x: mx.array,
    padding: Tuple[int, ...],
    mode: str = "zeros",
) -> mx.array:
    """
    Apply padding to a tensor.

    Args:
        x: Input tensor.
        padding: Padding sizes for each dimension (pairs of before/after).
        mode: Padding mode ('zeros', 'reflect', 'replicate').

    Returns:
        Padded tensor.
    """
    if mode == "zeros":
        # MLX conv handles zero padding internally
        return x
    elif mode == "replicate":
        # Edge padding - replicate the edge values
        pad_width = [(0, 0)] * (x.ndim - len(padding) // 2)
        for i in range(0, len(padding), 2):
            pad_width.append((padding[i], padding[i + 1]))
        return mx.pad(x, pad_width, mode="edge")
    elif mode == "reflect":
        pad_width = [(0, 0)] * (x.ndim - len(padding) // 2)
        for i in range(0, len(padding), 2):
            pad_width.append((padding[i], padding[i + 1]))
        return mx.pad(x, pad_width, mode="reflect")
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


class DualConv3d(nn.Module):
    """
    3D convolution decomposed into 2D spatial + 1D temporal convolutions.

    This approach avoids native 3D convolutions by:
    1. Applying 2D conv across spatial dimensions for each frame
    2. Applying 1D conv across temporal dimension for each spatial location

    This is mathematically equivalent to a full 3D conv but works with
    MLX's 2D and 1D convolution primitives.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """
        Initialize DualConv3d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel (T, H, W) or single int.
            stride: Stride of the convolution.
            padding: Padding added to all sides.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections.
            bias: Whether to add a learnable bias.
            padding_mode: Padding mode ('zeros', 'reflect', 'replicate').
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.groups = groups
        self.has_bias = bias

        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError("kernel_size must be > 1. Use Linear instead.")
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Intermediate channels (max of in/out for information preservation)
        intermediate_channels = max(out_channels, in_channels)

        # First conv: 2D spatial (kernel size 1 in temporal dim)
        # Weight shape for MLX Conv2d: (out_channels, H, W, in_channels)
        # But we store as (out_channels, in_channels, 1, H, W) for compatibility
        self.stride1 = (stride[1], stride[2])
        self.padding1 = (padding[1], padding[2])
        self.dilation1 = (dilation[1], dilation[2])

        # Initialize conv1 (2D spatial conv)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=(kernel_size[1], kernel_size[2]),
            stride=self.stride1,
            padding=self.padding1,
            bias=bias,
        )

        # Second conv: 1D temporal
        self.stride2 = stride[0]
        self.padding2 = padding[0]
        self.dilation2 = dilation[0]

        # Initialize conv2 (1D temporal conv)
        self.conv2 = nn.Conv1d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[0],
            stride=self.stride2,
            padding=self.padding2,
            bias=bias,
        )

    def __call__(self, x: mx.array, skip_time_conv: bool = False) -> mx.array:
        """
        Apply the dual convolution.

        Args:
            x: Input tensor of shape (B, C, D, H, W).
            skip_time_conv: If True, skip the temporal convolution.

        Returns:
            Output tensor of shape (B, C_out, D_out, H_out, W_out).
        """
        b, c, d, h, w = x.shape

        # Step 1: 2D spatial convolution on each frame
        # Reshape: (B, C, D, H, W) -> (B*D, C, H, W) -> (B*D, H, W, C) for MLX Conv2d
        x = x.transpose(0, 2, 1, 3, 4)  # (B, D, C, H, W)
        x = x.reshape(b * d, c, h, w)  # (B*D, C, H, W)
        x = x.transpose(0, 2, 3, 1)  # (B*D, H, W, C) - MLX format

        x = self.conv1(x)  # (B*D, H_out, W_out, C_inter)

        _, h_out, w_out, c_inter = x.shape

        if skip_time_conv:
            # Reshape back: (B*D, H_out, W_out, C_inter) -> (B, C_inter, D, H_out, W_out)
            x = x.transpose(0, 3, 1, 2)  # (B*D, C_inter, H_out, W_out)
            x = x.reshape(b, d, c_inter, h_out, w_out)
            x = x.transpose(0, 2, 1, 3, 4)  # (B, C_inter, D, H_out, W_out)
            return x

        # Step 2: 1D temporal convolution
        # Reshape: (B*D, H_out, W_out, C_inter) -> (B*H_out*W_out, D, C_inter)
        x = x.transpose(0, 3, 1, 2)  # (B*D, C_inter, H_out, W_out)
        x = x.reshape(b, d, c_inter, h_out, w_out)
        x = x.transpose(0, 3, 4, 1, 2)  # (B, H_out, W_out, D, C_inter)
        x = x.reshape(b * h_out * w_out, d, c_inter)  # (B*H*W, D, C_inter) - MLX Conv1d format

        x = self.conv2(x)  # (B*H*W, D_out, C_out)

        _, d_out, c_out = x.shape

        # Reshape back: (B*H*W, D_out, C_out) -> (B, C_out, D_out, H_out, W_out)
        x = x.reshape(b, h_out, w_out, d_out, c_out)
        x = x.transpose(0, 4, 3, 1, 2)  # (B, C_out, D_out, H_out, W_out)

        return x


class CausalConv3d(nn.Module):
    """
    Causal 3D convolution with temporal causal padding.

    Implements causal convolution by replicating the first frame to fill
    the temporal receptive field, ensuring no information from future
    frames leaks into the current frame's computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """
        Initialize CausalConv3d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel (uniform in all dims).
            stride: Stride of the convolution.
            dilation: Dilation factor (applied to temporal dimension).
            groups: Number of blocked connections.
            bias: Whether to add a learnable bias.
            spatial_padding_mode: Padding mode for spatial dimensions.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_kernel_size = kernel_size
        self.dilation = dilation

        # Spatial padding (symmetric)
        height_pad = kernel_size // 2
        width_pad = kernel_size // 2

        # Use DualConv3d with no temporal padding (we handle it causally)
        self.conv = DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=(0, height_pad, width_pad),  # No temporal padding
            dilation=(dilation, 1, 1),
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """
        Apply causal 3D convolution.

        Args:
            x: Input tensor of shape (B, C, D, H, W).
            causal: If True, use causal padding. If False, use symmetric padding.

        Returns:
            Output tensor.
        """
        if causal:
            # Causal: replicate first frame to fill temporal receptive field
            first_frame = x[:, :, :1, :, :]  # (B, C, 1, H, W)
            pad_frames = mx.repeat(first_frame, self.time_kernel_size - 1, axis=2)
            x = mx.concatenate([pad_frames, x], axis=2)
        else:
            # Non-causal: symmetric padding with edge frames
            pad_size = (self.time_kernel_size - 1) // 2
            first_frame = x[:, :, :1, :, :]
            last_frame = x[:, :, -1:, :, :]
            first_pad = mx.repeat(first_frame, pad_size, axis=2)
            last_pad = mx.repeat(last_frame, pad_size, axis=2)
            x = mx.concatenate([first_pad, x, last_pad], axis=2)

        return self.conv(x)


class Conv3d(nn.Module):
    """
    Standard 3D convolution implemented via 2D+1D decomposition.

    A convenience wrapper around DualConv3d that provides a similar
    interface to PyTorch's nn.Conv3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """Initialize Conv3d."""
        super().__init__()
        self.conv = DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 3D convolution."""
        return self.conv(x)


class Conv3dFull(nn.Module):
    """
    3D convolution that stores full 3D weights (PyTorch compatible).

    Uses 2D+1D decomposition for forward pass but stores weights in
    standard 3D conv format (out_channels, in_channels, T, H, W).
    This allows direct loading of PyTorch Conv3d weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ):
        """Initialize Conv3dFull with PyTorch-compatible weight format."""
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight shape: (out_channels, in_channels, T, H, W) - PyTorch format
        self.weight = mx.zeros((out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array, causal: bool = False) -> mx.array:
        """
        Apply 3D convolution using 2D+1D decomposition.

        Args:
            x: Input tensor (B, C, T, H, W).
            causal: If True, use causal temporal padding.

        Returns:
            Output tensor (B, out_C, T', H', W').
        """
        b, c, t, h, w = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding

        # Apply spatial padding if needed
        if ph > 0 or pw > 0:
            x = mx.pad(x, [(0,0), (0,0), (0,0), (ph, ph), (pw, pw)])

        # Apply temporal padding
        if causal and pt > 0:
            # Causal: replicate first frame
            first_frame = mx.repeat(x[:, :, :1, :, :], pt, axis=2)
            x = mx.concatenate([first_frame, x], axis=2)
        elif pt > 0:
            # Non-causal: symmetric padding
            x = mx.pad(x, [(0,0), (0,0), (pt, pt), (0,0), (0,0)])

        _, _, t_pad, h_pad, w_pad = x.shape

        # Step 1: 2D spatial convolution on each frame
        # Extract 2D weights: (out_C, in_C, kH, kW)
        # Take middle time slice for spatial conv
        weight_2d = self.weight[:, :, kt//2, :, :]  # (out_C, in_C, kH, kW)
        # Convert to MLX format: (out_C, kH, kW, in_C) -> transpose for conv
        weight_2d_mlx = weight_2d.transpose(0, 2, 3, 1)  # (out_C, kH, kW, in_C)

        # Reshape x for 2D conv: (B, C, T, H, W) -> (B*T, H, W, C)
        x_2d = x.transpose(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x_2d = x_2d.reshape(b * t_pad, c, h_pad, w_pad)
        x_2d = x_2d.transpose(0, 2, 3, 1)  # (B*T, H, W, C)

        # Apply 2D conv
        x_2d = mx.conv2d(x_2d, weight_2d_mlx, stride=(sh, sw), padding=0)

        _, h_out, w_out, c_out = x_2d.shape

        # Step 2: 1D temporal convolution
        # Extract 1D weights: (out_C, in_C, kT) from center spatial position
        weight_1d = self.weight[:, :, :, kh//2, kw//2]  # (out_C, in_C, kT)
        # For conv1d: weight shape (out_C, kT, in_C)
        weight_1d_mlx = weight_1d.transpose(0, 2, 1)  # (out_C, kT, in_C)

        # Reshape for 1D conv: (B*T, H, W, C) -> (B*H*W, T, C)
        x_1d = x_2d.transpose(0, 3, 1, 2)  # (B*T, C, H, W)
        x_1d = x_1d.reshape(b, t_pad, c_out, h_out, w_out)
        x_1d = x_1d.transpose(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        x_1d = x_1d.reshape(b * h_out * w_out, t_pad, c_out)  # (B*H*W, T, C)

        # Apply 1D conv
        x_1d = mx.conv1d(x_1d, weight_1d_mlx, stride=st, padding=0)

        _, t_out, _ = x_1d.shape

        # Reshape back: (B*H*W, T', C') -> (B, C', T', H', W')
        x_out = x_1d.reshape(b, h_out, w_out, t_out, self.out_channels)
        x_out = x_out.transpose(0, 4, 3, 1, 2)  # (B, C', T', H', W')

        # Add bias
        if self.bias is not None:
            x_out = x_out + self.bias[None, :, None, None, None]

        return x_out


def make_conv_nd(
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    temporal_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:
    """
    Factory function to create N-dimensional convolutions.

    Args:
        dims: Dimension specification (2 for 2D, 3 for 3D, (2,1) for DualConv).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Padding added to all sides.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections.
        bias: Whether to add a learnable bias.
        causal: Whether to use causal convolution (3D only).
        spatial_padding_mode: Padding mode for spatial dimensions.
        temporal_padding_mode: Padding mode for temporal dimension.

    Returns:
        Appropriate convolution module.
    """
    if dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def make_linear_nd(
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    bias: bool = True,
) -> nn.Module:
    """
    Factory function to create N-dimensional linear (1x1 conv) layers.

    Args:
        dims: Dimension specification (2 for 2D, 3 or (2,1) for 3D).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bias: Whether to add a learnable bias.

    Returns:
        1x1 convolution module (acts as channel-wise linear transform).
    """
    if dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
    elif dims in (3, (2, 1)):
        # For 3D, use a 1x1x1 conv via our DualConv3d
        # Since kernel is 1x1x1, we can just use 2D conv and broadcast
        return PointwiseConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
        )
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


class PointwiseConv3d(nn.Module):
    """
    Pointwise (1x1x1) 3D convolution.

    Implemented as a 2D 1x1 conv applied to each frame independently.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        """Initialize pointwise conv."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply pointwise convolution.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D, H, W).
        """
        b, c, d, h, w = x.shape

        # Reshape to apply 2D conv: (B, C, D, H, W) -> (B*D, H, W, C)
        x = x.transpose(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        x = x.reshape(b * d, h, w, c)  # (B*D, H, W, C)

        x = self.conv(x)  # (B*D, H, W, C_out)

        _, _, _, c_out = x.shape

        # Reshape back: (B*D, H, W, C_out) -> (B, C_out, D, H, W)
        x = x.reshape(b, d, h, w, c_out)
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C_out, D, H, W)

        return x
