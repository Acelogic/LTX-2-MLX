"""Spatial 2x Upscaler for LTX-2 MLX.

This module implements the spatial upscaler that doubles the resolution
of video latents: (B, C, F, H, W) -> (B, C, F, H*2, W*2)
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def conv3d(
    x: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
    stride: int = 1,
    padding: int = 1,
) -> mx.array:
    """
    Apply 3D convolution using iterative 2D convolutions.

    Args:
        x: Input tensor (B, C, T, H, W)
        weight: Kernel tensor (out_C, in_C, kT, kH, kW)
        bias: Optional bias tensor (out_C,)
        stride: Stride (applied to all dimensions)
        padding: Padding (applied to all dimensions)

    Returns:
        Output tensor (B, out_C, T', H', W')
    """
    b, c_in, t, h, w = x.shape
    c_out, _, kt, kh, kw = weight.shape

    # Pad input
    if padding > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding), (padding, padding)])

    _, _, t_pad, h_pad, w_pad = x.shape

    # Output dimensions
    t_out = (t_pad - kt) // stride + 1
    h_out = (h_pad - kh) // stride + 1
    w_out = (w_pad - kw) // stride + 1

    # Initialize output
    out = mx.zeros((b, c_out, t_out, h_out, w_out))

    # Iterate over temporal kernel
    for ti in range(kt):
        # Extract 2D kernel slice
        w_2d = weight[:, :, ti, :, :]  # (out_C, in_C, kH, kW)

        # Process each temporal output position
        for to in range(t_out):
            t_in = to * stride + ti
            if t_in < t_pad:
                # Get spatial slice: (B, C_in, H_pad, W_pad)
                x_slice = x[:, :, t_in, :, :]

                # Transpose for MLX conv2d: (B, C, H, W) -> (B, H, W, C)
                x_slice = x_slice.transpose(0, 2, 3, 1)

                # MLX conv2d expects weight: (out_C, kH, kW, in_C)
                w_2d_mlx = w_2d.transpose(0, 2, 3, 1)

                # Apply 2D convolution
                conv_out = mx.conv2d(x_slice, w_2d_mlx, stride=stride, padding=0)

                # Transpose back: (B, H', W', out_C) -> (B, out_C, H', W')
                conv_out = conv_out.transpose(0, 3, 1, 2)

                # Accumulate
                out[:, :, to, :, :] = out[:, :, to, :, :] + conv_out

    # Add bias
    if bias is not None:
        out = out + bias[None, :, None, None, None]

    return out


class Conv3d(nn.Module):
    """3D Convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight: (out_C, in_C, kT, kH, kW)
        self.weight = mx.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return conv3d(x, self.weight, self.bias, self.stride, self.padding)


class GroupNorm(nn.Module):
    """Group Normalization with configurable groups."""

    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps

        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T, H, W)
        b, c, t, h, w = x.shape

        # Reshape for group norm: (B, G, C//G, T, H, W)
        x = x.reshape(b, self.num_groups, c // self.num_groups, t, h, w)

        # Compute mean and variance over (C//G, T, H, W)
        mean = x.mean(axis=(2, 3, 4, 5), keepdims=True)
        var = x.var(axis=(2, 3, 4, 5), keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back: (B, C, T, H, W)
        x = x.reshape(b, c, t, h, w)

        # Apply scale and shift
        x = x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return x


class ResBlock3d(nn.Module):
    """3D Residual block with GroupNorm and SiLU activation."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = GroupNorm(channels, num_groups)
        self.conv1 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(channels, num_groups)
        self.conv2 = Conv3d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)

        return x + residual


class SpatialPixelShuffle(nn.Module):
    """
    Spatial 2x upsampler using pixel shuffle.

    Uses Conv2d to expand channels (1024 -> 4096), then pixel shuffle
    to convert channels to spatial resolution.
    """

    def __init__(self, in_channels: int = 1024, scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.out_channels = in_channels * scale_factor * scale_factor  # 4096

        # Conv2d: (1024, 4096) - applied per-frame
        # Weight: (out_C, kH, kW, in_C) for MLX
        self.conv_weight = mx.zeros((self.out_channels, 3, 3, in_channels))
        self.conv_bias = mx.zeros((self.out_channels,))

        # Blur kernel for anti-aliasing (optional, applied after)
        self.blur_kernel = mx.zeros((1, 1, 5, 5))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Upsample spatial dimensions by 2x.

        Args:
            x: Input (B, C, T, H, W) with C=1024

        Returns:
            Output (B, C, T, H*2, W*2) with C=1024
        """
        b, c, t, h, w = x.shape

        # Process each frame independently with Conv2d
        frames = []
        for ti in range(t):
            # Get frame: (B, C, H, W)
            frame = x[:, :, ti, :, :]

            # Transpose for MLX conv2d: (B, C, H, W) -> (B, H, W, C)
            frame = frame.transpose(0, 2, 3, 1)

            # Apply conv2d with padding=1 for same output size
            frame = mx.conv2d(frame, self.conv_weight, stride=1, padding=1)

            # Add bias
            frame = frame + self.conv_bias[None, None, None, :]

            # Transpose back: (B, H, W, C') -> (B, C', H, W)
            frame = frame.transpose(0, 3, 1, 2)  # (B, 4096, H, W)

            # Pixel shuffle: (B, C*r*r, H, W) -> (B, C, H*r, W*r)
            frame = self._pixel_shuffle(frame)  # (B, 1024, H*2, W*2)

            frames.append(frame)

        # Stack frames: (B, C, T, H*2, W*2)
        out = mx.stack(frames, axis=2)

        return out

    def _pixel_shuffle(self, x: mx.array) -> mx.array:
        """
        Pixel shuffle operation.

        Args:
            x: Input (B, C*r*r, H, W)

        Returns:
            Output (B, C, H*r, W*r)
        """
        b, c, h, w = x.shape
        r = self.scale_factor
        c_out = c // (r * r)

        # Reshape: (B, C, r, r, H, W) and rearrange
        x = x.reshape(b, c_out, r, r, h, w)
        x = x.transpose(0, 1, 4, 2, 5, 3)  # (B, C_out, H, r, W, r)
        x = x.reshape(b, c_out, h * r, w * r)

        return x


class SpatialUpscaler(nn.Module):
    """
    2x Spatial Upscaler for video latents.

    Takes latent (B, 128, F, H, W) and outputs (B, 128, F, H*2, W*2).

    Architecture:
    - initial_conv: Conv3d (128 -> 1024)
    - initial_norm: GroupNorm
    - res_blocks: 4x ResBlock3d
    - upsampler: SpatialPixelShuffle (2x)
    - post_upsample_res_blocks: 4x ResBlock3d
    - final_conv: Conv3d (1024 -> 128)
    """

    def __init__(
        self,
        latent_channels: int = 128,
        hidden_channels: int = 1024,
        num_res_blocks: int = 4,
        num_groups: int = 32,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype

        # Initial projection
        self.initial_conv = Conv3d(latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.initial_norm = GroupNorm(hidden_channels, num_groups)

        # Pre-upsample res blocks
        self.res_blocks = [ResBlock3d(hidden_channels, num_groups) for _ in range(num_res_blocks)]

        # Spatial upsampler (2x)
        self.upsampler = SpatialPixelShuffle(hidden_channels, scale_factor=2)

        # Post-upsample res blocks
        self.post_upsample_res_blocks = [ResBlock3d(hidden_channels, num_groups) for _ in range(num_res_blocks)]

        # Final projection
        self.final_conv = Conv3d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Upscale latent 2x spatially.

        Args:
            x: Latent tensor (B, 128, F, H, W)

        Returns:
            Upscaled latent (B, 128, F, H*2, W*2)
        """
        # Cast to compute dtype
        if self.compute_dtype != mx.float32:
            x = x.astype(self.compute_dtype)

        # Initial projection
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)

        # Pre-upsample residual blocks
        for block in self.res_blocks:
            x = block(x)
            mx.eval(x)

        # Upsample 2x spatially
        x = self.upsampler(x)
        mx.eval(x)

        # Post-upsample residual blocks
        for block in self.post_upsample_res_blocks:
            x = block(x)
            mx.eval(x)

        # Final projection
        x = nn.silu(x)
        x = self.final_conv(x)

        # Cast back
        if self.compute_dtype != mx.float32:
            x = x.astype(mx.float32)

        return x


def load_spatial_upscaler_weights(upscaler: SpatialUpscaler, weights_path: str) -> None:
    """
    Load spatial upscaler weights from safetensors file.

    Args:
        upscaler: SpatialUpscaler instance
        weights_path: Path to safetensors file
    """
    from safetensors import safe_open
    import torch

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    print(f"Loading Spatial Upscaler weights from {weights_path}...")
    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())

        if has_tqdm:
            key_iter = tqdm(keys, desc="Loading upscaler", ncols=80)
        else:
            key_iter = keys

        for key in key_iter:
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            value = mx.array(tensor.numpy())

            # Map weights to model
            if key == "initial_conv.weight":
                upscaler.initial_conv.weight = value
            elif key == "initial_conv.bias":
                upscaler.initial_conv.bias = value
            elif key == "initial_norm.weight":
                upscaler.initial_norm.weight = value
            elif key == "initial_norm.bias":
                upscaler.initial_norm.bias = value
            elif key == "final_conv.weight":
                upscaler.final_conv.weight = value
            elif key == "final_conv.bias":
                upscaler.final_conv.bias = value
            elif key.startswith("res_blocks."):
                _load_res_block_weight(upscaler.res_blocks, key, value)
            elif key.startswith("post_upsample_res_blocks."):
                _load_res_block_weight(upscaler.post_upsample_res_blocks, key.replace("post_upsample_res_blocks.", ""), value)
            elif key.startswith("upsampler."):
                _load_upsampler_weight(upscaler.upsampler, key, value)
            else:
                continue

            loaded_count += 1

    print(f"  Loaded {loaded_count} weight tensors")


def _load_res_block_weight(blocks: list, key: str, value: mx.array) -> None:
    """Load weight into a res_block."""
    # Parse key like "res_blocks.0.conv1.weight" or "0.conv1.weight"
    parts = key.replace("res_blocks.", "").split(".")
    block_idx = int(parts[0])
    layer_name = parts[1]  # conv1, conv2, norm1, norm2
    param_name = parts[2]  # weight, bias

    if block_idx >= len(blocks):
        return

    block = blocks[block_idx]
    layer = getattr(block, layer_name, None)
    if layer is not None:
        setattr(layer, param_name, value)


def _load_upsampler_weight(upsampler: SpatialPixelShuffle, key: str, value: mx.array) -> None:
    """Load weight into the upsampler."""
    if key == "upsampler.conv.weight":
        # PyTorch: (out_C, in_C, kH, kW) -> MLX: (out_C, kH, kW, in_C)
        value = value.transpose(0, 2, 3, 1)
        upsampler.conv_weight = value
    elif key == "upsampler.conv.bias":
        upsampler.conv_bias = value
    elif key == "upsampler.blur_down.kernel":
        upsampler.blur_kernel = value
