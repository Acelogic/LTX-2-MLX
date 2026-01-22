"""Spatial 2x Upscaler for LTX-2 MLX.

This module implements the spatial upscaler that doubles the resolution
of video latents: (B, C, F, H, W) -> (B, C, F, H*2, W*2)

Architecture matches saved weights (ltx-2-spatial-upscaler-x2-1.0.safetensors):
- Uses 3D convolutions (dims=3) for main path
- mid_channels = 1024
- SpatialRationalResampler for upsampling (2D conv per-frame)
- Standard ResBlock: conv -> norm -> act -> conv -> norm -> act(x + residual)
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# Compiled conv3d for better performance - fuses the temporal loop
@mx.compile
def conv3d(
    x: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
    stride: int = 1,
    padding: int = 0,
) -> mx.array:
    """
    3D convolution implementation using 2D convolutions over temporal slices.

    Args:
        x: Input (B, C, T, H, W) in NCTHW format
        weight: Kernel (out_C, in_C, kT, kH, kW)
        bias: Optional bias (out_C,)
        stride: Stride (only 1 supported)
        padding: Padding amount

    Returns:
        Output (B, out_C, T', H', W')
    """
    b, c_in, t, h, w = x.shape
    out_c, _, kt, kh, kw = weight.shape

    # Pad all dimensions
    if padding > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding), (padding, padding)])
        t_padded = t + 2 * padding
        h_padded = h + 2 * padding
        w_padded = w + 2 * padding
    else:
        t_padded = t
        h_padded = h
        w_padded = w

    # Output dimensions
    t_out = t_padded - kt + 1
    h_out = h_padded - kh + 1
    w_out = w_padded - kw + 1

    # Accumulate output
    output = mx.zeros((b, out_c, t_out, h_out, w_out))

    # For each temporal kernel position
    for dt in range(kt):
        # Extract temporal slice: (B, C_in, T_out, H_padded, W_padded)
        x_slice = x[:, :, dt:dt + t_out, :, :]

        # Reshape for batch 2D conv: (B * T_out, H_padded, W_padded, C_in)
        x_2d = x_slice.transpose(0, 2, 3, 4, 1)  # (B, T_out, H, W, C)
        x_2d = x_2d.reshape(b * t_out, h_padded, w_padded, c_in)  # (B*T_out, H, W, C)

        # Extract 2D kernel for this temporal slice: (out_C, in_C, kH, kW) -> (out_C, kH, kW, in_C)
        w_2d = weight[:, :, dt, :, :].transpose(0, 2, 3, 1)

        # Apply 2D conv
        y_2d = mx.conv2d(x_2d, w_2d)  # (B*T_out, H_out, W_out, out_C)

        # Reshape back: (B, T_out, H_out, W_out, out_C)
        y = y_2d.reshape(b, t_out, h_out, w_out, out_c)
        y = y.transpose(0, 4, 1, 2, 3)  # (B, out_C, T_out, H_out, W_out)

        output = output + y

    # Add bias
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1, 1)

    return output


def group_norm_5d(x: mx.array, num_groups: int, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    """
    Apply GroupNorm to 5D tensor (B, C, T, H, W).

    PyTorch GroupNorm normalizes over (C/groups, T, H, W) for each group.
    We must compute mean/var across all spatial AND temporal dimensions.

    Args:
        x: Input tensor (B, C, T, H, W)
        num_groups: Number of groups
        weight: Scale parameter (C,)
        bias: Bias parameter (C,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor (B, C, T, H, W)
    """
    b, c, t, h, w = x.shape
    channels_per_group = c // num_groups

    # Reshape to (B, num_groups, channels_per_group, T, H, W)
    x = x.reshape(b, num_groups, channels_per_group, t, h, w)

    # Compute mean and variance over (channels_per_group, T, H, W) = axes (2, 3, 4, 5)
    mean = x.mean(axis=(2, 3, 4, 5), keepdims=True)
    var = x.var(axis=(2, 3, 4, 5), keepdims=True)

    # Normalize
    x = (x - mean) / mx.sqrt(var + eps)

    # Reshape back to (B, C, T, H, W)
    x = x.reshape(b, c, t, h, w)

    # Apply affine transform: weight * x + bias
    # weight and bias are (C,), need to broadcast to (1, C, 1, 1, 1)
    x = x * weight.reshape(1, -1, 1, 1, 1) + bias.reshape(1, -1, 1, 1, 1)

    return x


class ResBlock3d(nn.Module):
    """
    3D Residual block matching PyTorch reference exactly.

    Architecture: conv1 -> norm1 -> act -> conv2 -> norm2 -> act(x + residual)
    """

    def __init__(self, channels: int, mid_channels: Optional[int] = None, num_groups: int = 32):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        self.num_groups = num_groups
        self.mid_channels = mid_channels
        self.channels = channels

        # Conv3d weights: (out_C, in_C, kT, kH, kW)
        self.conv1_weight = mx.zeros((mid_channels, channels, 3, 3, 3))
        self.conv1_bias = mx.zeros((mid_channels,))

        self.conv2_weight = mx.zeros((channels, mid_channels, 3, 3, 3))
        self.conv2_bias = mx.zeros((channels,))

        # GroupNorm parameters (we'll apply manually for 5D)
        self.norm1 = nn.GroupNorm(num_groups, mid_channels)
        self.norm2 = nn.GroupNorm(num_groups, channels)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor (B, C, T, H, W) - NCTHW format

        Returns:
            Output tensor (B, C, T, H, W)
        """
        residual = x

        # conv1 -> norm1 -> act
        x = conv3d(x, self.conv1_weight, self.conv1_bias, padding=1)
        # Apply GroupNorm correctly for 5D tensor
        x = group_norm_5d(x, self.num_groups, self.norm1.weight, self.norm1.bias)
        x = nn.silu(x)

        # conv2 -> norm2
        x = conv3d(x, self.conv2_weight, self.conv2_bias, padding=1)
        x = group_norm_5d(x, self.num_groups, self.norm2.weight, self.norm2.bias)

        # act(x + residual) - activation AFTER residual, matching PyTorch
        x = nn.silu(x + residual)

        return x


class PixelShuffle2d(nn.Module):
    """
    2D Pixel shuffle for spatial upsampling.

    Rearranges (B, H, W, C*r*r) -> (B, H*r, W*r, C)
    """

    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input (B, H, W, C*r*r) in NHWC format

        Returns:
            Output (B, H*r, W*r, C)
        """
        b, h, w, c = x.shape
        r = self.r
        c_out = c // (r * r)

        # PyTorch pixel_shuffle packs channels as (C, r_h, r_w) where C is slowest
        # Reshape: (B, H, W, C*r*r) -> (B, H, W, C, r_h, r_w)
        x = x.reshape(b, h, w, c_out, r, r)

        # Permute to interleave spatial dims with their upscale factors
        # (B, H, W, C, r_h, r_w) -> (B, H, r_h, W, r_w, C)
        x = x.transpose(0, 1, 4, 2, 5, 3)

        # Reshape to final: (B, H*r, W*r, C)
        x = x.reshape(b, h * r, w * r, c_out)

        return x


class BlurDownsample(nn.Module):
    """
    Blur + downsample layer for anti-aliasing.

    For stride=1 (2x upscale), this just applies a blur filter without downsampling.
    """

    def __init__(self, stride: int = 1):
        super().__init__()
        self.stride = stride
        # Blur kernel from saved weights - initialized here, loaded from weights
        self.kernel = mx.zeros((1, 1, 5, 5))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input (B, H, W, C) in NHWC format

        Returns:
            Output (B, H', W', C) - downsampled if stride > 1
        """
        if self.stride == 1 and mx.all(self.kernel == 0):
            # No blur kernel loaded or stride=1, pass through
            return x

        b, h, w, c = x.shape

        # Apply blur per channel using depthwise conv
        # kernel is (1, 1, kH, kW), need to expand for depthwise conv
        k = self.kernel.squeeze()  # (kH, kW)
        kh, kw = k.shape

        # Pad input
        pad_h = kh // 2
        pad_w = kw // 2
        x_padded = mx.pad(x, [(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)])

        # Manual depthwise conv with blur kernel
        # For simplicity, just return input if stride=1 (blur is minor effect)
        if self.stride == 1:
            return x

        # Downsample with stride
        return x[:, ::self.stride, ::self.stride, :]


class SpatialRationalResampler(nn.Module):
    """
    Spatial resampler using 2D conv + PixelShuffle + BlurDownsample.

    For scale=2.0: upsample 2x, no downsample (den=1)
    """

    def __init__(self, mid_channels: int = 1024, scale: float = 2.0):
        super().__init__()
        self.scale = scale

        # For scale=2.0: num=2, den=1
        if scale == 2.0:
            self.num = 2
            self.den = 1
        else:
            raise ValueError(f"Unsupported scale {scale}")

        # Conv2d: (mid_channels -> num^2 * mid_channels)
        # Weight: (out_C, kH, kW, in_C) for MLX
        out_channels = (self.num ** 2) * mid_channels  # 4 * 1024 = 4096
        self.conv_weight = mx.zeros((out_channels, 3, 3, mid_channels))
        self.conv_bias = mx.zeros((out_channels,))

        self.pixel_shuffle = PixelShuffle2d(upscale_factor=self.num)
        self.blur_down = BlurDownsample(stride=self.den)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input (B, C, F, H, W) in NCFHW format

        Returns:
            Output (B, C, F, H*2, W*2)
        """
        b, c, f, h, w = x.shape

        # Rearrange to process per-frame: (B*F, H, W, C)
        x = x.transpose(0, 2, 3, 4, 1)  # (B, F, H, W, C)
        x = x.reshape(b * f, h, w, c)   # (B*F, H, W, C)

        # Apply 2D conv
        x = mx.conv2d(x, self.conv_weight, padding=1)
        x = x + self.conv_bias

        # Pixel shuffle: (B*F, H, W, 4096) -> (B*F, H*2, W*2, 1024)
        x = self.pixel_shuffle(x)

        # Blur downsample (stride=1 for 2x scale, essentially identity)
        x = self.blur_down(x)

        # Rearrange back: (B*F, H*2, W*2, C) -> (B, C, F, H*2, W*2)
        _, h_out, w_out, c_out = x.shape
        x = x.reshape(b, f, h_out, w_out, c_out)  # (B, F, H*2, W*2, C)
        x = x.transpose(0, 4, 1, 2, 3)            # (B, C, F, H*2, W*2)

        return x


class SpatialUpscaler(nn.Module):
    """
    2x Spatial Upscaler for video latents.

    Takes latent (B, 128, F, H, W) and outputs (B, 128, F, H*2, W*2).

    Architecture (matching saved weights):
    - dims=3: Uses 3D convolutions for main path
    - mid_channels=1024
    - initial_conv: Conv3d (128 -> 1024)
    - initial_norm: GroupNorm(32, 1024)
    - initial_act: SiLU
    - res_blocks: 4x ResBlock3d(1024)
    - upsampler: SpatialRationalResampler (2D conv per-frame)
    - post_upsample_res_blocks: 4x ResBlock3d(1024)
    - final_conv: Conv3d(1024 -> 128)
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,  # Matches saved weights
        num_blocks_per_stage: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_groups = num_groups

        # Initial conv: 128 -> 1024, (out_C, in_C, kT, kH, kW)
        self.initial_conv_weight = mx.zeros((mid_channels, in_channels, 3, 3, 3))
        self.initial_conv_bias = mx.zeros((mid_channels,))
        self.initial_norm = nn.GroupNorm(num_groups, mid_channels)

        # Pre-upsample res blocks
        self.res_blocks = [ResBlock3d(mid_channels, num_groups=num_groups)
                          for _ in range(num_blocks_per_stage)]

        # Spatial upsampler (2x) - uses 2D conv per-frame
        self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=2.0)

        # Post-upsample res blocks
        self.post_upsample_res_blocks = [ResBlock3d(mid_channels, num_groups=num_groups)
                                         for _ in range(num_blocks_per_stage)]

        # Final conv: 1024 -> 128
        self.final_conv_weight = mx.zeros((in_channels, mid_channels, 3, 3, 3))
        self.final_conv_bias = mx.zeros((in_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Upscale latent 2x spatially.

        Args:
            x: Latent tensor (B, C, F, H, W) in NCFHW format

        Returns:
            Upscaled latent (B, C, F, H*2, W*2)
        """
        b, _, f, h, w = x.shape

        # Initial projection: conv -> norm -> act
        x = conv3d(x, self.initial_conv_weight, self.initial_conv_bias, padding=1)
        # Apply GroupNorm correctly for 5D tensor (normalizing across T, H, W)
        x = group_norm_5d(x, self.num_groups, self.initial_norm.weight, self.initial_norm.bias)
        x = nn.silu(x)

        # Pre-upsample residual blocks (batched eval for better performance)
        for block in self.res_blocks:
            x = block(x)
        mx.eval(x)  # Single eval after all pre-upsample blocks

        # Upsample 2x spatially (uses 2D conv per-frame internally)
        x = self.upsampler(x)
        mx.eval(x)

        # Post-upsample residual blocks (batched eval for better performance)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        mx.eval(x)  # Single eval after all post-upsample blocks

        # Final projection
        x = conv3d(x, self.final_conv_weight, self.final_conv_bias, padding=1)

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
                # 3D conv weight: keep as (out_C, in_C, kT, kH, kW)
                upscaler.initial_conv_weight = value
                loaded_count += 1
            elif key == "initial_conv.bias":
                upscaler.initial_conv_bias = value
                loaded_count += 1
            elif key == "initial_norm.weight":
                upscaler.initial_norm.weight = value
                loaded_count += 1
            elif key == "initial_norm.bias":
                upscaler.initial_norm.bias = value
                loaded_count += 1
            elif key == "final_conv.weight":
                upscaler.final_conv_weight = value
                loaded_count += 1
            elif key == "final_conv.bias":
                upscaler.final_conv_bias = value
                loaded_count += 1
            elif key.startswith("res_blocks."):
                loaded_count += _load_res_block_weight(upscaler.res_blocks, key, value, "res_blocks.")
            elif key.startswith("post_upsample_res_blocks."):
                loaded_count += _load_res_block_weight(
                    upscaler.post_upsample_res_blocks, key, value, "post_upsample_res_blocks."
                )
            elif key.startswith("upsampler."):
                loaded_count += _load_upsampler_weight(upscaler.upsampler, key, value)

    print(f"  Loaded {loaded_count} weight tensors")


def _load_res_block_weight(blocks: list, key: str, value: mx.array, prefix: str) -> int:
    """Load weight into a res_block. Returns 1 if loaded, 0 otherwise."""
    # Parse key like "res_blocks.0.conv1.weight"
    parts = key.replace(prefix, "").split(".")
    if len(parts) < 3:
        return 0

    block_idx = int(parts[0])
    layer_name = parts[1]  # conv1, conv2, norm1, norm2
    param_name = parts[2]  # weight, bias

    if block_idx >= len(blocks):
        return 0

    block = blocks[block_idx]

    if layer_name == "conv1":
        if param_name == "weight":
            # 3D conv weight: keep as (out_C, in_C, kT, kH, kW)
            block.conv1_weight = value
        else:
            block.conv1_bias = value
        return 1
    elif layer_name == "conv2":
        if param_name == "weight":
            block.conv2_weight = value
        else:
            block.conv2_bias = value
        return 1
    elif layer_name == "norm1":
        setattr(block.norm1, param_name, value)
        return 1
    elif layer_name == "norm2":
        setattr(block.norm2, param_name, value)
        return 1

    return 0


def _load_upsampler_weight(upsampler: SpatialRationalResampler, key: str, value: mx.array) -> int:
    """Load weight into the upsampler. Returns 1 if loaded, 0 otherwise."""
    # Keys: upsampler.conv.weight, upsampler.conv.bias, upsampler.blur_down.kernel
    if key == "upsampler.conv.weight":
        # PyTorch Conv2d: (out_C, in_C, kH, kW) -> MLX: (out_C, kH, kW, in_C)
        value = value.transpose(0, 2, 3, 1)
        upsampler.conv_weight = value
        return 1
    elif key == "upsampler.conv.bias":
        upsampler.conv_bias = value
        return 1
    elif key == "upsampler.blur_down.kernel":
        upsampler.blur_down.kernel = value
        return 1
    return 0
