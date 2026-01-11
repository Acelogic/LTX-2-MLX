"""Temporal 2x Upscaler for LTX-2 MLX.

This module implements the temporal upscaler that doubles the framerate
of video latents: (B, C, F, H, W) -> (B, C, F*2, H, W)

Architecture matches PyTorch reference:
- Uses 3D convolutions (temporal + spatial)
- mid_channels = 512
- Standard ResBlock: conv -> norm -> act -> conv -> norm -> act(x + residual)
- PixelShuffle for temporal upsampling (dims=1)
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

    This is a workaround since MLX doesn't have native 3D convolution.
    Processes each temporal slice with 2D convolutions and combines results.

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
    out_c, in_c, kt, kh, kw = weight.shape

    # Pad temporal dimension
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


class ResBlock3d(nn.Module):
    """
    3D Residual block matching PyTorch reference exactly.

    Architecture: conv1 -> norm1 -> act -> conv2 -> norm2 -> act(x + residual)
    """

    def __init__(self, channels: int, mid_channels: Optional[int] = None, num_groups: int = 32):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        # Conv3d weights: (out_C, in_C, kT, kH, kW)
        self.conv1_weight = mx.zeros((mid_channels, channels, 3, 3, 3))
        self.conv1_bias = mx.zeros((mid_channels,))

        self.conv2_weight = mx.zeros((channels, mid_channels, 3, 3, 3))
        self.conv2_bias = mx.zeros((channels,))

        # GroupNorm
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
        b, c, t, h, w = x.shape

        # conv1 -> norm1 -> act
        x = conv3d(x, self.conv1_weight, self.conv1_bias, padding=1)
        # GroupNorm expects (B, ..., C) - need to transpose
        x = x.transpose(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = x.reshape(b * t, h, w, -1)  # (B*T, H, W, C)
        x = self.norm1(x)
        x = x.reshape(b, t, h, w, -1)  # (B, T, H, W, C)
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = nn.silu(x)

        # conv2 -> norm2
        x = conv3d(x, self.conv2_weight, self.conv2_bias, padding=1)
        x = x.transpose(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = x.reshape(b * t, h, w, -1)  # (B*T, H, W, C)
        x = self.norm2(x)
        x = x.reshape(b, t, h, w, -1)  # (B, T, H, W, C)
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C, T, H, W)

        # act(x + residual) - activation AFTER residual, matching PyTorch
        x = nn.silu(x + residual)

        return x


class TemporalPixelShuffle(nn.Module):
    """
    Temporal pixel shuffle for 2x temporal upsampling.

    Uses Conv3d to expand channels (512 -> 1024), then temporal pixel shuffle
    to convert channels to temporal resolution.

    Matches PyTorch: PixelShuffleND(dims=1)
    """

    def __init__(self, in_channels: int = 512, scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.out_channels = in_channels * scale_factor  # 1024

        # Conv3d: (512 -> 1024)
        # Weight: (out_C, in_C, kT, kH, kW)
        self.conv_weight = mx.zeros((self.out_channels, in_channels, 3, 3, 3))
        self.conv_bias = mx.zeros((self.out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Upsample temporal dimension by 2x.

        Args:
            x: Input (B, C, T, H, W) with C=512

        Returns:
            Output (B, C, T*2, H, W) with C=512
        """
        # Apply 3D conv with padding=1 for same spatial output
        x = conv3d(x, self.conv_weight, self.conv_bias, padding=1)
        # x: (B, 1024, T, H, W)

        # Temporal pixel shuffle: (B, C*2, T, H, W) -> (B, C, T*2, H, W)
        x = self._temporal_pixel_shuffle(x)

        return x

    def _temporal_pixel_shuffle(self, x: mx.array) -> mx.array:
        """
        Temporal pixel shuffle operation (dims=1).

        Matches PyTorch: rearrange "b (c p1) f h w -> b c (f p1) h w"

        Args:
            x: Input (B, C*r, T, H, W)

        Returns:
            Output (B, C, T*r, H, W)
        """
        b, c, t, h, w = x.shape
        r = self.scale_factor
        c_out = c // r

        # Reshape: (B, r, C_out, T, H, W) - split channels into r and C_out
        x = x.reshape(b, r, c_out, t, h, w)
        # Permute: (B, C_out, T, r, H, W) - move r next to T
        x = x.transpose(0, 2, 3, 1, 4, 5)
        # Reshape: (B, C_out, T*r, H, W) - merge T and r
        x = x.reshape(b, c_out, t * r, h, w)

        return x


class TemporalUpscaler(nn.Module):
    """
    2x Temporal Upscaler for video latents.

    Takes latent (B, 128, F, H, W) and outputs (B, 128, F*2, H, W).

    Architecture (matching PyTorch reference):
    - initial_conv: Conv3d (128 -> 512)
    - initial_norm: GroupNorm(32, 512)
    - initial_act: SiLU
    - res_blocks: 4x ResBlock3d(512)
    - upsampler: Conv3d(512 -> 1024) + TemporalPixelShuffle(2)
    - post_upsample_res_blocks: 4x ResBlock3d(512)
    - final_conv: Conv3d(512 -> 128)
    """

    def __init__(
        self,
        latent_channels: int = 128,
        hidden_channels: int = 512,
        num_res_blocks: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels

        # Initial conv: 128 -> 512, (out_C, in_C, kT, kH, kW)
        self.initial_conv_weight = mx.zeros((hidden_channels, latent_channels, 3, 3, 3))
        self.initial_conv_bias = mx.zeros((hidden_channels,))
        self.initial_norm = nn.GroupNorm(num_groups, hidden_channels)

        # Pre-upsample res blocks
        self.res_blocks = [ResBlock3d(hidden_channels, num_groups=num_groups)
                          for _ in range(num_res_blocks)]

        # Temporal upsampler (2x)
        self.upsampler = TemporalPixelShuffle(hidden_channels, scale_factor=2)

        # Post-upsample res blocks
        self.post_upsample_res_blocks = [ResBlock3d(hidden_channels, num_groups=num_groups)
                                         for _ in range(num_res_blocks)]

        # Final conv: 512 -> 128
        self.final_conv_weight = mx.zeros((latent_channels, hidden_channels, 3, 3, 3))
        self.final_conv_bias = mx.zeros((latent_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Upscale latent 2x temporally.

        Args:
            x: Latent tensor (B, C, F, H, W) in NCFHW format

        Returns:
            Upscaled latent (B, C, F*2, H, W)
        """
        b, c, f, h, w = x.shape

        # Initial projection: conv -> norm -> act
        x = conv3d(x, self.initial_conv_weight, self.initial_conv_bias, padding=1)
        # GroupNorm expects NHWC, need to transpose for per-frame norm
        x = x.transpose(0, 2, 3, 4, 1)  # (B, F, H, W, C)
        x = x.reshape(b * f, h, w, -1)  # (B*F, H, W, C)
        x = self.initial_norm(x)
        x = x.reshape(b, f, h, w, -1)  # (B, F, H, W, C)
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C, F, H, W)
        x = nn.silu(x)

        # Pre-upsample residual blocks (batched eval for better performance)
        for block in self.res_blocks:
            x = block(x)
        mx.eval(x)  # Single eval after all pre-upsample blocks

        # Upsample 2x temporally
        x = self.upsampler(x)
        # Remove first frame after upsampling (per PyTorch reference)
        # First frame encodes one pixel frame
        x = x[:, :, 1:, :, :]
        mx.eval(x)

        # Post-upsample residual blocks (batched eval for better performance)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        mx.eval(x)  # Single eval after all post-upsample blocks

        # Final projection
        x = conv3d(x, self.final_conv_weight, self.final_conv_bias, padding=1)

        return x


def load_temporal_upscaler_weights(upscaler: TemporalUpscaler, weights_path: str) -> None:
    """
    Load temporal upscaler weights from safetensors file.

    Args:
        upscaler: TemporalUpscaler instance
        weights_path: Path to safetensors file
    """
    from safetensors import safe_open
    import torch

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    print(f"Loading Temporal Upscaler weights from {weights_path}...")
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
                # upsampler.0.weight, upsampler.0.bias (Conv3d in Sequential)
                if "weight" in key:
                    upscaler.upsampler.conv_weight = value
                    loaded_count += 1
                elif "bias" in key:
                    upscaler.upsampler.conv_bias = value
                    loaded_count += 1

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
