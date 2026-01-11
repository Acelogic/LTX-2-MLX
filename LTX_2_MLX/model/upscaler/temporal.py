"""Temporal 2x Upscaler for LTX-2 MLX.

This module implements the temporal upscaler that doubles the framerate
of video latents: (B, C, F, H, W) -> (B, C, F*2, H, W)
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .spatial import Conv3d, GroupNorm, ResBlock3d, conv3d


class TemporalPixelShuffle(nn.Module):
    """
    Temporal 2x upsampler using pixel shuffle.

    Uses Conv3d to expand channels (512 -> 1024), then temporal pixel shuffle
    to convert channels to temporal resolution.
    """

    def __init__(self, in_channels: int = 512, scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.out_channels = in_channels * scale_factor  # 1024

        # Conv3d: (512, 1024) - 3D conv for temporal upsampling
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
        b, c, t, h, w = x.shape

        # Apply 3D conv with padding=1 for same spatial output
        x = conv3d(x, self.conv_weight, self.conv_bias, stride=1, padding=1)
        # x: (B, 1024, T, H, W)

        # Temporal pixel shuffle: (B, C*2, T, H, W) -> (B, C, T*2, H, W)
        x = self._temporal_pixel_shuffle(x)

        return x

    def _temporal_pixel_shuffle(self, x: mx.array) -> mx.array:
        """
        Temporal pixel shuffle operation.

        Args:
            x: Input (B, C*r, T, H, W)

        Returns:
            Output (B, C, T*r, H, W)
        """
        b, c, t, h, w = x.shape
        r = self.scale_factor
        c_out = c // r

        # Reshape: (B, r, C_out, T, H, W) and rearrange
        x = x.reshape(b, r, c_out, t, h, w)
        x = x.transpose(0, 2, 3, 1, 4, 5)  # (B, C_out, T, r, H, W)
        x = x.reshape(b, c_out, t * r, h, w)

        return x


class TemporalUpscaler(nn.Module):
    """
    2x Temporal Upscaler for video latents.

    Takes latent (B, 128, F, H, W) and outputs (B, 128, F*2, H, W).

    Architecture:
    - initial_conv: Conv3d (128 -> 512)
    - initial_norm: GroupNorm
    - res_blocks: 4x ResBlock3d
    - upsampler: TemporalPixelShuffle (2x)
    - post_upsample_res_blocks: 4x ResBlock3d
    - final_conv: Conv3d (512 -> 128)
    """

    def __init__(
        self,
        latent_channels: int = 128,
        hidden_channels: int = 512,
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

        # Temporal upsampler (2x)
        self.upsampler = TemporalPixelShuffle(hidden_channels, scale_factor=2)

        # Post-upsample res blocks
        self.post_upsample_res_blocks = [ResBlock3d(hidden_channels, num_groups) for _ in range(num_res_blocks)]

        # Final projection
        self.final_conv = Conv3d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Upscale latent 2x temporally.

        Args:
            x: Latent tensor (B, 128, F, H, W)

        Returns:
            Upscaled latent (B, 128, F*2, H, W)
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

        # Upsample 2x temporally
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


def _load_upsampler_weight(upsampler: TemporalPixelShuffle, key: str, value: mx.array) -> None:
    """Load weight into the temporal upsampler."""
    # Keys are: upsampler.0.weight, upsampler.0.bias (Sequential wrapper in PyTorch)
    if key == "upsampler.0.weight":
        upsampler.conv_weight = value
    elif key == "upsampler.0.bias":
        upsampler.conv_bias = value
