"""Simplified Video VAE Decoder for inference with PyTorch weight loading."""

from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .ops import unpatchify


class Conv3dSimple(nn.Module):
    """
    3D convolution that stores PyTorch weights and decomposes to 2D+1D.

    Weight format: (out_channels, in_channels, T, H, W) - PyTorch format
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # PyTorch weight format: (out_C, in_C, T, H, W)
        k = kernel_size
        self.weight = mx.zeros((out_channels, in_channels, k, k, k))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """
        Apply 3D conv decomposed as 2D spatial + 1D temporal.

        Args:
            x: Input tensor (B, C, T, H, W)
            causal: Whether to use causal temporal padding

        Returns:
            Output tensor (B, C_out, T, H, W)
        """
        b, c, t, h, w = x.shape
        p = self.padding
        k = self.kernel_size

        # Spatial padding
        if p > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (p, p), (p, p)])

        # Temporal padding (causal: replicate first frame)
        if causal and p > 0:
            first_frames = mx.repeat(x[:, :, :1], p, axis=2)
            x = mx.concatenate([first_frames, x], axis=2)
        elif p > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (p, p), (0, 0), (0, 0)])

        # Extract center slice of 3D kernel for 2D spatial conv
        # Weight: (out_C, in_C, T, H, W) -> spatial slice: (out_C, in_C, H, W)
        w_spatial = self.weight[:, :, k // 2, :, :]  # (out_C, in_C, kH, kW)
        w_spatial = w_spatial.transpose(0, 2, 3, 1)  # MLX: (out_C, kH, kW, in_C)

        # Reshape for 2D conv: (B, C, T, H, W) -> (B*T, H, W, C)
        _, _, t_pad, h_pad, w_pad = x.shape
        x = x.transpose(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(b * t_pad, c, h_pad, w_pad)
        x = x.transpose(0, 2, 3, 1)  # (B*T, H, W, C)

        # Apply 2D spatial convolution
        x = mx.conv2d(x, w_spatial, padding=0)

        # Reshape back: (B*T, H_out, W_out, C_out) -> (B, C_out, T, H_out, W_out)
        _, h_out, w_out, c_out = x.shape
        x = x.reshape(b, t_pad, h_out, w_out, c_out)
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C_out, T, H_out, W_out)

        # Crop temporal if we padded
        if p > 0:
            x = x[:, :, p:p+t, :, :]

        # Add bias
        x = x + self.bias[None, :, None, None, None]

        return x


class ResBlock3d(nn.Module):
    """3D residual block with pixel norm and scale/shift conditioning."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = Conv3dSimple(channels, channels)
        self.conv2 = Conv3dSimple(channels, channels)
        self.scale_shift_table = mx.zeros((4, channels))

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Apply residual block with pixel norm."""
        residual = x

        # Pixel norm -> scale/shift -> conv1 -> silu
        x = _pixel_norm(x)
        # Apply scale/shift from table (simplified - no timestep conditioning)
        scale1 = 1 + self.scale_shift_table[0][None, :, None, None, None]
        shift1 = self.scale_shift_table[1][None, :, None, None, None]
        x = x * scale1 + shift1
        x = self.conv1(x, causal=causal)
        x = nn.silu(x)

        # Pixel norm -> scale/shift -> conv2
        x = _pixel_norm(x)
        scale2 = 1 + self.scale_shift_table[2][None, :, None, None, None]
        shift2 = self.scale_shift_table[3][None, :, None, None, None]
        x = x * scale2 + shift2
        x = self.conv2(x, causal=causal)

        return x + residual


class DepthToSpaceUpsample3d(nn.Module):
    """Upsample using depth-to-space (pixel shuffle) in 3D."""

    def __init__(self, in_channels: int, factor: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.factor = factor
        ft, fh, fw = factor
        out_channels = in_channels * ft * fh * fw
        self.conv = Conv3dSimple(in_channels, out_channels)

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Upsample via conv then depth-to-space."""
        x = self.conv(x, causal=causal)

        b, c, t, h, w = x.shape
        ft, fh, fw = self.factor
        c_out = c // (ft * fh * fw)

        # Depth to space: (B, C*ft*fh*fw, T, H, W) -> (B, C, T*ft, H*fh, W*fw)
        x = x.reshape(b, c_out, ft, fh, fw, t, h, w)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (B, C, T, ft, H, fh, W, fw)
        x = x.reshape(b, c_out, t * ft, h * fh, w * fw)

        return x


class ResBlockGroup(nn.Module):
    """Group of residual blocks with optional timestep embedding."""

    def __init__(self, channels: int, num_blocks: int = 5):
        super().__init__()
        self.res_blocks = [ResBlock3d(channels) for _ in range(num_blocks)]

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Apply all res blocks sequentially."""
        for block in self.res_blocks:
            x = block(x, causal=causal)
        return x


def _pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Apply pixel normalization (normalize across channels)."""
    variance = mx.mean(x * x, axis=1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


class SimpleVideoDecoder(nn.Module):
    """
    Simplified VAE decoder that matches PyTorch weight structure.

    Architecture:
    - conv_in: 128 -> 1024
    - up_blocks.0: 5 res blocks (1024 ch)
    - up_blocks.1: depth-to-space upsample (1024 -> 512 ch, 2x2x2 spatial/temporal)
    - up_blocks.2: 5 res blocks (512 ch)
    - up_blocks.3: depth-to-space upsample (512 -> 256 ch, 2x2x2)
    - up_blocks.4: 5 res blocks (256 ch)
    - up_blocks.5: depth-to-space upsample (256 -> 128 ch, 2x2x2)
    - up_blocks.6: 5 res blocks (128 ch)
    - conv_out: 128 -> 48 (for patch_size=4 unpatchify)

    Total upsampling: 8x temporal, 8x spatial from d2s + 4x spatial from unpatchify = 32x spatial
    """

    def __init__(self):
        super().__init__()

        # Per-channel statistics for denormalization
        self.mean_of_means = mx.zeros((128,))
        self.std_of_means = mx.zeros((128,))

        # Conv in: 128 -> 1024
        self.conv_in = Conv3dSimple(128, 1024)

        # Up blocks
        self.up_blocks_0 = ResBlockGroup(1024, num_blocks=5)
        self.up_blocks_1 = DepthToSpaceUpsample3d(1024, factor=(2, 2, 2))  # -> 512 ch
        self.up_blocks_2 = ResBlockGroup(512, num_blocks=5)
        self.up_blocks_3 = DepthToSpaceUpsample3d(512, factor=(2, 2, 2))   # -> 256 ch
        self.up_blocks_4 = ResBlockGroup(256, num_blocks=5)
        self.up_blocks_5 = DepthToSpaceUpsample3d(256, factor=(2, 2, 2))   # -> 128 ch
        self.up_blocks_6 = ResBlockGroup(128, num_blocks=5)

        # Conv out: 128 -> 48 (3 * 16 for patch_size=4 unpatchify)
        self.conv_out = Conv3dSimple(128, 48)

        # Scale/shift for final norm (not fully used in simplified version)
        self.last_scale_shift_table = mx.zeros((2, 128))

    def __call__(self, latent: mx.array, show_progress: bool = True) -> mx.array:
        """
        Decode latent to video.

        Args:
            latent: Latent tensor (B, 128, T, H, W).
            show_progress: Whether to show progress bar.

        Returns:
            Video tensor (B, 3, T*8, H*32, W*32).
        """
        try:
            from tqdm import tqdm
            has_tqdm = show_progress
        except ImportError:
            has_tqdm = False

        def step(x, block, desc):
            x = block(x, causal=True)
            mx.eval(x)
            if has_tqdm:
                pbar.update(1)
                pbar.set_description(desc)
            return x

        if has_tqdm:
            pbar = tqdm(total=10, desc="VAE decode", ncols=80)

        # Denormalize latent using per-channel statistics
        x = latent * self.std_of_means[None, :, None, None, None]
        x = x + self.mean_of_means[None, :, None, None, None]

        # Conv in
        x = self.conv_in(x, causal=True)
        mx.eval(x)
        if has_tqdm:
            pbar.update(1)
            pbar.set_description("conv_in done")

        # Up blocks with progress
        x = step(x, self.up_blocks_0, "res_blocks 1/4")
        x = step(x, self.up_blocks_1, "upsample 1/3")
        x = step(x, self.up_blocks_2, "res_blocks 2/4")
        x = step(x, self.up_blocks_3, "upsample 2/3")
        x = step(x, self.up_blocks_4, "res_blocks 3/4")
        x = step(x, self.up_blocks_5, "upsample 3/3")
        x = step(x, self.up_blocks_6, "res_blocks 4/4")

        # Final norm and activation
        x = _pixel_norm(x)
        scale = 1 + self.last_scale_shift_table[0][None, :, None, None, None]
        shift = self.last_scale_shift_table[1][None, :, None, None, None]
        x = x * scale + shift
        x = nn.silu(x)

        # Conv out
        x = self.conv_out(x, causal=True)
        mx.eval(x)
        if has_tqdm:
            pbar.update(1)
            pbar.set_description("conv_out done")

        # Unpatchify: (B, 48, T, H, W) -> (B, 3, T, H*4, W*4)
        x = unpatchify(x, patch_size_hw=4, patch_size_t=1)
        mx.eval(x)
        if has_tqdm:
            pbar.update(1)
            pbar.set_description("unpatchify done")
            pbar.close()

        return x


def load_vae_decoder_weights(decoder: SimpleVideoDecoder, weights_path: str) -> None:
    """
    Load VAE decoder weights from PyTorch safetensors file.

    Args:
        decoder: SimpleVideoDecoder instance to load weights into.
        weights_path: Path to safetensors file containing VAE weights.
    """
    from safetensors import safe_open
    import torch

    print(f"Loading VAE decoder weights from {weights_path}...")

    loaded_count = 0
    skipped_count = 0

    with safe_open(weights_path, framework="pt") as f:
        # Load per-channel statistics
        for stat_key in ["mean-of-means", "std-of-means"]:
            pt_key = f"vae.per_channel_statistics.{stat_key}"
            if pt_key in f.keys():
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                if stat_key == "mean-of-means":
                    decoder.mean_of_means = value
                else:
                    decoder.std_of_means = value
                loaded_count += 1

        # Load conv_in
        for suffix in ["weight", "bias"]:
            pt_key = f"vae.decoder.conv_in.conv.{suffix}"
            if pt_key in f.keys():
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                if suffix == "weight":
                    decoder.conv_in.weight = value
                else:
                    decoder.conv_in.bias = value
                loaded_count += 1

        # Load up_blocks
        block_mapping = [
            (0, "up_blocks_0", "res"),
            (1, "up_blocks_1", "upsample"),
            (2, "up_blocks_2", "res"),
            (3, "up_blocks_3", "upsample"),
            (4, "up_blocks_4", "res"),
            (5, "up_blocks_5", "upsample"),
            (6, "up_blocks_6", "res"),
        ]

        for pt_idx, mlx_name, block_type in block_mapping:
            block = getattr(decoder, mlx_name)

            if block_type == "res":
                # Load res blocks
                for res_idx in range(5):
                    res_block = block.res_blocks[res_idx]

                    # Load conv1 and conv2
                    for conv_name in ["conv1", "conv2"]:
                        conv = getattr(res_block, conv_name)
                        for suffix in ["weight", "bias"]:
                            pt_key = f"vae.decoder.up_blocks.{pt_idx}.res_blocks.{res_idx}.{conv_name}.conv.{suffix}"
                            if pt_key in f.keys():
                                tensor = f.get_tensor(pt_key)
                                if tensor.dtype == torch.bfloat16:
                                    tensor = tensor.to(torch.float32)
                                value = mx.array(tensor.numpy())

                                if suffix == "weight":
                                    conv.weight = value
                                else:
                                    conv.bias = value
                                loaded_count += 1

                    # Load scale_shift_table
                    pt_key = f"vae.decoder.up_blocks.{pt_idx}.res_blocks.{res_idx}.scale_shift_table"
                    if pt_key in f.keys():
                        tensor = f.get_tensor(pt_key)
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.to(torch.float32)
                        res_block.scale_shift_table = mx.array(tensor.numpy())
                        loaded_count += 1

            else:  # upsample
                # Load upsample conv
                for suffix in ["weight", "bias"]:
                    pt_key = f"vae.decoder.up_blocks.{pt_idx}.conv.conv.{suffix}"
                    if pt_key in f.keys():
                        tensor = f.get_tensor(pt_key)
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.to(torch.float32)
                        value = mx.array(tensor.numpy())

                        if suffix == "weight":
                            block.conv.weight = value
                        else:
                            block.conv.bias = value
                        loaded_count += 1

        # Load conv_out
        for suffix in ["weight", "bias"]:
            pt_key = f"vae.decoder.conv_out.conv.{suffix}"
            if pt_key in f.keys():
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                if suffix == "weight":
                    decoder.conv_out.weight = value
                else:
                    decoder.conv_out.bias = value
                loaded_count += 1

        # Load last_scale_shift_table
        pt_key = "vae.decoder.last_scale_shift_table"
        if pt_key in f.keys():
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            decoder.last_scale_shift_table = mx.array(tensor.numpy())
            loaded_count += 1

    print(f"  Loaded {loaded_count} weight tensors")


def decode_latent(latent: mx.array, decoder: SimpleVideoDecoder) -> mx.array:
    """
    Decode latent to video frames.

    Args:
        latent: Latent tensor (B, 128, T, H, W) or (128, T, H, W).
        decoder: Loaded SimpleVideoDecoder instance.

    Returns:
        Video frames as uint8 (T, H, W, 3) in [0, 255].
    """
    # Add batch dim if needed
    if latent.ndim == 4:
        latent = latent[None]

    # Decode
    video = decoder(latent)

    # Convert to uint8: assume output is in [-1, 1]
    video = mx.clip((video + 1) / 2, 0, 1) * 255
    video = video.astype(mx.uint8)

    # Rearrange: (B, C, T, H, W) -> (T, H, W, C)
    video = video[0]  # Remove batch
    video = video.transpose(1, 2, 3, 0)

    return video
