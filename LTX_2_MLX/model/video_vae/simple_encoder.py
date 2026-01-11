"""Simplified Video VAE Encoder for inference with PyTorch weight loading."""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .ops import patchify, PerChannelStatistics


def _pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Apply pixel normalization (normalize across channels)."""
    variance = mx.mean(x * x, axis=1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


class Conv3dSimple(nn.Module):
    """
    3D convolution implemented via multiple 2D convolutions over temporal slices.

    This properly applies the full 3D kernel by iterating over temporal kernel
    positions and accumulating contributions from each 2D spatial slice.

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
        Apply full 3D convolution by iterating over temporal kernel positions.

        Args:
            x: Input tensor (B, C, T, H, W)
            causal: Whether to use causal temporal padding

        Returns:
            Output tensor (B, C_out, T, H, W)
        """
        b, c, _, _, _ = x.shape
        p = self.padding
        k = self.kernel_size

        # Spatial padding (p on each side)
        if p > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (p, p), (p, p)])

        # Temporal padding: need k-1 total padding to preserve temporal dim
        t_pad_needed = k - 1
        if causal and t_pad_needed > 0:
            # Causal: all padding at the beginning (replicate first frame)
            first_frames = mx.repeat(x[:, :, :1], t_pad_needed, axis=2)
            x = mx.concatenate([first_frames, x], axis=2)
        elif t_pad_needed > 0:
            # Non-causal: symmetric padding
            pad_before = t_pad_needed // 2
            pad_after = t_pad_needed - pad_before
            x = mx.pad(x, [(0, 0), (0, 0), (pad_before, pad_after), (0, 0), (0, 0)])

        _, _, t_pad, h_pad, w_pad = x.shape

        # Output dimensions after spatial conv (no padding in conv2d since we pre-padded)
        h_out = h_pad - k + 1
        w_out = w_pad - k + 1

        # Initialize output accumulator
        output = None

        # Iterate over temporal kernel positions
        for kt in range(k):
            # Extract the 2D kernel slice for this temporal position
            # weight shape: (out_C, in_C, kT, kH, kW)
            w_slice = self.weight[:, :, kt, :, :]  # (out_C, in_C, kH, kW)
            w_slice = w_slice.transpose(0, 2, 3, 1)  # MLX format: (out_C, kH, kW, in_C)

            # Get the temporal slice of input that corresponds to this kernel position
            t_out_len = t_pad - k + 1  # Number of output temporal positions
            x_slice = x[:, :, kt:kt + t_out_len, :, :]  # (B, C, T_out, H_pad, W_pad)

            # Reshape for 2D conv: (B, C, T_out, H, W) -> (B*T_out, H, W, C)
            x_2d = x_slice.transpose(0, 2, 1, 3, 4)  # (B, T_out, C, H, W)
            x_2d = x_2d.reshape(b * t_out_len, c, h_pad, w_pad)
            x_2d = x_2d.transpose(0, 2, 3, 1)  # (B*T_out, H, W, C)

            # Apply 2D spatial convolution
            conv_out = mx.conv2d(x_2d, w_slice, padding=0)  # (B*T_out, H_out, W_out, C_out)

            # Reshape back: (B*T_out, H_out, W_out, C_out) -> (B, C_out, T_out, H_out, W_out)
            _, _, _, c_out = conv_out.shape
            conv_out = conv_out.reshape(b, t_out_len, h_out, w_out, c_out)
            conv_out = conv_out.transpose(0, 4, 1, 2, 3)  # (B, C_out, T_out, H_out, W_out)

            # Accumulate
            if output is None:
                output = conv_out
            else:
                output = output + conv_out

        # Add bias
        output = output + self.bias[None, :, None, None, None]

        return output


class EncoderResBlock3d(nn.Module):
    """
    3D residual block for encoder with pixel norm.

    Simpler than decoder ResBlock3d - no scale/shift conditioning.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = Conv3dSimple(channels, channels)
        self.conv2 = Conv3dSimple(channels, channels)

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """
        Apply residual block with pixel norm.

        Args:
            x: Input tensor (B, C, T, H, W)
            causal: Whether to use causal padding

        Returns:
            Output tensor (B, C, T, H, W)
        """
        residual = x

        # Block 1: norm -> activation -> conv1
        x = _pixel_norm(x)
        x = nn.silu(x)
        x = self.conv1(x, causal=causal)

        # Block 2: norm -> activation -> conv2
        x = _pixel_norm(x)
        x = nn.silu(x)
        x = self.conv2(x, causal=causal)

        return x + residual


class EncoderResBlockGroup(nn.Module):
    """Group of residual blocks for encoder."""

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.channels = channels
        self.res_blocks = [EncoderResBlock3d(channels) for _ in range(num_blocks)]

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Apply all res blocks sequentially."""
        for block in self.res_blocks:
            x = block(x, causal=causal)
        return x


class SpaceToDepthDownsample3d(nn.Module):
    """
    Downsample using space-to-depth with residual connection.

    This block increases channel count while downsampling spatially/temporally.
    For stride (2,2,2): in_channels -> out_channels, with 2x downsampling in T,H,W.

    Matches SpaceToDepthDownsample from PyTorch LTX-2:
    - Conv before space-to-depth rearrangement
    - Residual path: space-to-depth on input, then mean across grouped channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        stride_product = stride[0] * stride[1] * stride[2]

        # group_size for residual path mean computation
        # After space-to-depth: in_channels * stride_product channels
        # We need to group them to match out_channels
        self.group_size = (in_channels * stride_product) // out_channels

        # Conv outputs: out_channels // stride_product
        # After space-to-depth: (out_channels // stride_product) * stride_product = out_channels
        conv_out_channels = out_channels // stride_product
        self.conv = Conv3dSimple(in_channels, conv_out_channels)

    def _space_to_depth(self, x: mx.array) -> mx.array:
        """Apply space-to-depth rearrangement."""
        b, c, t, h, w = x.shape
        st, sh, sw = self.stride

        # Reshape to separate stride factors
        # (B, C, T, H, W) -> (B, C, T/st, st, H/sh, sh, W/sw, sw)
        x = x.reshape(b, c, t // st, st, h // sh, sh, w // sw, sw)

        # Permute to pack stride factors into channels
        # -> (B, C, st, sh, sw, T/st, H/sh, W/sw)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)

        # Flatten to (B, C*st*sh*sw, T/st, H/sh, W/sw)
        x = x.reshape(b, c * st * sh * sw, t // st, h // sh, w // sw)

        return x

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Downsample via conv then space-to-depth with residual."""
        st = self.stride[0]

        # Temporal padding for causal consistency (duplicate first frame)
        if st == 2:
            x = mx.concatenate([x[:, :, :1, :, :], x], axis=2)

        # Residual path: space-to-depth then group mean
        b = x.shape[0]

        # Space-to-depth on input: (B, C, T, H, W) -> (B, C*prod(stride), T', H', W')
        x_in = self._space_to_depth(x)

        # Group channels and take mean to match out_channels
        # (B, C*prod(stride), T', H', W') -> (B, out_channels, group_size, T', H', W') -> mean -> (B, out_channels, T', H', W')
        _, _, t_out, h_out, w_out = x_in.shape
        x_in = x_in.reshape(b, self.out_channels, self.group_size, t_out, h_out, w_out)
        x_in = mx.mean(x_in, axis=2)

        # Main path: conv then space-to-depth
        x = self.conv(x, causal=causal)
        x = self._space_to_depth(x)

        # Add residual
        x = x + x_in

        return x


class SimpleVideoEncoder(nn.Module):
    """
    Simplified VAE encoder that matches PyTorch weight structure.

    Architecture (from weight analysis):
    - patchify: 3 -> 48 channels (patch_size=4)
    - conv_in: 48 -> 128
    - down_blocks.0: 4 res blocks (128 ch)
    - down_blocks.1: SpaceToDepthDownsample (128 -> 256, stride (1,2,2))
    - down_blocks.2: 6 res blocks (256 ch)
    - down_blocks.3: SpaceToDepthDownsample (256 -> 512, stride (2,1,1))
    - down_blocks.4: 6 res blocks (512 ch)
    - down_blocks.5: SpaceToDepthDownsample (512 -> 1024, stride (2,2,2))
    - down_blocks.6: 2 res blocks (1024 ch)
    - down_blocks.7: SpaceToDepthDownsample (1024 -> 2048, stride (2,2,2))
    - down_blocks.8: 2 res blocks (2048 ch)
    - pixel_norm + SiLU
    - conv_out: 2048 -> 129 (128 means + 1 uniform logvar)
    - per-channel normalization

    Total compression: 1:192 (32x spatial from 8x d2s + 4x patchify, 8x temporal)
    """

    def __init__(self, compute_dtype: mx.Dtype = mx.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.patch_size = 4

        # Per-channel statistics for normalization
        self.per_channel_statistics = PerChannelStatistics(latent_channels=128)

        # Conv in: 48 (patchified RGB) -> 128
        self.conv_in = Conv3dSimple(48, 128)

        # Down blocks - architecture from weight analysis
        self.down_blocks_0 = EncoderResBlockGroup(128, num_blocks=4)
        self.down_blocks_1 = SpaceToDepthDownsample3d(128, 256, stride=(1, 2, 2))  # compress_space_res
        self.down_blocks_2 = EncoderResBlockGroup(256, num_blocks=6)
        self.down_blocks_3 = SpaceToDepthDownsample3d(256, 512, stride=(2, 1, 1))  # compress_time_res
        self.down_blocks_4 = EncoderResBlockGroup(512, num_blocks=6)
        self.down_blocks_5 = SpaceToDepthDownsample3d(512, 1024, stride=(2, 2, 2))  # compress_all_res
        self.down_blocks_6 = EncoderResBlockGroup(1024, num_blocks=2)
        self.down_blocks_7 = SpaceToDepthDownsample3d(1024, 2048, stride=(2, 2, 2))  # compress_all_res
        self.down_blocks_8 = EncoderResBlockGroup(2048, num_blocks=2)

        # Conv out: 2048 -> 129 (128 means + 1 uniform logvar)
        self.conv_out = Conv3dSimple(2048, 129)

    def __call__(
        self,
        video: mx.array,
        show_progress: bool = True,
    ) -> mx.array:
        """
        Encode video to latent.

        Args:
            video: Video tensor (B, 3, F, H, W) in [-1, 1].
                   F must satisfy (F - 1) % 8 == 0 (e.g., 1, 9, 17, 25, 33...).
            show_progress: Whether to show progress bar.

        Returns:
            Normalized latent tensor (B, 128, F', H', W') where:
            - F' = 1 + (F - 1) / 8
            - H' = H / 32
            - W' = W / 32
        """
        pbar = None
        try:
            from tqdm import tqdm
            has_tqdm = show_progress
        except ImportError:
            has_tqdm = False
            tqdm = None

        # Validate frame count
        frames_count = video.shape[2]
        if (frames_count - 1) % 8 != 0:
            raise ValueError(
                f"Invalid number of frames: {frames_count}. "
                f"Encoder input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)."
            )

        # Cast to compute dtype for memory efficiency
        if self.compute_dtype != mx.float32:
            video = video.astype(self.compute_dtype)

        def step(x, block, desc):
            nonlocal pbar
            x = block(x, causal=True)
            mx.eval(x)
            if has_tqdm and pbar is not None:
                pbar.update(1)
                pbar.set_description(desc)
            return x

        if has_tqdm:
            pbar = tqdm(total=12, desc="VAE encode", ncols=80)

        # Patchify: (B, 3, F, H, W) -> (B, 48, F, H/4, W/4)
        x = patchify(video, patch_size_hw=self.patch_size, patch_size_t=1)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("patchify done")

        # Conv in
        x = self.conv_in(x, causal=True)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("conv_in done")

        # Down blocks with progress
        x = step(x, self.down_blocks_0, "down_block 0")  # 128ch
        x = step(x, self.down_blocks_1, "down_block 1")  # 128->256, compress_space
        x = step(x, self.down_blocks_2, "down_block 2")  # 256ch
        x = step(x, self.down_blocks_3, "down_block 3")  # 256->512, compress_time
        x = step(x, self.down_blocks_4, "down_block 4")  # 512ch
        x = step(x, self.down_blocks_5, "down_block 5")  # 512->1024, compress_all
        x = step(x, self.down_blocks_6, "down_block 6")  # 1024ch
        x = step(x, self.down_blocks_7, "down_block 7")  # 1024->2048, compress_all
        x = step(x, self.down_blocks_8, "down_block 8")  # 2048ch

        # Final norm and activation
        x = _pixel_norm(x)
        x = nn.silu(x)

        # Conv out: 2048 -> 129 (128 means + 1 uniform logvar)
        x = self.conv_out(x, causal=True)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("conv_out done")
            pbar.close()

        # Extract means (first 128 channels), discard logvar
        means = x[:, :128, :, :, :]

        # Normalize using per-channel statistics
        means = self.per_channel_statistics.normalize(means)

        # Cast output back to float32
        if self.compute_dtype != mx.float32:
            means = means.astype(mx.float32)

        return means


def load_vae_encoder_weights(encoder: SimpleVideoEncoder, weights_path: str) -> None:
    """
    Load VAE encoder weights from PyTorch safetensors file.

    Args:
        encoder: SimpleVideoEncoder instance to load weights into.
        weights_path: Path to safetensors file containing VAE weights.
    """
    from safetensors import safe_open
    import torch

    print(f"Loading VAE encoder weights from {weights_path}...")

    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        # Load per-channel statistics
        for stat_key in ["mean-of-means", "std-of-means", "mean-of-stds", "mean-of-stds-over-std-of-means", "channel"]:
            pt_key = f"vae.per_channel_statistics.{stat_key}"
            if pt_key in f.keys():
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                # Convert hyphenated names to underscored attribute names
                attr_name = stat_key.replace("-", "_")
                setattr(encoder.per_channel_statistics, attr_name, value)
                loaded_count += 1

        # Load conv_in
        for suffix in ["weight", "bias"]:
            pt_key = f"vae.encoder.conv_in.conv.{suffix}"
            if pt_key in f.keys():
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                if suffix == "weight":
                    encoder.conv_in.weight = value
                else:
                    encoder.conv_in.bias = value
                loaded_count += 1

        # Load down_blocks
        # Block structure from weights:
        # 0: res_x (4 blocks at 128ch)
        # 1: compress_space_res (conv only)
        # 2: res_x (6 blocks at 256ch)
        # 3: compress_time_res (conv only)
        # 4: res_x (6 blocks at 512ch)
        # 5: compress_all_res (conv only)
        # 6: res_x (2 blocks at 1024ch)
        # 7: compress_all_res (conv only)
        # 8: res_x (2 blocks at 2048ch)

        block_config = [
            (0, "down_blocks_0", "res", 4),
            (1, "down_blocks_1", "downsample", 0),
            (2, "down_blocks_2", "res", 6),
            (3, "down_blocks_3", "downsample", 0),
            (4, "down_blocks_4", "res", 6),
            (5, "down_blocks_5", "downsample", 0),
            (6, "down_blocks_6", "res", 2),
            (7, "down_blocks_7", "downsample", 0),
            (8, "down_blocks_8", "res", 2),
        ]

        for pt_idx, mlx_name, block_type, num_blocks in block_config:
            block = getattr(encoder, mlx_name)

            if block_type == "res":
                # Load res blocks
                for res_idx in range(num_blocks):
                    res_block = block.res_blocks[res_idx]

                    # Load conv1 and conv2
                    for conv_name in ["conv1", "conv2"]:
                        conv = getattr(res_block, conv_name)
                        for suffix in ["weight", "bias"]:
                            pt_key = f"vae.encoder.down_blocks.{pt_idx}.res_blocks.{res_idx}.{conv_name}.conv.{suffix}"
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

            else:  # downsample
                # Load downsample conv
                for suffix in ["weight", "bias"]:
                    pt_key = f"vae.encoder.down_blocks.{pt_idx}.conv.conv.{suffix}"
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
            pt_key = f"vae.encoder.conv_out.conv.{suffix}"
            if pt_key in f.keys():
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                if suffix == "weight":
                    encoder.conv_out.weight = value
                else:
                    encoder.conv_out.bias = value
                loaded_count += 1

    print(f"  Loaded {loaded_count} weight tensors")


def encode_video(
    video: mx.array,
    encoder: SimpleVideoEncoder,
) -> mx.array:
    """
    Encode video frames to normalized latent representation.

    Args:
        video: Video frames as uint8 (T, H, W, 3) in [0, 255] or
               float32 (B, 3, T, H, W) in [-1, 1].
        encoder: Loaded SimpleVideoEncoder instance.

    Returns:
        Normalized latent tensor (B, 128, T', H', W').
    """
    # Convert uint8 (T, H, W, C) to float32 (B, C, T, H, W)
    if video.ndim == 4 and video.shape[-1] == 3:
        # (T, H, W, C) -> (B, C, T, H, W)
        video = video.transpose(3, 0, 1, 2)[None]  # (1, C, T, H, W)
        if video.dtype == mx.uint8:
            video = video.astype(mx.float32) / 127.5 - 1.0

    # Add batch dim if needed
    if video.ndim == 4:
        video = video[None]

    # Encode
    latent = encoder(video)

    return latent
