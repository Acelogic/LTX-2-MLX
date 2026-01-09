"""Simplified Video VAE Decoder for inference with PyTorch weight loading."""

from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import math

from .ops import unpatchify


def get_timestep_embedding(timesteps: mx.array, embedding_dim: int = 256) -> mx.array:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: Scalar or 1D array of timesteps (B,)
        embedding_dim: Output embedding dimension (default 256)

    Returns:
        Timestep embeddings (B, embedding_dim)
    """
    # Ensure timesteps is at least 1D
    if timesteps.ndim == 0:
        timesteps = timesteps.reshape(1)

    half_dim = embedding_dim // 2
    # Log-spaced frequencies
    freqs = mx.exp(
        -math.log(10000.0) * mx.arange(half_dim, dtype=mx.float32) / half_dim
    )

    # Outer product of timesteps and frequencies
    args = timesteps[:, None] * freqs[None, :]

    # Concatenate sin and cos
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    return embedding


class TimestepEmbedder(nn.Module):
    """
    MLP for processing timestep embeddings.

    Takes sinusoidal embedding and projects to output dimension.
    """

    def __init__(self, hidden_dim: int, output_dim: int, input_dim: int = 256):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Project timestep embedding."""
        x = self.linear_1(x)
        x = nn.silu(x)
        x = self.linear_2(x)
        return x


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
        b, c, t, h, w = x.shape
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
            # For output time t_out, we need input times [t_out, t_out+1, ..., t_out+k-1]
            # So for kernel position kt, we use input[:, :, kt:kt+t_out_len, :, :]
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


class ResBlock3d(nn.Module):
    """3D residual block with pixel norm and scale/shift conditioning."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = Conv3dSimple(channels, channels)
        self.conv2 = Conv3dSimple(channels, channels)
        self.scale_shift_table = mx.zeros((4, channels))

    def __call__(
        self, x: mx.array, causal: bool = True, time_emb: Optional[mx.array] = None
    ) -> mx.array:
        """
        Apply residual block with pixel norm.

        Args:
            x: Input tensor (B, C, T, H, W)
            causal: Whether to use causal padding
            time_emb: Optional timestep embedding (B, 4*C) to add to scale_shift_table

        Returns:
            Output tensor (B, C, T, H, W)
        """
        residual = x

        # Get scale/shift values, optionally adding timestep embedding
        if time_emb is not None:
            # time_emb shape: (B, 4*C) -> reshape to (B, 4, C)
            b = time_emb.shape[0]
            time_emb = time_emb.reshape(b, 4, self.channels)
            # Add to table: (4, C) + (B, 4, C) -> (B, 4, C)
            ss_table = self.scale_shift_table[None, :, :] + time_emb
            # Reshape for broadcasting: (B, 4, C) -> extract rows
            shift1 = ss_table[:, 0, :][:, :, None, None, None]  # (B, C, 1, 1, 1)
            scale1 = 1 + ss_table[:, 1, :][:, :, None, None, None]
            shift2 = ss_table[:, 2, :][:, :, None, None, None]
            scale2 = 1 + ss_table[:, 3, :][:, :, None, None, None]
        else:
            shift1 = self.scale_shift_table[0][None, :, None, None, None]
            scale1 = 1 + self.scale_shift_table[1][None, :, None, None, None]
            shift2 = self.scale_shift_table[2][None, :, None, None, None]
            scale2 = 1 + self.scale_shift_table[3][None, :, None, None, None]

        # Block 1: norm -> scale/shift -> activation -> conv1
        x = _pixel_norm(x)
        x = x * scale1 + shift1
        x = nn.silu(x)
        x = self.conv1(x, causal=causal)

        # Block 2: norm -> scale/shift -> activation -> conv2
        x = _pixel_norm(x)
        x = x * scale2 + shift2
        x = nn.silu(x)
        x = self.conv2(x, causal=causal)

        return x + residual


class DepthToSpaceUpsample3d(nn.Module):
    """
    Upsample using depth-to-space (pixel shuffle) in 3D with residual connection.

    This block halves the channel count while upsampling spatially/temporally.
    For factor (2,2,2): in_channels -> in_channels/2, with 2x upsampling in T,H,W.

    Matches LTX2VideoUpsampler3d from diffusers:
    - Conv expands channels by factor/upscale_factor
    - Reshape/permute/flatten performs depth-to-space
    - Trims first (factor[0]-1) frames for causal consistency
    - Residual path: input is upsampled via d2s and repeated channels, then added
    """

    def __init__(self, in_channels: int, factor: Tuple[int, int, int] = (2, 2, 2), residual: bool = False):
        super().__init__()
        self.factor = factor
        self.residual = residual  # TODO: implement proper FIR upsampler for residual path
        ft, fh, fw = factor
        factor_product = ft * fh * fw

        # Output channels after depth-to-space is half of input
        # upscale_factor = 2 in LTX
        self.upscale_factor = 2
        self.out_channels = in_channels // self.upscale_factor
        # Conv outputs enough channels for d2s to produce out_channels
        conv_out_channels = self.out_channels * factor_product
        self.conv = Conv3dSimple(in_channels, conv_out_channels)

        # Number of channel repeats for residual path
        # repeats = factor_product // upscale_factor = 8 // 2 = 4
        self.channel_repeats = factor_product // self.upscale_factor

    def _depth_to_space(self, x: mx.array, c_out: int) -> mx.array:
        """Apply depth-to-space rearrangement."""
        b, c, t, h, w = x.shape
        ft, fh, fw = self.factor

        # Reshape to separate stride factors
        x = x.reshape(b, c_out, ft, fh, fw, t, h, w)
        # Permute to interleave spatial/temporal with their stride factors
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (B, C, T, ft, H, fh, W, fw)
        # Flatten to get upsampled dimensions
        x = x.reshape(b, c_out, t * ft, h * fh, w * fw)
        return x

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Upsample via conv then depth-to-space with optional residual."""
        ft, fh, fw = self.factor

        # Residual path: upsample input via d2s, repeat channels
        if self.residual:
            b, c_in, t, h, w = x.shape
            # The input has in_channels, d2s expects c_out * factor_product
            # We need to expand input to match this
            # c_in = out_channels * upscale_factor = out_channels * 2
            # For d2s: need c_out * factor_product = out_channels * 8
            # So we repeat by 8/2 = 4
            residual = self._depth_to_space(x, c_in // (ft * fh * fw // self.channel_repeats))
            # Actually the input channel count may not work directly, use a simpler approach:
            # Repeat channels then d2s
            residual = mx.repeat(x, self.channel_repeats, axis=1)  # (B, C*4, T, H, W)
            residual = self._depth_to_space(residual, self.out_channels)
            if ft > 1 and causal:
                residual = residual[:, :, ft - 1:]

        # Main path: conv then d2s
        x = self.conv(x, causal=causal)
        x = self._depth_to_space(x, self.out_channels)

        # Trim first (ft-1) frames for causal consistency
        if ft > 1 and causal:
            x = x[:, :, ft - 1:]

        # Add residual
        if self.residual:
            x = x + residual

        return x


class ResBlockGroup(nn.Module):
    """Group of residual blocks with optional timestep embedding."""

    def __init__(self, channels: int, num_blocks: int = 5):
        super().__init__()
        self.channels = channels
        self.res_blocks = [ResBlock3d(channels) for _ in range(num_blocks)]
        # Time embedder: outputs 4*channels for each res block (scale/shift for 2 norms)
        # But in LTX, the time_embedder outputs 4*channels and broadcasts to all blocks
        self.time_embedder = None  # Will be set during weight loading if available

    def __call__(
        self, x: mx.array, causal: bool = True, timestep: Optional[mx.array] = None
    ) -> mx.array:
        """
        Apply all res blocks sequentially.

        Args:
            x: Input tensor (B, C, T, H, W)
            causal: Whether to use causal padding
            timestep: Optional scaled timestep (B,) for conditioning

        Returns:
            Output tensor (B, C, T, H, W)
        """
        # Compute time embedding if timestep provided and embedder exists
        time_emb = None
        if timestep is not None and self.time_embedder is not None:
            # Get sinusoidal embedding
            t_emb = get_timestep_embedding(timestep, 256)  # (B, 256)
            # Project through time embedder
            time_emb = self.time_embedder(t_emb)  # (B, 4*channels)

        for block in self.res_blocks:
            x = block(x, causal=causal, time_emb=time_emb)
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
    - up_blocks.0: 5 res blocks (1024 ch) + timestep conditioning
    - up_blocks.1: depth-to-space upsample (1024 -> 512 ch, 2x2x2 spatial/temporal)
    - up_blocks.2: 5 res blocks (512 ch) + timestep conditioning
    - up_blocks.3: depth-to-space upsample (512 -> 256 ch, 2x2x2)
    - up_blocks.4: 5 res blocks (256 ch) + timestep conditioning
    - up_blocks.5: depth-to-space upsample (256 -> 128 ch, 2x2x2)
    - up_blocks.6: 5 res blocks (128 ch) + timestep conditioning
    - final norm + timestep conditioning
    - conv_out: 128 -> 48 (for patch_size=4 unpatchify)

    Total upsampling: 8x temporal, 8x spatial from d2s + 4x spatial from unpatchify = 32x spatial
    """

    def __init__(self):
        super().__init__()

        # Per-channel statistics for denormalization
        self.mean_of_means = mx.zeros((128,))
        self.std_of_means = mx.zeros((128,))

        # Timestep conditioning
        self.timestep_scale_multiplier = mx.array(1000.0)

        # Conv in: 128 -> 1024
        self.conv_in = Conv3dSimple(128, 1024)

        # Up blocks
        # Factor (2,2,2) = 8x channel reduction (also halves channels) + 2x upsample in T,H,W
        # 1024 -> conv(4096) -> d2s -> 512 ch, 2x spatial/temporal
        self.up_blocks_0 = ResBlockGroup(1024, num_blocks=5)
        self.up_blocks_1 = DepthToSpaceUpsample3d(1024, factor=(2, 2, 2))  # -> 512 ch
        self.up_blocks_2 = ResBlockGroup(512, num_blocks=5)
        self.up_blocks_3 = DepthToSpaceUpsample3d(512, factor=(2, 2, 2))   # -> 256 ch
        self.up_blocks_4 = ResBlockGroup(256, num_blocks=5)
        self.up_blocks_5 = DepthToSpaceUpsample3d(256, factor=(2, 2, 2))   # -> 128 ch
        self.up_blocks_6 = ResBlockGroup(128, num_blocks=5)

        # Conv out: 128 -> 48 (3 * 16 for patch_size=4 unpatchify)
        self.conv_out = Conv3dSimple(128, 48)

        # Scale/shift for final norm
        self.last_scale_shift_table = mx.zeros((2, 128))
        # Time embedder for final norm (outputs 2*128=256 for scale+shift)
        self.last_time_embedder = None  # Will be set during weight loading

    def __call__(
        self,
        latent: mx.array,
        timestep: Optional[float] = 0.05,
        show_progress: bool = True,
    ) -> mx.array:
        """
        Decode latent to video.

        Args:
            latent: Latent tensor (B, 128, T, H, W).
            timestep: Timestep for conditioning (default 0.05 for denoising).
                      Use 0.0 for no denoising, None to disable timestep conditioning.
            show_progress: Whether to show progress bar.

        Returns:
            Video tensor (B, 3, T*8, H*32, W*32).
        """
        pbar = None
        try:
            from tqdm import tqdm
            has_tqdm = show_progress
        except ImportError:
            has_tqdm = False
            tqdm = None

        batch_size = latent.shape[0]

        # Compute scaled timestep
        scaled_timestep = None
        if timestep is not None:
            t = mx.array([timestep] * batch_size)
            scaled_timestep = t * self.timestep_scale_multiplier

        def step_res(x, block, desc):
            nonlocal pbar
            x = block(x, causal=True, timestep=scaled_timestep)
            mx.eval(x)
            if has_tqdm and pbar is not None:
                pbar.update(1)
                pbar.set_description(desc)
            return x

        def step_up(x, block, desc):
            nonlocal pbar
            x = block(x, causal=True)
            mx.eval(x)
            if has_tqdm and pbar is not None:
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
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("conv_in done")

        # Up blocks with progress and timestep conditioning
        x = step_res(x, self.up_blocks_0, "res_blocks 1/4")
        x = step_up(x, self.up_blocks_1, "upsample 1/3")
        x = step_res(x, self.up_blocks_2, "res_blocks 2/4")
        x = step_up(x, self.up_blocks_3, "upsample 2/3")
        x = step_res(x, self.up_blocks_4, "res_blocks 3/4")
        x = step_up(x, self.up_blocks_5, "upsample 3/3")
        x = step_res(x, self.up_blocks_6, "res_blocks 4/4")

        # Final norm and activation with optional timestep conditioning
        x = _pixel_norm(x)

        # Get scale/shift, optionally adding timestep embedding
        # LTX ordering: (shift, scale) = unbind(dim=1), so row 0 is shift, row 1 is scale
        if scaled_timestep is not None and self.last_time_embedder is not None:
            # Get sinusoidal embedding
            t_emb = get_timestep_embedding(scaled_timestep, 256)  # (B, 256)
            # Project through time embedder
            time_emb = self.last_time_embedder(t_emb)  # (B, 256) = (B, 2*128)
            time_emb = time_emb.reshape(batch_size, 2, 128)
            # Add to table: (2, 128) + (B, 2, 128) -> (B, 2, 128)
            ss_table = self.last_scale_shift_table[None, :, :] + time_emb
            # Row 0 = shift, Row 1 = scale (LTX ordering)
            shift = ss_table[:, 0, :][:, :, None, None, None]  # (B, C, 1, 1, 1)
            scale = 1 + ss_table[:, 1, :][:, :, None, None, None]
        else:
            # Without timestep: row 0 = shift, row 1 = scale
            shift = self.last_scale_shift_table[0][None, :, None, None, None]
            scale = 1 + self.last_scale_shift_table[1][None, :, None, None, None]

        x = x * scale + shift
        x = nn.silu(x)

        # Conv out
        x = self.conv_out(x, causal=True)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("conv_out done")

        # Unpatchify: (B, 48, T, H, W) -> (B, 3, T, H*4, W*4)
        x = unpatchify(x, patch_size_hw=4, patch_size_t=1)
        mx.eval(x)
        if has_tqdm and pbar is not None:
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

        # Load timestep_scale_multiplier
        pt_key = "vae.decoder.timestep_scale_multiplier"
        if pt_key in f.keys():
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            decoder.timestep_scale_multiplier = mx.array(tensor.numpy())
            loaded_count += 1

        # Load last_time_embedder
        pt_prefix = "vae.decoder.last_time_embedder.timestep_embedder"
        if f"{pt_prefix}.linear_1.weight" in f.keys():
            # Create the embedder: 256 -> 256 -> 256 (output is 2*128=256 for scale+shift)
            decoder.last_time_embedder = TimestepEmbedder(
                hidden_dim=256, output_dim=256, input_dim=256
            )
            for layer_name in ["linear_1", "linear_2"]:
                for suffix in ["weight", "bias"]:
                    pt_key = f"{pt_prefix}.{layer_name}.{suffix}"
                    if pt_key in f.keys():
                        tensor = f.get_tensor(pt_key)
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.to(torch.float32)
                        value = mx.array(tensor.numpy())
                        layer = getattr(decoder.last_time_embedder, layer_name)
                        setattr(layer, suffix, value)
                        loaded_count += 1

        # Load time embedders for res blocks (up_blocks 0, 2, 4, 6)
        res_block_channels = {0: 1024, 2: 512, 4: 256, 6: 128}
        for pt_idx, channels in res_block_channels.items():
            mlx_name = f"up_blocks_{pt_idx}"
            block = getattr(decoder, mlx_name)
            pt_prefix = f"vae.decoder.up_blocks.{pt_idx}.time_embedder.timestep_embedder"

            if f"{pt_prefix}.linear_1.weight" in f.keys():
                # Create the embedder: 256 -> hidden -> 4*channels (for scale/shift of 2 norms)
                # Output dim is 4*channels because each res block has 2 norms with scale+shift
                output_dim = 4 * channels
                # Hidden dim from weight shape
                l1_weight = f.get_tensor(f"{pt_prefix}.linear_1.weight")
                hidden_dim = l1_weight.shape[0]

                block.time_embedder = TimestepEmbedder(
                    hidden_dim=hidden_dim, output_dim=output_dim, input_dim=256
                )
                for layer_name in ["linear_1", "linear_2"]:
                    for suffix in ["weight", "bias"]:
                        pt_key = f"{pt_prefix}.{layer_name}.{suffix}"
                        if pt_key in f.keys():
                            tensor = f.get_tensor(pt_key)
                            if tensor.dtype == torch.bfloat16:
                                tensor = tensor.to(torch.float32)
                            value = mx.array(tensor.numpy())
                            layer = getattr(block.time_embedder, layer_name)
                            setattr(layer, suffix, value)
                            loaded_count += 1

    print(f"  Loaded {loaded_count} weight tensors")


def decode_latent(
    latent: mx.array,
    decoder: SimpleVideoDecoder,
    timestep: Optional[float] = 0.05,
) -> mx.array:
    """
    Decode latent to video frames.

    Args:
        latent: Latent tensor (B, 128, T, H, W) or (128, T, H, W).
        decoder: Loaded SimpleVideoDecoder instance.
        timestep: Timestep for conditioning (default 0.05 for denoising).
                  Use 0.0 for no denoising, None to disable timestep conditioning.

    Returns:
        Video frames as uint8 (T, H, W, 3) in [0, 255].
    """
    # Add batch dim if needed
    if latent.ndim == 4:
        latent = latent[None]

    # Decode with timestep conditioning
    video = decoder(latent, timestep=timestep)

    # Apply bias correction to center output at 0
    # The decoder outputs with a consistent negative bias (~-0.31)
    # This correction brings brightness from ~35% to ~50%
    video = video + 0.31

    # Convert to uint8: assume output is in [-1, 1]
    video = mx.clip((video + 1) / 2, 0, 1) * 255
    video = video.astype(mx.uint8)

    # Rearrange: (B, C, T, H, W) -> (T, H, W, C)
    video = video[0]  # Remove batch
    video = video.transpose(1, 2, 3, 0)

    return video
