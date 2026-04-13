"""Simplified Video VAE Decoder for inference with PyTorch weight loading."""

from typing import Optional, List, Tuple, Any

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

        # Spatial padding with reflect mode (matches PyTorch decoder_spatial_padding_mode=REFLECT)
        if p > 0:
            # MLX doesn't have native reflect padding, so we implement it manually
            # Reflect padding mirrors the edge pixels: [1,2,3] with pad=2 -> [3,2,1,2,3,2,1]
            # For spatial dimensions (H and W)
            # Pad height (dim 3)
            h_pad_top = x[:, :, :, 1:p+1, :][:, :, :, ::-1, :]  # Reflect top edge
            h_pad_bot = x[:, :, :, -(p+1):-1, :][:, :, :, ::-1, :]  # Reflect bottom edge
            x = mx.concatenate([h_pad_top, x, h_pad_bot], axis=3)
            # Pad width (dim 4)
            w_pad_left = x[:, :, :, :, 1:p+1][:, :, :, :, ::-1]  # Reflect left edge
            w_pad_right = x[:, :, :, :, -(p+1):-1][:, :, :, :, ::-1]  # Reflect right edge
            x = mx.concatenate([w_pad_left, x, w_pad_right], axis=4)

        # Temporal padding: need k-1 total padding to preserve temporal dim
        t_pad_needed = k - 1
        if causal and t_pad_needed > 0:
            # Causal: all padding at the beginning (replicate first frame)
            first_frames = mx.repeat(x[:, :, :1], t_pad_needed, axis=2)
            x = mx.concatenate([first_frames, x], axis=2)
        elif t_pad_needed > 0:
            # Non-causal: symmetric padding using frame replication (NOT zero padding)
            # This matches PyTorch CausalConv3d behavior
            pad_before = t_pad_needed // 2
            pad_after = t_pad_needed - pad_before
            # Replicate first frame for padding at start
            first_frames = mx.repeat(x[:, :, :1], pad_before, axis=2)
            # Replicate last frame for padding at end
            last_frames = mx.repeat(x[:, :, -1:], pad_after, axis=2)
            x = mx.concatenate([first_frames, x, last_frames], axis=2)

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
        # Note: kept as float32 for numerical stability
        self.scale_shift_table = mx.zeros((4, channels), dtype=mx.float32)

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
    Upsample using depth-to-space (pixel shuffle) in 3D with optional residual connection.

    Supports variable stride factors and channel reduction:
    - stride (2,2,2): 2x upsample in T, H, W (compress_all)
    - stride (2,1,1): 2x upsample in T only (compress_time)
    - stride (1,2,2): 2x upsample in H, W only (compress_space)
    - out_channels_reduction_factor: channel reduction (1=keep, 2=halve)
    """

    def __init__(
        self,
        in_channels: int,
        stride: Tuple[int, int, int] = (2, 2, 2),
        residual: bool = False,
        out_channels_reduction_factor: int = 2,
    ):
        super().__init__()
        self.stride = stride
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        stride_product = math.prod(stride)

        # Final output channels after d2s = in_channels / reduction_factor
        self.final_out_channels = in_channels // out_channels_reduction_factor

        # Conv output channels (before d2s rearrange)
        conv_out_channels = stride_product * in_channels // out_channels_reduction_factor
        self.conv = Conv3dSimple(in_channels, conv_out_channels)

    def _depth_to_space(self, x: mx.array, c_out: int) -> mx.array:
        """Apply depth-to-space rearrangement."""
        b, c, t, h, w = x.shape
        ft, fh, fw = self.stride

        # Reshape to separate stride factors
        x = x.reshape(b, c_out, ft, fh, fw, t, h, w)
        # Permute to interleave spatial/temporal with their stride factors
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (B, C, T, ft, H, fh, W, fw)
        # Flatten to get upsampled dimensions
        x = x.reshape(b, c_out, t * ft, h * fh, w * fw)
        return x

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Upsample via conv then depth-to-space with optional residual."""
        ft, fh, fw = self.stride
        stride_product = ft * fh * fw

        # Residual path
        if self.residual:
            b, c_in, t, h, w = x.shape
            c_d2s = c_in // stride_product
            residual = self._depth_to_space(x, c_d2s)
            if ft > 1:
                residual = residual[:, :, 1:]
            num_repeat = stride_product // self.out_channels_reduction_factor
            residual = mx.tile(residual, (1, num_repeat, 1, 1, 1))

        # Main path: conv then d2s
        x = self.conv(x, causal=causal)
        x = self._depth_to_space(x, self.final_out_channels)

        # Trim first frame when temporal stride > 1
        if ft > 1:
            x = x[:, :, 1:]

        if self.residual:
            x = x + residual

        return x


class ResBlockGroup(nn.Module):
    """Group of residual blocks with optional timestep embedding."""

    def __init__(self, channels: int, num_blocks: int = 5):
        super().__init__()
        self.channels = channels
        self.res_blocks = [ResBlock3d(channels) for _ in range(num_blocks)]
        self.time_embedder = None  # Will be set during weight loading if available

    def __call__(
        self, x: mx.array, causal: bool = True, timestep: Optional[mx.array] = None
    ) -> mx.array:
        # Compute time embedding if timestep provided and embedder exists
        time_emb = None
        if timestep is not None and self.time_embedder is not None:
            t_emb = get_timestep_embedding(timestep, 256)
            time_emb = self.time_embedder(t_emb)

        for block in self.res_blocks:
            x = block(x, causal=causal, time_emb=time_emb)
        return x


def _pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Apply pixel normalization (normalize across channels)."""
    variance = mx.mean(x * x, axis=1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


# Stride mapping for decoder block types
_STRIDE_MAP = {
    "compress_all": (2, 2, 2),
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
}

# Default V2.0 decoder blocks (used when no config is provided)
_DEFAULT_DECODER_BLOCKS = [
    ["res_x", {"num_layers": 5}],
    ["compress_all", {"multiplier": 2, "residual": True}],
    ["res_x", {"num_layers": 5}],
    ["compress_all", {"multiplier": 2, "residual": True}],
    ["res_x", {"num_layers": 5}],
    ["compress_all", {"multiplier": 2, "residual": True}],
    ["res_x", {"num_layers": 5}],
]


class SimpleVideoDecoder(nn.Module):
    """
    Config-driven VAE decoder that supports both V2.0 and V2.3 architectures.

    Architecture is determined by decoder_blocks config from checkpoint metadata.
    Blocks are built in reverse order (latent → output), matching PyTorch.
    """

    def __init__(
        self,
        decoder_blocks: Optional[List] = None,
        base_channels: int = 128,
        timestep_conditioning: bool = True,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.timestep_conditioning = timestep_conditioning

        if decoder_blocks is None:
            decoder_blocks = _DEFAULT_DECODER_BLOCKS

        # Per-channel statistics for denormalization
        self.mean_of_means = mx.zeros((128,), dtype=mx.float32)
        self.std_of_means = mx.zeros((128,), dtype=mx.float32)

        # Noise injection scale
        self.decode_noise_scale = 0.025

        # Feature channels start at base_channels * 8
        feature_channels = base_channels * 8

        # Conv in: latent_channels -> feature_channels
        self.conv_in = Conv3dSimple(128, feature_channels)

        # Build up_blocks from reversed config (PyTorch reverses decoder_blocks)
        self.up_blocks = []
        self.block_types = []  # Track type for forward pass dispatch

        for block_name, block_params in reversed(decoder_blocks):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            if block_name == "res_x":
                num_layers = block_config["num_layers"]
                block = ResBlockGroup(feature_channels, num_blocks=num_layers)
                self.up_blocks.append(block)
                self.block_types.append("res")

            elif block_name in _STRIDE_MAP:
                stride = _STRIDE_MAP[block_name]
                multiplier = block_config.get("multiplier", 1)
                residual = block_config.get("residual", False)
                block = DepthToSpaceUpsample3d(
                    in_channels=feature_channels,
                    stride=stride,
                    residual=residual,
                    out_channels_reduction_factor=multiplier,
                )
                feature_channels = feature_channels // multiplier
                self.up_blocks.append(block)
                self.block_types.append("upsample")

            else:
                raise ValueError(f"Unknown decoder block: {block_name}")

        # Final output channel count (should be base_channels after all reductions)
        self.final_channels = feature_channels

        # Conv out: final_channels -> 48 (3 * patch_size^2 for unpatchify)
        self.conv_out = Conv3dSimple(feature_channels, 48)

        # Scale/shift for final norm
        self.last_scale_shift_table = mx.zeros((2, feature_channels), dtype=mx.float32)

        # Timestep conditioning (only for V2.0-style decoders)
        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = None  # Created during weight loading
        else:
            self.timestep_scale_multiplier = None
            self.last_time_embedder = None

    def __call__(
        self,
        latent: mx.array,
        timestep: Optional[float] = 0.05,
        show_progress: bool = True,
        causal: bool = False,
    ) -> mx.array:
        """
        Decode latent to video.

        Args:
            latent: Latent tensor (B, 128, T, H, W).
            timestep: Timestep for conditioning (default 0.05).
            show_progress: Whether to show progress bar.
            causal: Whether to use causal temporal convolutions.

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

        # Cast to compute dtype for memory efficiency
        if self.compute_dtype != mx.float32:
            latent = latent.astype(self.compute_dtype)

        # Compute scaled timestep
        scaled_timestep = None
        if self.timestep_conditioning and timestep is not None and self.timestep_scale_multiplier is not None:
            t = mx.array([timestep] * batch_size)
            scaled_timestep = t * self.timestep_scale_multiplier

        num_blocks = len(self.up_blocks)
        total_steps = num_blocks + 3  # +3 for conv_in, conv_out, unpatchify

        if has_tqdm:
            pbar = tqdm(total=total_steps, desc="VAE decode", ncols=80)

        # Denormalize latent using per-channel statistics
        x = latent * self.std_of_means[None, :, None, None, None]
        x = x + self.mean_of_means[None, :, None, None, None]

        # Noise injection when timestep conditioning is enabled
        if self.timestep_conditioning and timestep is not None:
            noise = mx.random.normal(x.shape) * self.decode_noise_scale
            x = noise + (1.0 - self.decode_noise_scale) * x

        # Conv in
        x = self.conv_in(x, causal=causal)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("conv_in done")

        # Up blocks
        res_count = 0
        up_count = 0
        total_res = sum(1 for bt in self.block_types if bt == "res")
        total_up = sum(1 for bt in self.block_types if bt == "upsample")

        for i, (block, btype) in enumerate(zip(self.up_blocks, self.block_types)):
            if btype == "res":
                res_count += 1
                x = block(x, causal=causal, timestep=scaled_timestep)
                desc = f"res_blocks {res_count}/{total_res}"
            else:
                up_count += 1
                x = block(x, causal=causal)
                desc = f"upsample {up_count}/{total_up}"
            mx.eval(x)
            if has_tqdm and pbar is not None:
                pbar.update(1)
                pbar.set_description(desc)

        # Final norm and activation with optional timestep conditioning
        x = _pixel_norm(x)

        if scaled_timestep is not None and self.last_time_embedder is not None:
            t_emb = get_timestep_embedding(scaled_timestep, 256)
            time_emb = self.last_time_embedder(t_emb)
            time_emb = time_emb.reshape(batch_size, 2, self.final_channels)
            ss_table = self.last_scale_shift_table[None, :, :] + time_emb
            shift = ss_table[:, 0, :][:, :, None, None, None]
            scale = 1 + ss_table[:, 1, :][:, :, None, None, None]
        else:
            shift = self.last_scale_shift_table[0][None, :, None, None, None]
            scale = 1 + self.last_scale_shift_table[1][None, :, None, None, None]

        x = x * scale + shift
        x = nn.silu(x)

        # Conv out
        x = self.conv_out(x, causal=causal)
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

        # Cast output back to float32 for video export
        if self.compute_dtype != mx.float32:
            x = x.astype(mx.float32)

        return x


def load_vae_decoder_weights(decoder: SimpleVideoDecoder, weights_path: str) -> None:
    """
    Load VAE decoder weights from PyTorch safetensors file.
    Works generically for any decoder architecture built from config.
    """
    from safetensors import safe_open
    import torch

    print(f"Loading VAE decoder weights from {weights_path}...")

    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        all_keys = set(f.keys())

        def load_tensor(pt_key):
            nonlocal loaded_count
            if pt_key not in all_keys:
                return None
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            loaded_count += 1
            return mx.array(tensor.numpy())

        # Load per-channel statistics
        for stat_key, attr_name in [("mean-of-means", "mean_of_means"), ("std-of-means", "std_of_means")]:
            val = load_tensor(f"vae.per_channel_statistics.{stat_key}")
            if val is not None:
                setattr(decoder, attr_name, val)

        # Load conv_in
        for suffix in ["weight", "bias"]:
            val = load_tensor(f"vae.decoder.conv_in.conv.{suffix}")
            if val is not None:
                setattr(decoder.conv_in, suffix, val)

        # Load up_blocks generically
        for pt_idx, (block, btype) in enumerate(zip(decoder.up_blocks, decoder.block_types)):
            if btype == "res":
                # Load res blocks
                for res_idx, res_block in enumerate(block.res_blocks):
                    for conv_name in ["conv1", "conv2"]:
                        conv = getattr(res_block, conv_name)
                        for suffix in ["weight", "bias"]:
                            val = load_tensor(f"vae.decoder.up_blocks.{pt_idx}.res_blocks.{res_idx}.{conv_name}.conv.{suffix}")
                            if val is not None:
                                setattr(conv, suffix, val)

                    # Load scale_shift_table
                    val = load_tensor(f"vae.decoder.up_blocks.{pt_idx}.res_blocks.{res_idx}.scale_shift_table")
                    if val is not None:
                        res_block.scale_shift_table = val

                # Load time embedder for this res block group
                pt_prefix = f"vae.decoder.up_blocks.{pt_idx}.time_embedder.timestep_embedder"
                l1_key = f"{pt_prefix}.linear_1.weight"
                if l1_key in all_keys:
                    l1_weight = f.get_tensor(l1_key)
                    hidden_dim = l1_weight.shape[0]
                    output_dim = 4 * block.channels

                    block.time_embedder = TimestepEmbedder(
                        hidden_dim=hidden_dim, output_dim=output_dim, input_dim=256
                    )
                    for layer_name in ["linear_1", "linear_2"]:
                        for suffix in ["weight", "bias"]:
                            val = load_tensor(f"{pt_prefix}.{layer_name}.{suffix}")
                            if val is not None:
                                layer = getattr(block.time_embedder, layer_name)
                                setattr(layer, suffix, val)

            else:  # upsample
                for suffix in ["weight", "bias"]:
                    val = load_tensor(f"vae.decoder.up_blocks.{pt_idx}.conv.conv.{suffix}")
                    if val is not None:
                        setattr(block.conv, suffix, val)

        # Load conv_out
        for suffix in ["weight", "bias"]:
            val = load_tensor(f"vae.decoder.conv_out.conv.{suffix}")
            if val is not None:
                setattr(decoder.conv_out, suffix, val)

        # Load last_scale_shift_table
        val = load_tensor("vae.decoder.last_scale_shift_table")
        if val is not None:
            decoder.last_scale_shift_table = val

        # Load timestep conditioning params
        if decoder.timestep_conditioning:
            val = load_tensor("vae.decoder.timestep_scale_multiplier")
            if val is not None:
                decoder.timestep_scale_multiplier = val

            pt_prefix = "vae.decoder.last_time_embedder.timestep_embedder"
            if f"{pt_prefix}.linear_1.weight" in all_keys:
                decoder.last_time_embedder = TimestepEmbedder(
                    hidden_dim=256, output_dim=2 * decoder.final_channels, input_dim=256
                )
                for layer_name in ["linear_1", "linear_2"]:
                    for suffix in ["weight", "bias"]:
                        val = load_tensor(f"{pt_prefix}.{layer_name}.{suffix}")
                        if val is not None:
                            layer = getattr(decoder.last_time_embedder, layer_name)
                            setattr(layer, suffix, val)

    print(f"  Loaded {loaded_count} weight tensors")


def decode_latent(
    latent: mx.array,
    decoder: SimpleVideoDecoder,
    timestep: Optional[float] = 0.05,
    key: Optional[mx.array] = None,
    temporal_chunk_size: int = 7,
    temporal_overlap: int = 2,
) -> mx.array:
    """
    Decode latent to video frames.

    Uses temporal chunking for sequences longer than temporal_chunk_size to work
    around a 3D convolution bug in MLX where early frames of long sequences
    produce noise artifacts. Chunks are decoded independently and blended in
    the overlap region using a linear ramp.

    Args:
        latent: Latent tensor (B, 128, T, H, W) or (128, T, H, W).
        decoder: Loaded SimpleVideoDecoder instance.
        timestep: Timestep for conditioning (default 0.05 for denoising).
                  Use 0.0 for no denoising, None to disable timestep conditioning.
        key: Optional random key for deterministic decoding (reserved for future use).
        temporal_chunk_size: Max latent frames per chunk (default 7, proven clean).
        temporal_overlap: Overlap in latent frames between chunks for blending (default 2).

    Returns:
        Video frames as uint8 (T, H, W, 3) in [0, 255].
    """
    # Add batch dim if needed
    if latent.ndim == 4:
        latent = latent[None]

    T = latent.shape[2]  # Temporal dim in latent space

    if T <= temporal_chunk_size:
        # Short enough to decode in one pass
        video = decoder(latent, timestep=timestep)
    else:
        # Tiled temporal decoding: split into overlapping chunks and blend.
        # This works around a 3D convolution bug in MLX where early frames
        # of long temporal sequences produce noise artifacts.

        def latent_t_to_pixel_t(lt):
            """Convert latent temporal dim to pixel frames through 3 upsample stages."""
            pt = lt
            for _ in range(3):
                pt = pt * 2 - 1
            return pt

        # Compute expected total pixel frames
        total_pixel_frames = latent_t_to_pixel_t(T)

        stride = temporal_chunk_size - temporal_overlap

        # Decode chunks
        decoded_chunks = []  # (start_latent, end_latent, decoded_video)
        t = 0
        while t < T:
            end = min(t + temporal_chunk_size, T)
            # Ensure last chunk is at least overlap+1 frames
            if end - t < temporal_overlap + 1 and t > 0:
                t = max(0, end - temporal_chunk_size)
                end = min(t + temporal_chunk_size, T)

            chunk_latent = latent[:, :, t:end, :, :]
            chunk_video = decoder(chunk_latent, timestep=timestep)
            mx.eval(chunk_video)
            decoded_chunks.append((t, end, chunk_video))

            if end >= T:
                break
            t += stride

        if len(decoded_chunks) == 1:
            # Only one chunk — trim to expected length
            video = decoded_chunks[0][2][:, :, :total_pixel_frames, :, :]
        else:
            # Multiple chunks — stitch with overlap blending.
            # Compute overlap in pixel space by decoding the overlap latent count
            # as a standalone chunk to get its exact pixel length.
            overlap_pixel_ref = latent_t_to_pixel_t(temporal_overlap)

            # Start with the first chunk
            video = decoded_chunks[0][2]

            for i in range(1, len(decoded_chunks)):
                curr_start_l, curr_end_l, curr_video = decoded_chunks[i]
                curr_pixel_len = curr_video.shape[2]

                # The overlap in pixel space
                overlap_pixels = min(overlap_pixel_ref, curr_pixel_len, video.shape[2])

                if overlap_pixels <= 1:
                    # No meaningful overlap — just concatenate
                    video = mx.concatenate([video, curr_video], axis=2)
                    continue

                # Get overlap regions from both sides
                prev_overlap = video[:, :, -overlap_pixels:, :, :]
                curr_overlap = curr_video[:, :, :overlap_pixels, :, :]
                curr_tail = curr_video[:, :, overlap_pixels:, :, :]

                # Linear crossfade
                ramp = mx.linspace(0.0, 1.0, overlap_pixels).reshape(1, 1, overlap_pixels, 1, 1)
                blended = prev_overlap * (1.0 - ramp) + curr_overlap * ramp

                # Stitch: keep everything before overlap, add blend, add tail
                video = mx.concatenate([
                    video[:, :, :-overlap_pixels, :, :],
                    blended,
                    curr_tail,
                ], axis=2)

            # Trim to exact expected length
            video = video[:, :, :total_pixel_frames, :, :]

    # Convert to uint8: output is in [-1, 1] (matching PyTorch)
    video = mx.clip((video + 1) / 2, 0, 1) * 255
    video = video.astype(mx.uint8)

    # Rearrange: (B, C, T, H, W) -> (T, H, W, C)
    video = video[0]  # Remove batch
    video = video.transpose(1, 2, 3, 0)

    return video
