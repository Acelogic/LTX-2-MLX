"""Audio VAE Decoder for LTX-2 MLX."""

from enum import Enum
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn

from LTX_2_MLX.components.patchifiers import AudioPatchifier
from LTX_2_MLX.types import AudioLatentShape


LATENT_DOWNSAMPLE_FACTOR = 4


class CausalityAxis(Enum):
    """Enum for specifying the causality axis in causal convolutions.

    Matches PyTorch: ltx_core/model/audio_vae/causality_axis.py
    """

    NONE = None
    WIDTH = "width"
    HEIGHT = "height"


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.

    For each element along the channel dimension, this layer normalizes the tensor
    by the root-mean-square of its values:
        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)

    Matches PyTorch: ltx_core/model/common/normalization.py
    Compatible with causal convolutions (unlike GroupNorm).
    """

    def __init__(self, dim: int = 1, eps: float = 1e-6):
        """
        Args:
            dim: Dimension along which to compute the RMS (typically channels=1).
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply per-pixel RMS normalization."""
        # RMS normalization: x / sqrt(mean(x^2) + eps)
        rms = mx.sqrt(mx.mean(x * x, axis=self.dim, keepdims=True) + self.eps)
        return x / rms


class PerChannelStatistics(nn.Module):
    """Per-channel normalization statistics for audio latents.

    Buffer names match PyTorch checkpoint: `mean-of-means` and `std-of-means`.
    Note: MLX doesn't support hyphens in attribute names, so we use underscores
    internally and handle the mapping during weight loading.
    """

    def __init__(self, latent_channels: int):
        super().__init__()
        self.latent_channels = latent_channels
        # Named to match PyTorch checkpoint (with underscores for MLX compatibility)
        self.mean_of_means = mx.zeros((latent_channels,))
        self.std_of_means = mx.ones((latent_channels,))

    def normalize(self, x: mx.array) -> mx.array:
        """Normalize input: (x - mean) / std. Input shape: (B, T, C)."""
        mean = self.mean_of_means[None, None, :]
        std = self.std_of_means[None, None, :]
        return (x - mean) / std

    def denormalize(self, x: mx.array) -> mx.array:
        """Denormalize input: x * std + mean. Input shape: (B, T, C)."""
        mean = self.mean_of_means[None, None, :]
        std = self.std_of_means[None, None, :]
        return x * std + mean


class CausalConv2d(nn.Module):
    """2D causal convolution with configurable causality axis.

    Supports causality along HEIGHT (frequency/mel), WIDTH (time), or NONE.
    PyTorch default for audio VAE is HEIGHT (causal along frequency axis).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.causality_axis = causality_axis

        # MLX conv2d weight shape: (out_C, kH, kW, in_C)
        self.weight = mx.zeros((out_channels, kernel_size, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

        # Pre-compute padding based on causality axis
        # PyTorch F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
        # MLX mx.pad format: [(batch_before, batch_after), (C_before, C_after), (H_before, H_after), (W_before, W_after)]
        pad_h = kernel_size - 1
        pad_w = kernel_size - 1

        if causality_axis == CausalityAxis.NONE:
            # Symmetric padding
            self._padding = [(0, 0), (0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
        elif causality_axis == CausalityAxis.WIDTH:
            # Causal along width (time): pad left only, symmetric on height
            self._padding = [(0, 0), (0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w, 0)]
        elif causality_axis == CausalityAxis.HEIGHT:
            # Causal along height (frequency): pad top only, symmetric on width
            self._padding = [(0, 0), (0, 0), (pad_h, 0), (pad_w // 2, pad_w - pad_w // 2)]
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

    def __call__(self, x: mx.array) -> mx.array:
        """Apply causal 2D convolution."""
        # x shape: (B, C, H, W) where H is frequency/mel_bins, W is time

        # Apply pre-computed padding
        x = mx.pad(x, self._padding)

        # Transpose for MLX conv2d: (B, C, H, W) -> (B, H, W, C)
        x = x.transpose(0, 2, 3, 1)

        # Apply conv
        out = mx.conv2d(x, self.weight, stride=self.stride)

        # Transpose back: (B, H, W, C) -> (B, C, H, W)
        out = out.transpose(0, 3, 1, 2)

        # Add bias
        out = out + self.bias[None, :, None, None]

        return out


class SimpleResBlock2d(nn.Module):
    """
    2D residual block with PixelNorm normalization.

    Architecture (matching PyTorch ResnetBlock with PIXEL norm type):
        h = norm1(x)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)
        return x + h  (with optional skip projection)

    Uses PixelNorm (RMS normalization) which is compatible with causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Normalization layers (PixelNorm for causal compatibility)
        self.norm1 = PixelNorm(dim=1, eps=1e-6)
        self.norm2 = PixelNorm(dim=1, eps=1e-6)

        # Convolution layers
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3)

        # Skip connection for channel mismatch
        if in_channels != out_channels:
            self.skip = CausalConv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual block with normalization."""
        h = x

        # First block: norm -> silu -> conv
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)

        # Second block: norm -> silu -> conv
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)

        # Skip connection
        if self.skip is not None:
            x = self.skip(x)

        return x + h


class Upsample2d(nn.Module):
    """2D upsampling with conv and causal axis handling.

    After upsampling and conv, drops the first element along the causal axis
    to undo encoder's padding, matching PyTorch behavior.
    """

    def __init__(self, channels: int, causality_axis: CausalityAxis = CausalityAxis.HEIGHT):
        super().__init__()
        self.conv = CausalConv2d(channels, channels, kernel_size=3, causality_axis=causality_axis)
        self.causality_axis = causality_axis

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample by 2x."""
        b, c, h, w = x.shape

        # Nearest neighbor upsample 2x
        x = x[:, :, :, None, :, None]  # (B, C, H, 1, W, 1)
        x = mx.broadcast_to(x, (b, c, h, 2, w, 2))
        x = x.reshape(b, c, h * 2, w * 2)

        x = self.conv(x)

        # Drop first element along causal axis to undo encoder padding
        # This keeps the output length as 1 + 2*n instead of 2 + 2*n
        if self.causality_axis == CausalityAxis.HEIGHT:
            x = x[:, :, 1:, :]  # Drop first row
        elif self.causality_axis == CausalityAxis.WIDTH:
            x = x[:, :, :, 1:]  # Drop first column
        # CausalityAxis.NONE: no drop needed

        return x




class AudioDecoder(nn.Module):
    """
    Audio VAE Decoder - reconstructs mel spectrograms from latent representations.

    Architecture (from LTX-2 checkpoint inspection):
    - conv_in: z_channels (8) -> base_channels (512)
    - Mid block: 2x SimpleResBlock (no attention, no norms)
    - Upsampling path: 3 levels
      - level 2: 512 -> 512, then upsample
      - level 1: 512 -> 256, then upsample
      - level 0: 256 -> 128 (no upsample)
    - conv_out: 128 -> 2 (stereo)

    Input: (B, 8, frames, mel_bins) - latent representation
    Output: (B, 2, frames*4, mel_bins) - stereo mel spectrogram
    """

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 2,
        ch_mult: Tuple[int, ...] = (1, 2, 4),  # 3 levels
        num_res_blocks: int = 3,  # ResBlocks per level (from checkpoint)
        z_channels: int = 8,
        mel_bins: int = 16,  # Latent mel bins (64/4 = 16)
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.z_channels = z_channels
        self.mel_bins = mel_bins
        self.is_causal = is_causal
        self.compute_dtype = compute_dtype

        # Per-channel statistics for denormalizing latents (patchified: ch = z_channels * mel_bins)
        # PyTorch AudioDecoder uses ch (128) for the stats, which equals z_channels * mel_bins
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)

        # Patchifier for denormalization pipeline (patchify → denormalize → unpatchify)
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        # Base block channels (highest level = ch * ch_mult[-1])
        base_block_channels = ch * ch_mult[-1]  # 128 * 4 = 512

        # Input conv: z_channels -> base_block_channels
        self.conv_in = CausalConv2d(z_channels, base_block_channels, kernel_size=3)

        # Mid block: 2 ResBlocks at base_block_channels
        self.mid_block_1 = SimpleResBlock2d(base_block_channels, base_block_channels)
        self.mid_block_2 = SimpleResBlock2d(base_block_channels, base_block_channels)

        # Upsampling path (in reverse order of ch_mult)
        # Build: level 2 (512), level 1 (256), level 0 (128)
        self.up_blocks: List[dict] = []
        block_in = base_block_channels

        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]

            # ResBlocks for this level
            res_blocks = []
            for _ in range(num_res_blocks):
                res_blocks.append(SimpleResBlock2d(block_in, block_out))
                block_in = block_out

            # Upsample (except for level 0)
            upsample = Upsample2d(block_out) if i_level != 0 else None

            self.up_blocks.append({
                "res_blocks": res_blocks,
                "upsample": upsample,
            })

        # Output normalization and conv: ch -> out_ch
        # PixelNorm before final activation (matches PyTorch norm_out)
        self.norm_out = PixelNorm(dim=1, eps=1e-6)
        self.conv_out = CausalConv2d(ch, out_ch, kernel_size=3)

    def _denormalize_latents(self, sample: mx.array) -> mx.array:
        """
        Denormalize decoder latents using per-channel statistics.

        Follows PyTorch pipeline: patchify → denormalize → unpatchify.
        """
        # Build shape info for patchifier
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[1],
            frames=sample.shape[2],
            mel_bins=sample.shape[3],
        )

        # PyTorch pipeline: patchify → denormalize → unpatchify
        # Patchify: (B, C, T, F) -> (B, T, C*F)
        sample_patched = self.patchifier.patchify(sample)

        # Denormalize in patchified space
        sample_denormalized = self.per_channel_statistics.denormalize(sample_patched)

        # Unpatchify back to 4D: (B, T, C*F) -> (B, C, T, F)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        return sample

    def __call__(self, sample: mx.array) -> mx.array:
        """
        Decode latent to mel spectrogram.

        Args:
            sample: Latent tensor (B, z_channels, frames, mel_bins)

        Returns:
            Mel spectrogram (B, out_ch, frames*4, mel_bins)
        """
        # Cast to compute dtype
        if self.compute_dtype != mx.float32:
            sample = sample.astype(self.compute_dtype)

        # Denormalize latents first (matching PyTorch)
        sample = self._denormalize_latents(sample)

        # Get input dimensions
        _b, _c, t, f = sample.shape

        # Store target shape for output adjustment
        target_frames = t * LATENT_DOWNSAMPLE_FACTOR
        # Adjust for causal padding (matches PyTorch CausalityAxis handling)
        if self.is_causal:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        # Conv in
        h = self.conv_in(sample)
        mx.eval(h)

        # Mid block
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        mx.eval(h)

        # Upsampling path
        for level in self.up_blocks:
            for res_block in level["res_blocks"]:
                h = res_block(h)

            if level["upsample"] is not None:
                h = level["upsample"](h)
            mx.eval(h)

        # Output: norm -> activation -> conv (matches PyTorch _finalize_output)
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)

        # Adjust output shape to target (frames are upsampled 4x, mel bins are upsampled 4x)
        # Output shape: (B, out_ch, frames_upsampled, mel_bins_upsampled)
        target_mel = f * LATENT_DOWNSAMPLE_FACTOR  # 16 * 4 = 64
        h = h[:, :self.out_ch, :target_frames, :target_mel]

        # Cast back to float32
        if self.compute_dtype != mx.float32:
            h = h.astype(mx.float32)

        return h


def load_audio_decoder_weights(decoder: AudioDecoder, weights_path: str) -> None:
    """
    Load Audio VAE decoder weights from safetensors file.

    Args:
        decoder: AudioDecoder instance to load weights into.
        weights_path: Path to safetensors file.
    """
    from safetensors import safe_open
    import torch

    print(f"Loading Audio VAE decoder weights from {weights_path}...")
    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        keys = f.keys()

        # Check if audio VAE weights exist
        audio_keys = [k for k in keys if k.startswith("audio_vae.")]
        if not audio_keys:
            print("  Warning: No audio VAE weights found in checkpoint")
            return

        # Load conv_in
        _load_conv_weights(f, "audio_vae.decoder.conv_in.conv", decoder.conv_in, keys)
        loaded_count += 1

        # Load mid block
        _load_simple_resblock_weights(f, "audio_vae.decoder.mid.block_1", decoder.mid_block_1, keys)
        _load_simple_resblock_weights(f, "audio_vae.decoder.mid.block_2", decoder.mid_block_2, keys)
        loaded_count += 2

        # Load upsampling blocks
        for i_level, level_blocks in enumerate(decoder.up_blocks):
            # Map to PyTorch level indexing (reversed)
            pt_level = decoder.num_resolutions - 1 - i_level

            for i_block, res_block in enumerate(level_blocks["res_blocks"]):
                prefix = f"audio_vae.decoder.up.{pt_level}.block.{i_block}"
                _load_simple_resblock_weights(f, prefix, res_block, keys)
                loaded_count += 1

            if level_blocks["upsample"] is not None:
                # Checkpoint has double .conv: audio_vae.decoder.up.X.upsample.conv.conv.weight
                prefix = f"audio_vae.decoder.up.{pt_level}.upsample.conv.conv"
                _load_conv_weights(f, prefix, level_blocks["upsample"].conv, keys)
                loaded_count += 1

        # Load conv_out
        _load_conv_weights(f, "audio_vae.decoder.conv_out.conv", decoder.conv_out, keys)
        loaded_count += 1

        # Load per-channel statistics
        # PyTorch uses hyphenated names: mean-of-means, std-of-means
        # Note: The checkpoint stores these at audio_vae.per_channel_statistics (not audio_vae.decoder.)
        mean_key = "audio_vae.per_channel_statistics.mean-of-means"
        std_key = "audio_vae.per_channel_statistics.std-of-means"

        if mean_key in keys:
            import torch
            mean = f.get_tensor(mean_key)
            if mean.dtype == torch.bfloat16:
                mean = mean.to(torch.float32)
            decoder.per_channel_statistics.mean_of_means = mx.array(mean.numpy())
            print(f"    Loaded decoder mean-of-means: shape={mean.shape}, mean={float(mean.mean()):.4f}")
            loaded_count += 1

        if std_key in keys:
            import torch
            std = f.get_tensor(std_key)
            if std.dtype == torch.bfloat16:
                std = std.to(torch.float32)
            decoder.per_channel_statistics.std_of_means = mx.array(std.numpy())
            print(f"    Loaded decoder std-of-means: shape={std.shape}, mean={float(std.mean()):.4f}")
            loaded_count += 1

    print(f"  Loaded {loaded_count} audio decoder weight tensors")


def _load_conv_weights(f, prefix: str, conv: CausalConv2d, keys) -> None:
    """Load weights for a CausalConv2d layer."""
    import torch

    for suffix in ["weight", "bias"]:
        pt_key = f"{prefix}.{suffix}"
        if pt_key in keys:
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            value = tensor.numpy()

            if suffix == "weight":
                # PyTorch: (out, in, kH, kW) -> MLX: (out, kH, kW, in)
                value = value.transpose(0, 2, 3, 1)
                conv.weight = mx.array(value)
            else:
                conv.bias = mx.array(value)


def _load_simple_resblock_weights(f, prefix: str, block: SimpleResBlock2d, keys) -> None:
    """Load weights for a SimpleResBlock2d."""
    _load_conv_weights(f, f"{prefix}.conv1.conv", block.conv1, keys)
    _load_conv_weights(f, f"{prefix}.conv2.conv", block.conv2, keys)

    if block.skip is not None:
        _load_conv_weights(f, f"{prefix}.nin_shortcut.conv", block.skip, keys)
