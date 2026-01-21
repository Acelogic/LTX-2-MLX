"""Audio VAE Encoder for LTX-2 MLX.

Encodes audio spectrograms into latent representations.
"""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from LTX_2_MLX.components.patchifiers import AudioPatchifier
from LTX_2_MLX.types import AudioLatentShape

from .decoder import (
    CausalConv2d,
    CausalityAxis,
    PerChannelStatistics,
    SimpleResBlock2d,
    LATENT_DOWNSAMPLE_FACTOR,
)


class Downsample2d(nn.Module):
    """2D downsampling with strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        # Strided conv for downsampling (stride=2)
        self.conv = CausalConv2d(channels, channels, kernel_size=3, stride=2)

    def __call__(self, x: mx.array) -> mx.array:
        """Downsample by 2x."""
        return self.conv(x)


class AudioEncoder(nn.Module):
    """
    Audio VAE Encoder - compresses mel spectrograms to latent representations.

    Architecture mirrors AudioDecoder but in reverse:
    - conv_in: in_channels (2) -> base_channels (128)
    - Downsampling path: 3 levels
      - level 0: 128 -> 256, then downsample
      - level 1: 256 -> 512, then downsample
      - level 2: 512 -> 512 (no downsample)
    - Mid block: 2x SimpleResBlock
    - conv_out: 512 -> z_channels*2 (8*2=16 for mean/logvar)

    Input: (B, 2, time, mel_bins) - stereo mel spectrogram
    Output: (B, 8, frames, mel_bins/4) - latent representation
    """

    def __init__(
        self,
        ch: int = 128,
        in_ch: int = 2,  # Stereo audio
        ch_mult: Tuple[int, ...] = (1, 2, 4),  # 3 levels
        num_res_blocks: int = 3,
        z_channels: int = 8,
        mel_bins: int = 16,  # Latent mel bins (64/4 = 16)
        double_z: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.ch = ch
        self.in_ch = in_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.z_channels = z_channels
        self.mel_bins = mel_bins
        self.double_z = double_z
        self.compute_dtype = compute_dtype

        # Per-channel statistics for normalization (patchified: ch = z_channels * mel_bins)
        # PyTorch AudioEncoder uses ch (128) for the stats, which equals z_channels * mel_bins
        self.per_channel_statistics = PerChannelStatistics(ch)

        # Patchifier for normalization pipeline (patchify → normalize → unpatchify)
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        # Input conv: in_ch -> base channels
        self.conv_in = CausalConv2d(in_ch, ch, kernel_size=3)

        # Downsampling path (forward order of ch_mult)
        # Build: level 0 (128), level 1 (256), level 2 (512)
        self.down_blocks = []
        block_in = ch

        for i_level in range(self.num_resolutions):
            block_out = ch * ch_mult[i_level]

            # ResBlocks for this level
            res_blocks = []
            for _ in range(num_res_blocks):
                res_blocks.append(SimpleResBlock2d(block_in, block_out))
                block_in = block_out

            # Downsample (except for last level)
            downsample = Downsample2d(block_out) if i_level != self.num_resolutions - 1 else None

            self.down_blocks.append({
                "res_blocks": res_blocks,
                "downsample": downsample,
            })

        # Base block channels (highest level = ch * ch_mult[-1])
        base_block_channels = ch * ch_mult[-1]  # 128 * 4 = 512

        # Mid block: 2 ResBlocks at base_block_channels
        self.mid_block_1 = SimpleResBlock2d(base_block_channels, base_block_channels)
        self.mid_block_2 = SimpleResBlock2d(base_block_channels, base_block_channels)

        # Output conv: base_channels -> z_channels*2 (if double_z) or z_channels
        out_channels = z_channels * 2 if double_z else z_channels
        self.conv_out = CausalConv2d(base_block_channels, out_channels, kernel_size=3)

    def __call__(self, spectrogram: mx.array) -> mx.array:
        """
        Encode mel spectrogram to latent.

        Args:
            spectrogram: Mel spectrogram (B, in_ch, time, mel_bins)

        Returns:
            Latent tensor (B, z_channels, frames, mel_bins/4)
        """
        # Cast to compute dtype
        if self.compute_dtype != mx.float32:
            spectrogram = spectrogram.astype(self.compute_dtype)

        # Conv in
        h = self.conv_in(spectrogram)
        mx.eval(h)

        # Downsampling path
        for level in self.down_blocks:
            for res_block in level["res_blocks"]:
                h = res_block(h)

            if level["downsample"] is not None:
                h = level["downsample"](h)
            mx.eval(h)

        # Mid block
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        mx.eval(h)

        # Output (with activation before conv)
        h = nn.silu(h)
        h = self.conv_out(h)

        # Normalize latents
        h = self._normalize_latents(h)

        # Cast back to float32
        if self.compute_dtype != mx.float32:
            h = h.astype(mx.float32)

        return h

    def _normalize_latents(self, latent_output: mx.array) -> mx.array:
        """
        Normalize encoder latents using per-channel statistics.

        Follows PyTorch pipeline: patchify → normalize → unpatchify.
        When double_z=True, we only normalize the mean part (first half).
        """
        # Extract mean latent when double_z
        if self.double_z:
            mean_latent = latent_output[:, :self.z_channels, :, :]
        else:
            mean_latent = latent_output

        # Build shape info for patchifier
        latent_shape = AudioLatentShape(
            batch=mean_latent.shape[0],
            channels=mean_latent.shape[1],
            frames=mean_latent.shape[2],
            mel_bins=mean_latent.shape[3],
        )

        # PyTorch pipeline: patchify → normalize → unpatchify
        # Patchify: (B, C, T, F) -> (B, T, C*F)
        latent_patched = self.patchifier.patchify(mean_latent)

        # Normalize in patchified space
        latent_normalized = self.per_channel_statistics.normalize(latent_patched)

        # Unpatchify back to 4D: (B, T, C*F) -> (B, C, T, F)
        latent_output = self.patchifier.unpatchify(latent_normalized, latent_shape)

        return latent_output


def load_audio_encoder_weights(encoder: AudioEncoder, weights_path: str) -> None:
    """
    Load Audio VAE encoder weights from safetensors file.

    Args:
        encoder: AudioEncoder model to load weights into.
        weights_path: Path to safetensors file.
    """
    from safetensors import safe_open
    import torch as pt

    print(f"  Loading audio encoder weights from {weights_path}...")

    # Weight mapping from PyTorch to MLX
    # PyTorch: audio_vae.encoder.*
    # MLX conv weight shape: (out_C, kH, kW, in_C) vs PyTorch (out_C, in_C, kH, kW)

    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())

        # Filter to audio encoder keys
        encoder_keys = [k for k in keys if k.startswith("audio_vae.encoder.")]

        if not encoder_keys:
            print("  Warning: No audio encoder keys found in weights file")
            return

        # Load conv_in
        _load_conv_weights(encoder.conv_in, f, "audio_vae.encoder.conv_in")

        # Load down blocks
        for i_level, level in enumerate(encoder.down_blocks):
            for i_block, res_block in enumerate(level["res_blocks"]):
                prefix = f"audio_vae.encoder.down.{i_level}.block.{i_block}"
                _load_resblock_weights(res_block, f, prefix)

            if level["downsample"] is not None:
                prefix = f"audio_vae.encoder.down.{i_level}.downsample"
                _load_conv_weights(level["downsample"].conv, f, f"{prefix}.conv")

        # Load mid blocks
        _load_resblock_weights(encoder.mid_block_1, f, "audio_vae.encoder.mid.block_1")
        _load_resblock_weights(encoder.mid_block_2, f, "audio_vae.encoder.mid.block_2")

        # Load conv_out
        _load_conv_weights(encoder.conv_out, f, "audio_vae.encoder.conv_out")

        # Load per-channel statistics
        # PyTorch uses hyphenated names: mean-of-means, std-of-means
        mean_key = "audio_vae.encoder.per_channel_statistics.mean-of-means"
        std_key = "audio_vae.encoder.per_channel_statistics.std-of-means"

        if mean_key in keys:
            mean = f.get_tensor(mean_key)
            if mean.dtype == pt.bfloat16:
                mean = mean.to(pt.float32)
            encoder.per_channel_statistics.mean_of_means = mx.array(mean.numpy())
            print(f"    Loaded mean-of-means: shape={mean.shape}, mean={float(mean.mean()):.4f}")

        if std_key in keys:
            std = f.get_tensor(std_key)
            if std.dtype == pt.bfloat16:
                std = std.to(pt.float32)
            encoder.per_channel_statistics.std_of_means = mx.array(std.numpy())
            print(f"    Loaded std-of-means: shape={std.shape}, mean={float(std.mean()):.4f}")

    mx.eval(encoder.parameters())
    print("  Audio encoder weights loaded successfully")


def _load_conv_weights(conv: CausalConv2d, f, prefix: str) -> None:
    """Load conv weights with shape transposition."""
    import torch

    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"

    if weight_key in list(f.keys()):
        w = f.get_tensor(weight_key)
        # Transpose: PyTorch (out_C, in_C, kH, kW) -> MLX (out_C, kH, kW, in_C)
        w = w.permute(0, 2, 3, 1)
        if w.dtype == torch.bfloat16:
            w = w.to(torch.float32)
        conv.weight = mx.array(w.numpy())

    if bias_key in list(f.keys()):
        b = f.get_tensor(bias_key)
        if b.dtype == torch.bfloat16:
            b = b.to(torch.float32)
        conv.bias = mx.array(b.numpy())


def _load_resblock_weights(block: SimpleResBlock2d, f, prefix: str) -> None:
    """Load ResBlock weights."""
    _load_conv_weights(block.conv1, f, f"{prefix}.conv1")
    _load_conv_weights(block.conv2, f, f"{prefix}.conv2")
    if block.skip is not None:
        _load_conv_weights(block.skip, f, f"{prefix}.nin_shortcut")


def encode_audio(
    spectrogram: mx.array,
    encoder: AudioEncoder,
) -> mx.array:
    """
    Encode audio spectrogram to latent representation.

    Args:
        spectrogram: Mel spectrogram (B, 2, time, mel_bins)
        encoder: AudioEncoder model.

    Returns:
        Latent representation (B, z_channels, frames, mel_bins/4)
    """
    return encoder(spectrogram)
