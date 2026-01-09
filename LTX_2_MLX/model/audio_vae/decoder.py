"""Audio VAE Decoder for LTX-2 MLX."""

from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


LATENT_DOWNSAMPLE_FACTOR = 4


class CausalConv2d(nn.Module):
    """2D causal convolution with padding along time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal

        # MLX conv2d weight shape: (out_C, kH, kW, in_C)
        self.weight = mx.zeros((out_channels, kernel_size, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply causal 2D convolution."""
        # x shape: (B, C, H, W) where H is frequency/mel_bins, W is time
        k = self.kernel_size
        s = self.stride

        # Calculate padding
        pad_h = (k - 1) // 2
        pad_w = k - 1 if self.causal else (k - 1) // 2

        # Apply padding
        if self.causal:
            # Causal: pad on the left (past) only for time dimension (W)
            x = mx.pad(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, 0)])
        else:
            # Non-causal: symmetric padding
            x = mx.pad(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])

        # Transpose for MLX conv2d: (B, C, H, W) -> (B, H, W, C)
        x = x.transpose(0, 2, 3, 1)

        # Apply conv
        out = mx.conv2d(x, self.weight, stride=s)

        # Transpose back: (B, H, W, C) -> (B, C, H, W)
        out = out.transpose(0, 3, 1, 2)

        # Add bias
        out = out + self.bias[None, :, None, None]

        return out


class SimpleResBlock2d(nn.Module):
    """
    Simple 2D residual block with just conv1 and conv2 (no norms).

    Matches the LTX-2 checkpoint structure.
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

        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3)

        if in_channels != out_channels:
            self.skip = CausalConv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual block."""
        h = x

        h = nn.silu(h)
        h = self.conv1(h)

        h = nn.silu(h)
        h = self.conv2(h)

        if self.skip is not None:
            x = self.skip(x)

        return x + h


class Upsample2d(nn.Module):
    """2D upsampling with conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv2d(channels, channels, kernel_size=3)

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample by 2x."""
        b, c, h, w = x.shape

        # Nearest neighbor upsample 2x
        x = x[:, :, :, None, :, None]  # (B, C, H, 1, W, 1)
        x = mx.broadcast_to(x, (b, c, h, 2, w, 2))
        x = x.reshape(b, c, h * 2, w * 2)

        x = self.conv(x)
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
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.z_channels = z_channels
        self.compute_dtype = compute_dtype

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

        # Output conv: ch -> out_ch
        self.conv_out = CausalConv2d(ch, out_ch, kernel_size=3)

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

        # Get input dimensions
        _b, _c, t, f = sample.shape

        # Store target shape for output adjustment
        target_frames = t * LATENT_DOWNSAMPLE_FACTOR
        # Adjust for causal padding
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

        # Output (with activation before conv)
        h = nn.silu(h)
        h = self.conv_out(h)

        # Adjust output shape to target
        h = h[:, :self.out_ch, :target_frames, :f]

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
                prefix = f"audio_vae.decoder.up.{pt_level}.upsample.conv"
                _load_conv_weights(f, prefix, level_blocks["upsample"].conv, keys)
                loaded_count += 1

        # Load conv_out
        _load_conv_weights(f, "audio_vae.decoder.conv_out.conv", decoder.conv_out, keys)
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
