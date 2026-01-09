"""HiFi-GAN Vocoder for LTX-2 MLX."""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


LRELU_SLOPE = 0.1


class Conv1d(nn.Module):
    """1D convolution with dilation support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight shape: (out_channels, kernel_size, in_channels) for MLX
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 1D convolution.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, out_C, T')
        """
        b, c, t = x.shape

        # Calculate effective kernel size with dilation
        effective_k = (self.kernel_size - 1) * self.dilation + 1

        # Apply padding
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, self.padding)])

        # Handle dilation by inserting zeros into kernel
        if self.dilation > 1:
            # Create dilated kernel
            dilated_k = effective_k
            dilated_weight = mx.zeros((self.out_channels, dilated_k, self.in_channels))
            for i in range(self.kernel_size):
                dilated_weight[:, i * self.dilation, :] = self.weight[:, i, :]
            weight = dilated_weight
        else:
            weight = self.weight

        # Transpose for MLX conv1d: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)

        # MLX conv1d expects weight: (out_C, kW, in_C)
        out = mx.conv1d(x, weight, stride=self.stride)

        # Transpose back: (B, T, C) -> (B, C, T)
        out = out.transpose(0, 2, 1)

        # Add bias
        out = out + self.bias[None, :, None]

        return out


class ConvTranspose1d(nn.Module):
    """1D transposed convolution for upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight shape: (out_channels, kernel_size, in_channels) for MLX conv_transpose1d
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply transposed 1D convolution.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, out_C, T')
        """
        b, c, t = x.shape

        # Transpose for MLX: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)

        # Use conv_transpose1d
        out = mx.conv_transpose1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
        )

        # Transpose back: (B, T, C) -> (B, C, T)
        out = out.transpose(0, 2, 1)

        # Add bias
        out = out + self.bias[None, :, None]

        return out


class ResBlock1(nn.Module):
    """HiFi-GAN residual block type 1 with multiple dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.channels = channels
        self.dilations = dilations

        # Two sets of convolutions (convs1 and convs2) per dilation
        self.convs1 = []
        self.convs2 = []

        for d in dilations:
            # Calculate padding for same output size
            pad = (kernel_size - 1) * d // 2
            self.convs1.append(
                Conv1d(channels, channels, kernel_size, padding=pad, dilation=d)
            )
            self.convs2.append(
                Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
            )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual block."""
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, negative_slope=LRELU_SLOPE)
            xt = conv1(xt)
            xt = nn.leaky_relu(xt, negative_slope=LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x


class Vocoder(nn.Module):
    """
    HiFi-GAN Vocoder for converting mel spectrograms to audio waveforms.

    Architecture:
    - conv_pre: Initial 1D convolution
    - Upsampling stages with ConvTranspose1d
    - Multi-receptive field fusion (ResBlocks with different kernel sizes)
    - conv_post: Final 1D convolution with tanh

    Input: Mel spectrogram (B, 2, T, 64) for stereo
    Output: Audio waveform (B, 2, audio_samples) at 24kHz
    """

    def __init__(
        self,
        resblock_kernel_sizes: Optional[List[int]] = None,
        upsample_rates: Optional[List[int]] = None,
        upsample_kernel_sizes: Optional[List[int]] = None,
        resblock_dilation_sizes: Optional[List[List[int]]] = None,
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        output_sample_rate: int = 24000,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()

        # Default values matching LTX-2 config
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.compute_dtype = compute_dtype

        # Input channels: stereo mel = 128 (2 channels × 64 mel bins)
        in_channels = 128 if stereo else 64

        # Initial conv
        self.conv_pre = Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        # Upsampling layers
        self.ups = []
        for i, (rate, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            padding = (k - rate) // 2
            self.ups.append(ConvTranspose1d(in_ch, out_ch, k, rate, padding))

        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, tuple(dilations)))

        # Output conv
        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = Conv1d(final_channels, out_channels, 7, padding=3)

        # Calculate upsample factor
        self.upsample_factor = math.prod(upsample_rates)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            x: Mel spectrogram (B, 2, T, mel_bins) for stereo

        Returns:
            Audio waveform (B, 2, audio_length)
        """
        # Cast to compute dtype
        if self.compute_dtype != mx.float32:
            x = x.astype(self.compute_dtype)

        # Transpose: (B, channels, time, mel_bins) -> (B, channels, mel_bins, time)
        x = x.transpose(0, 1, 3, 2)

        # For stereo: (B, 2, mel_bins, time) -> (B, 2*mel_bins, time)
        b, s, m, t = x.shape
        x = x.reshape(b, s * m, t)

        # Initial conv
        x = self.conv_pre(x)

        # Upsampling with residual blocks
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=LRELU_SLOPE)
            x = self.ups[i](x)

            # Multi-receptive field fusion
            start_idx = i * self.num_kernels
            end_idx = start_idx + self.num_kernels

            # Compute all resblock outputs
            block_outputs = []
            for idx in range(start_idx, end_idx):
                block_outputs.append(self.resblocks[idx](x))

            # Average the outputs
            x = mx.stack(block_outputs, axis=0).mean(axis=0)

            mx.eval(x)

        # Output
        x = nn.leaky_relu(x, negative_slope=LRELU_SLOPE)
        x = self.conv_post(x)
        x = mx.tanh(x)

        # Cast back to float32
        if self.compute_dtype != mx.float32:
            x = x.astype(mx.float32)

        return x


def load_vocoder_weights(vocoder: Vocoder, weights_path: str) -> None:
    """
    Load Vocoder weights from safetensors file.

    Args:
        vocoder: Vocoder instance to load weights into.
        weights_path: Path to safetensors file.
    """
    from safetensors import safe_open
    import torch

    print(f"Loading Vocoder weights from {weights_path}...")
    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        keys = f.keys()

        # Check if vocoder weights exist
        vocoder_keys = [k for k in keys if k.startswith("vocoder.")]
        if not vocoder_keys:
            print("  Warning: No vocoder weights found in checkpoint")
            return

        # Load conv_pre
        _load_conv1d_weights(f, "vocoder.conv_pre", vocoder.conv_pre, keys)
        loaded_count += 1

        # Load upsampling layers
        for i, up in enumerate(vocoder.ups):
            _load_conv_transpose1d_weights(f, f"vocoder.ups.{i}", up, keys)
            loaded_count += 1

        # Load resblocks
        for i, block in enumerate(vocoder.resblocks):
            prefix = f"vocoder.resblocks.{i}"
            for j, conv in enumerate(block.convs1):
                _load_conv1d_weights(f, f"{prefix}.convs1.{j}", conv, keys)
                loaded_count += 1
            for j, conv in enumerate(block.convs2):
                _load_conv1d_weights(f, f"{prefix}.convs2.{j}", conv, keys)
                loaded_count += 1

        # Load conv_post
        _load_conv1d_weights(f, "vocoder.conv_post", vocoder.conv_post, keys)
        loaded_count += 1

    print(f"  Loaded {loaded_count} vocoder weight tensors")


def _load_conv1d_weights(f, prefix: str, conv: Conv1d, keys) -> None:
    """Load weights for a Conv1d layer."""
    import torch

    for suffix in ["weight", "bias"]:
        pt_key = f"{prefix}.{suffix}"
        if pt_key in keys:
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            value = tensor.numpy()

            if suffix == "weight":
                # PyTorch: (out, in, k) -> MLX: (out, k, in)
                value = value.transpose(0, 2, 1)
                conv.weight = mx.array(value)
            else:
                conv.bias = mx.array(value)


def _load_conv_transpose1d_weights(f, prefix: str, conv: ConvTranspose1d, keys) -> None:
    """Load weights for a ConvTranspose1d layer."""
    import torch

    for suffix in ["weight", "bias"]:
        pt_key = f"{prefix}.{suffix}"
        if pt_key in keys:
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            value = tensor.numpy()

            if suffix == "weight":
                # PyTorch transpose: (in, out, k) -> MLX: (out, k, in)
                value = value.transpose(1, 2, 0)
                conv.weight = mx.array(value)
            else:
                conv.bias = mx.array(value)
