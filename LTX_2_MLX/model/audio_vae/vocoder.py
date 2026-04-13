"""HiFi-GAN Vocoder for LTX-2 MLX."""

import math
from typing import List, Optional, Tuple

import numpy as np
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
        # Apply padding
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, self.padding)])

        # Transpose for MLX conv1d: (B, C, T) -> (B, T, C)
        x = x.transpose(0, 2, 1)

        # MLX conv1d natively supports dilation
        out = mx.conv1d(x, self.weight, stride=self.stride, dilation=self.dilation)

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


# ---------------------------------------------------------------------------
# Anti-aliased resampling helpers (kaiser-sinc filters) for BigVGAN v2
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """Snake activation with separate alpha and beta parameters (log-scale).

    Forward: x + (1 / exp(beta)) * sin(x * exp(alpha))^2
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((in_features,))
        self.beta = mx.zeros((in_features,))
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha[None, :, None])
        beta = mx.exp(self.beta[None, :, None])
        return x + (1.0 / (beta + self.eps)) * mx.power(mx.sin(x * alpha), 2)


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> mx.array:
    """Compute a kaiser-windowed sinc filter using numpy, return as mx.array.

    Returns shape (1, 1, kernel_size).
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Compute kaiser beta from amplitude
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0

    window = np.kaiser(kernel_size, beta)

    if even:
        time = np.arange(-half_size, half_size) + 0.5
    else:
        time = np.arange(kernel_size) - half_size

    if cutoff == 0:
        filter_ = np.zeros_like(time)
    else:
        x = 2 * cutoff * time
        safe_denom = np.where(x == 0, 1.0, np.pi * x)
        sinc = np.where(x == 0, 1.0, np.sin(np.pi * x) / safe_denom)
        filter_ = 2 * cutoff * window * sinc
        filter_ /= filter_.sum()

    return mx.array(filter_.reshape(1, 1, kernel_size).astype(np.float32))


def _replicate_pad_1d(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """Replicate (edge) padding for 1D signal in (B, C, T) format."""
    parts = []
    if pad_left > 0:
        parts.append(mx.repeat(x[:, :, :1], pad_left, axis=2))
    parts.append(x)
    if pad_right > 0:
        parts.append(mx.repeat(x[:, :, -1:], pad_right, axis=2))
    return mx.concatenate(parts, axis=2)


def _depthwise_conv1d(
    x: mx.array, filt: mx.array, stride: int = 1
) -> mx.array:
    """Depthwise 1D convolution.

    MLX conv1d does not support groups, so we reshape to (B*C, 1, T),
    convolve with a (1, K, 1) filter, and reshape back.

    Args:
        x: Input (B, C, T).
        filt: Filter (1, 1, K) — will be transposed for MLX.
        stride: Convolution stride.
    Returns:
        Output (B, C, T').
    """
    b, c, t = x.shape
    # (B, C, T) -> (B*C, T, 1) for MLX conv1d (expects B, T, C)
    x_flat = x.reshape(b * c, t, 1)
    # Filter: (1, 1, K) -> MLX weight (out_c=1, K, in_c=1) -> need (1, K, 1) for MLX
    # MLX conv1d weight shape: (out_channels, kernel_width, in_channels)
    k = filt.shape[2]
    w = filt.reshape(1, k, 1)
    out = mx.conv1d(x_flat, w, stride=stride)
    # (B*C, T', 1) -> (B, C, T')
    return out.reshape(b, c, -1)


def _depthwise_conv_transpose1d(
    x: mx.array, filt: mx.array, stride: int = 1
) -> mx.array:
    """Depthwise transposed 1D convolution.

    Args:
        x: Input (B, C, T).
        filt: Filter (1, 1, K).
        stride: Transposed conv stride.
    Returns:
        Output (B, C, T').
    """
    b, c, t = x.shape
    # (B, C, T) -> (B*C, T, 1)
    x_flat = x.reshape(b * c, t, 1)
    k = filt.shape[2]
    w = filt.reshape(1, k, 1)
    out = mx.conv_transpose1d(x_flat, w, stride=stride)
    return out.reshape(b, c, -1)


class LowPassFilter1d(nn.Module):
    """Low-pass filter using depthwise conv1d with a kaiser sinc kernel."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.do_padding = padding
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        if self.do_padding:
            x = _replicate_pad_1d(x, self.pad_left, self.pad_right)
        return _depthwise_conv1d(x, self.filter, stride=self.stride)


class UpSample1d(nn.Module):
    """Anti-aliased upsampling using transposed depthwise conv with sinc filter."""

    def __init__(
        self,
        ratio: int = 2,
        kernel_size: Optional[int] = None,
        window_type: str = "kaiser",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            # Hann-windowed sinc filter (matches torchaudio.functional.resample)
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio

            time_axis = np.arange(self.kernel_size) / ratio - width
            time_axis_rolloff = time_axis * rolloff
            time_clamped = np.clip(
                time_axis_rolloff, -lowpass_filter_width, lowpass_filter_width
            )
            window = np.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            # Use safe division to avoid RuntimeWarning: invalid value in divide
            # (np.where evaluates both branches before masking)
            safe_denom = np.where(time_axis_rolloff == 0, 1.0, np.pi * time_axis_rolloff)
            sinc_vals = np.where(
                time_axis_rolloff == 0,
                1.0,
                np.sin(np.pi * time_axis_rolloff) / safe_denom,
            )
            sinc_filter = (sinc_vals * window * rolloff / ratio).reshape(1, 1, -1)
        else:
            # Kaiser-windowed sinc filter (BigVGAN default)
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (
                self.kernel_size - self.stride
            ) // 2
            self.pad_right = self.pad * self.stride + (
                self.kernel_size - self.stride + 1
            ) // 2
            sinc_filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )
            # kaiser_sinc_filter1d returns mx.array; convert to numpy for consistency
            sinc_filter = np.array(sinc_filter).reshape(1, 1, -1)

        self.filter = mx.array(sinc_filter.astype(np.float32))

    def __call__(self, x: mx.array) -> mx.array:
        x = _replicate_pad_1d(x, self.pad, self.pad)
        x = self.ratio * _depthwise_conv_transpose1d(x, self.filter, stride=self.stride)
        return x[:, :, self.pad_left : x.shape[2] - self.pad_right]


class DownSample1d(nn.Module):
    """Anti-aliased downsampling wrapping LowPassFilter1d with stride."""

    def __init__(
        self, ratio: int = 2, kernel_size: Optional[int] = None
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Upsample -> activation -> downsample for anti-aliased nonlinearity."""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class AMPBlock1(nn.Module):
    """BigVGAN v2 residual block with anti-aliased SnakeBeta activations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5),
        activation: str = "snake",
    ) -> None:
        super().__init__()

        self.convs1 = []
        self.convs2 = []
        self.acts1 = []
        self.acts2 = []

        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.convs1.append(
                Conv1d(channels, channels, kernel_size, padding=pad, dilation=d)
            )
            self.convs2.append(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                )
            )
            self.acts1.append(Activation1d(SnakeBeta(channels)))
            self.acts2.append(Activation1d(SnakeBeta(channels)))

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, a1, a2 in zip(
            self.convs1, self.convs2, self.acts1, self.acts2
        ):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
            mx.eval(x)  # Prevent GPU watchdog timeout on long audio
        return x


class _STFTFn(nn.Module):
    """STFT implemented as conv1d with precomputed DFT bases.

    The forward_basis and inverse_basis buffers are loaded from the checkpoint.
    """

    def __init__(
        self, filter_length: int, hop_length: int, win_length: int
    ) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        # Buffers — shape (n_freqs*2, 1, filter_length); loaded from checkpoint
        self.forward_basis = mx.zeros((n_freqs * 2, 1, filter_length))
        self.inverse_basis = mx.zeros((n_freqs * 2, 1, filter_length))

    def __call__(
        self, y: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Compute magnitude and phase from waveform.

        Args:
            y: (B, T) waveform.
        Returns:
            magnitude: (B, n_freqs, T_frames)
            phase: (B, n_freqs, T_frames)
        """
        if y.ndim == 2:
            y = y[:, None, :]  # (B, 1, T)

        # Causal left-only padding
        left_pad = max(0, self.win_length - self.hop_length)
        if left_pad > 0:
            y = mx.pad(y, [(0, 0), (0, 0), (left_pad, 0)])

        # Conv1d with forward_basis: (n_freqs*2, 1, filter_length)
        # Input (B, 1, T) -> MLX (B, T, 1); weight (n_freqs*2, filter_length, 1)
        b, _, t = y.shape
        y_mlx = y.transpose(0, 2, 1)  # (B, T, 1)
        # Weight: PyTorch (out, in, k) -> MLX (out, k, in)
        w = self.forward_basis.transpose(0, 2, 1)  # (n_freqs*2, filter_length, 1) -> already good shape
        # Actually MLX weight shape is (out_channels, kernel_width, in_channels)
        # forward_basis is (n_freqs*2, 1, filter_length) in PyTorch = (out, in, k)
        # MLX needs (out, k, in) = (n_freqs*2, filter_length, 1)
        spec = mx.conv1d(y_mlx, w, stride=self.hop_length)
        mx.eval(spec)
        spec = spec.transpose(0, 2, 1)  # (B, n_freqs*2, T_frames)

        n_freqs = spec.shape[1] // 2
        real = spec[:, :n_freqs]
        imag = spec[:, n_freqs:]
        magnitude = mx.sqrt(real ** 2 + imag ** 2)
        phase = mx.arctan2(imag, real)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Log-mel spectrogram module with buffers loaded from checkpoint."""

    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel_channels, n_freqs))

    def mel_spectrogram(
        self, y: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute log-mel spectrogram.

        Args:
            y: Waveform (B, T).
        Returns:
            log_mel: (B, n_mel_channels, T_frames)
            magnitude: (B, n_freqs, T_frames)
            phase: (B, n_freqs, T_frames)
            energy: (B, T_frames)
        """
        magnitude, phase = self.stft_fn(y)
        energy = mx.sqrt((magnitude ** 2).sum(axis=1))
        # mel_basis: (n_mel, n_freqs), magnitude: (B, n_freqs, T_frames)
        # We want (B, n_mel, T_frames) = mel_basis @ magnitude per batch
        mel = mx.einsum("mf,bft->bmt", self.mel_basis, magnitude)
        log_mel = mx.log(mx.clip(mel, a_min=1e-5, a_max=None))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(nn.Module):
    """Vocoder with bandwidth extension (BWE) upsampling.

    Chains a mel-to-wav vocoder with a BWE module that upsamples the output
    to a higher sample rate.
    """

    def __init__(
        self,
        vocoder: "Vocoder",
        bwe_generator: "Vocoder",
        mel_stft: MelSTFT,
        input_sampling_rate: int,
        output_sampling_rate: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.output_sample_rate = output_sampling_rate
        self.resampler = UpSample1d(
            ratio=output_sampling_rate // input_sampling_rate,
            window_type="hann",
        )

    def _compute_mel(self, audio: mx.array) -> mx.array:
        """Compute log-mel spectrogram from waveform.

        Args:
            audio: (B, C, T) waveform.
        Returns:
            mel: (B, C, n_mels, T_frames).
        """
        b, n_channels, t = audio.shape
        flat = audio.reshape(b * n_channels, t)  # (B*C, T)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)  # (B*C, n_mels, T_frames)
        return mel.reshape(b, n_channels, mel.shape[1], mel.shape[2])

    def __call__(self, mel_spec: mx.array) -> mx.array:
        """Run vocoder + BWE forward pass.

        Runs in float32 regardless of input dtype.  bfloat16 accumulation
        errors compound through 108 sequential convolutions in the BigVGAN v2
        architecture and degrade spectral metrics by 40-90%.  fp32 eliminates
        this degradation.  (Matches upstream PyTorch autocast strategy.)

        Args:
            mel_spec: (B, 2, T, mel_bins) stereo mel spectrogram.
        Returns:
            Waveform (B, out_channels, T_out) clipped to [-1, 1].
        """
        input_dtype = mel_spec.dtype
        # Force fp32 for the entire vocoder + BWE chain.
        # bfloat16 accumulation errors degrade spectral metrics by 40-90%.
        mel_spec = mel_spec.astype(mx.float32)

        # Stage 1: Main vocoder (108 convolutions)
        x = self.vocoder(mel_spec)
        mx.eval(x)
        mx.clear_cache()

        _, _, length_low_rate = x.shape
        output_length = (
            length_low_rate
            * self.output_sampling_rate
            // self.input_sampling_rate
        )

        # Pad to multiple of hop_length
        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            pad_amount = self.hop_length - remainder
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_amount)])

        # Stage 2: Compute mel from vocoder output for BWE
        mel = self._compute_mel(x)  # (B, C, n_mels, T_frames)
        mx.eval(mel)
        mx.clear_cache()

        # Stage 3: BWE generator (another 108 convolutions) + resampler
        mel_for_bwe = mel.transpose(0, 1, 3, 2)  # (B, C, T_frames, mel_bins)
        del mel  # Free mel before BWE
        residual = self.bwe_generator(mel_for_bwe)
        mx.eval(residual)
        del mel_for_bwe
        mx.clear_cache()

        skip = self.resampler(x)
        mx.eval(skip)
        del x  # Free base waveform
        mx.clear_cache()

        result = mx.clip(residual + skip, -1, 1)[:, :, :output_length]
        mx.eval(result)
        return result.astype(input_dtype)


class Vocoder(nn.Module):
    """
    HiFi-GAN / BigVGAN v2 Vocoder for converting mel spectrograms to audio waveforms.

    Architecture:
    - conv_pre: Initial 1D convolution
    - Upsampling stages with ConvTranspose1d
    - Multi-receptive field fusion (ResBlocks with different kernel sizes)
    - conv_post: Final 1D convolution with optional tanh

    When resblock="1" (default): classic HiFi-GAN with leaky_relu activations.
    When resblock="AMP1": BigVGAN v2 with anti-aliased SnakeBeta activations.

    Input: Mel spectrogram (B, 2, T, 64) for stereo
    Output: Audio waveform (B, 2, audio_samples)
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
        resblock: str = "1",
        activation: str = "snake",
        apply_final_activation: bool = True,
        use_tanh_at_final: bool = True,
        use_bias_at_final: bool = True,
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
        self.is_amp = resblock == "AMP1"
        self.apply_final_activation = apply_final_activation
        self.use_tanh_at_final = use_tanh_at_final

        # Input channels: stereo mel = 128 (2 channels x 64 mel bins)
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
                if self.is_amp:
                    self.resblocks.append(
                        AMPBlock1(ch, k, tuple(dilations), activation=activation)
                    )
                else:
                    self.resblocks.append(ResBlock1(ch, k, tuple(dilations)))

        # Post-activation
        final_channels = upsample_initial_channel // (2 ** self.num_upsamples)
        if self.is_amp:
            self.act_post = Activation1d(SnakeBeta(final_channels))
        else:
            self.act_post = None  # use leaky_relu inline

        # Output conv
        out_channels = 2 if stereo else 1
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
        # Always run in fp32 — bfloat16 accumulation errors compound through
        # 108 sequential convolutions causing 40-90% spectral degradation
        x = x.astype(mx.float32)

        # Transpose: (B, channels, time, mel_bins) -> (B, channels, mel_bins, time)
        x = x.transpose(0, 1, 3, 2)

        # For stereo: (B, 2, mel_bins, time) -> (B, 2*mel_bins, time)
        b, s, m, t = x.shape
        x = x.reshape(b, s * m, t)

        # Initial conv
        x = self.conv_pre(x)

        # Upsampling with residual blocks
        for i in range(self.num_upsamples):
            if not self.is_amp:
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
            mx.clear_cache()  # Free intermediate buffers between vocoder stages

        # Post-activation
        if self.is_amp and self.act_post is not None:
            x = self.act_post(x)
        else:
            # PyTorch uses default leaky_relu slope (0.01) here, not LRELU_SLOPE
            x = nn.leaky_relu(x)

        x = self.conv_post(x)
        mx.eval(x)

        if self.apply_final_activation:
            if self.use_tanh_at_final:
                x = mx.tanh(x)
            else:
                x = mx.clip(x, -1, 1)

        return x


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


def _load_snakebeta_weights(f, prefix: str, snake: SnakeBeta, keys) -> None:
    """Load SnakeBeta alpha and beta parameters."""
    import torch

    for param_name in ["alpha", "beta"]:
        pt_key = f"{prefix}.{param_name}"
        if pt_key in keys:
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            setattr(snake, param_name, mx.array(tensor.numpy()))


def _load_lowpass_filter_weights(f, prefix: str, lpf: LowPassFilter1d, keys) -> None:
    """Load LowPassFilter1d filter buffer."""
    import torch

    pt_key = f"{prefix}.filter"
    if pt_key in keys:
        tensor = f.get_tensor(pt_key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        lpf.filter = mx.array(tensor.numpy())


def _load_upsample1d_weights(f, prefix: str, up: UpSample1d, keys) -> None:
    """Load UpSample1d filter buffer."""
    import torch

    pt_key = f"{prefix}.filter"
    if pt_key in keys:
        tensor = f.get_tensor(pt_key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        up.filter = mx.array(tensor.numpy())


def _load_activation1d_weights(f, prefix: str, act1d: Activation1d, keys) -> None:
    """Load Activation1d weights (SnakeBeta + upsample/downsample filters)."""
    _load_snakebeta_weights(f, f"{prefix}.act", act1d.act, keys)
    _load_upsample1d_weights(f, f"{prefix}.upsample", act1d.upsample, keys)
    _load_lowpass_filter_weights(
        f, f"{prefix}.downsample.lowpass", act1d.downsample.lowpass, keys
    )


def _load_amp_block_weights(f, prefix: str, block: AMPBlock1, keys) -> None:
    """Load AMPBlock1 weights including convolutions, SnakeBeta, and filter buffers."""
    for j, conv in enumerate(block.convs1):
        _load_conv1d_weights(f, f"{prefix}.convs1.{j}", conv, keys)
    for j, conv in enumerate(block.convs2):
        _load_conv1d_weights(f, f"{prefix}.convs2.{j}", conv, keys)
    for j, act in enumerate(block.acts1):
        _load_activation1d_weights(f, f"{prefix}.acts1.{j}", act, keys)
    for j, act in enumerate(block.acts2):
        _load_activation1d_weights(f, f"{prefix}.acts2.{j}", act, keys)


def _load_vocoder_inner(f, vocoder: Vocoder, prefix: str, keys) -> int:
    """Load weights for a single Vocoder instance.

    Shared logic used by both load_vocoder_weights and load_vocoder_with_bwe_weights.
    """
    loaded_count = 0

    # Load conv_pre
    _load_conv1d_weights(f, f"{prefix}.conv_pre", vocoder.conv_pre, keys)
    loaded_count += 1

    # Load upsampling layers
    for i, up in enumerate(vocoder.ups):
        _load_conv_transpose1d_weights(f, f"{prefix}.ups.{i}", up, keys)
        loaded_count += 1

    # Load resblocks
    for i, block in enumerate(vocoder.resblocks):
        block_prefix = f"{prefix}.resblocks.{i}"
        if isinstance(block, AMPBlock1):
            _load_amp_block_weights(f, block_prefix, block, keys)
        else:
            for j, conv in enumerate(block.convs1):
                _load_conv1d_weights(f, f"{block_prefix}.convs1.{j}", conv, keys)
            for j, conv in enumerate(block.convs2):
                _load_conv1d_weights(f, f"{block_prefix}.convs2.{j}", conv, keys)
        loaded_count += 1

    # Load act_post (for AMP mode)
    if vocoder.is_amp and vocoder.act_post is not None:
        _load_activation1d_weights(f, f"{prefix}.act_post", vocoder.act_post, keys)
        loaded_count += 1

    # Load conv_post
    _load_conv1d_weights(f, f"{prefix}.conv_post", vocoder.conv_post, keys)
    loaded_count += 1

    return loaded_count


def load_vocoder_weights(vocoder: Vocoder, weights_path: str) -> None:
    """Load Vocoder weights from safetensors file."""
    from safetensors import safe_open

    print(f"Loading Vocoder weights from {weights_path}...")

    with safe_open(weights_path, framework="pt") as f:
        keys = f.keys()

        # Check if vocoder weights exist
        vocoder_keys = [k for k in keys if k.startswith("vocoder.")]
        if not vocoder_keys:
            print("  Warning: No vocoder weights found in checkpoint")
            return

        loaded_count = _load_vocoder_inner(f, vocoder, "vocoder", keys)

    print(f"  Loaded {loaded_count} vocoder weight tensors")


def load_vocoder_with_bwe_weights(
    vocoder_with_bwe: VocoderWithBWE, weights_path: str
) -> None:
    """Load weights for VocoderWithBWE from a safetensors checkpoint."""
    from safetensors import safe_open
    import torch

    print(f"Loading VocoderWithBWE weights from {weights_path}...")
    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        keys = f.keys()

        # Load inner vocoder
        loaded_count += _load_vocoder_inner(
            f, vocoder_with_bwe.vocoder, "vocoder.vocoder", keys
        )

        # Load BWE generator
        loaded_count += _load_vocoder_inner(
            f, vocoder_with_bwe.bwe_generator, "vocoder.bwe_generator", keys
        )

        # Load mel_stft
        mel_stft = vocoder_with_bwe.mel_stft
        stft_fn = mel_stft.stft_fn

        # Load STFT forward_basis and inverse_basis
        for buf_name in ["forward_basis", "inverse_basis"]:
            pt_key = f"vocoder.mel_stft.stft_fn.{buf_name}"
            if pt_key in keys:
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                setattr(stft_fn, buf_name, mx.array(tensor.numpy()))
                loaded_count += 1

        # Load mel_basis
        pt_key = "vocoder.mel_stft.mel_basis"
        if pt_key in keys:
            tensor = f.get_tensor(pt_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            mel_stft.mel_basis = mx.array(tensor.numpy())
            loaded_count += 1

    print(f"  Loaded {loaded_count} vocoder+BWE weight tensors")
