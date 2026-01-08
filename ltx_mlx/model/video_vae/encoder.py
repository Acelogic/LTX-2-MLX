"""Video VAE Encoder for LTX-2."""

from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .convolution import make_conv_nd, PaddingModeType, NormLayerType
from .ops import patchify, PerChannelStatistics
from .resnet import ResnetBlock3D, UNetMidBlock3D, PixelNorm
from .sampling import SpaceToDepthDownsample


class LogVarianceType:
    """Log variance types for VAE."""

    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


def _make_encoder_block(
    block_name: str,
    block_config: Dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """
    Create an encoder block based on the block name and configuration.

    Returns:
        Tuple of (block module, output channels).
    """
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Unknown encoder block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):
    """
    Video VAE Encoder.

    Encodes video frames into a latent representation. The encoder compresses
    the input video through a series of downsampling operations.

    Standard LTX Video configuration:
        - patch_size=4
        - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
        - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
        - Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16)
        - Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, Union[int, Dict[str, Any]]]] = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: str = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """
        Initialize VideoEncoder.

        Args:
            convolution_dimensions: Number of dimensions for convolutions (2 or 3).
            in_channels: Number of input channels (3 for RGB).
            out_channels: Number of latent channels (128 for LTX-2).
            encoder_blocks: List of (block_name, params) tuples.
            patch_size: Initial spatial compression factor.
            norm_layer: Normalization layer type.
            latent_log_var: Log variance mode.
            encoder_spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()

        if encoder_blocks is None:
            encoder_blocks = []

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        # After patchify, channels increase by patch_size^2
        in_channels = in_channels * patch_size ** 2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = []
        for block_name, block_params in encoder_blocks:
            # Convert int to dict format
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )
            self.down_blocks.append(block)

        # Output normalization and projection
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_groups=self._norm_num_groups, dims=feature_channels, eps=1e-6)
        else:  # PIXEL_NORM
            self.conv_norm_out = PixelNorm()

        # Output channels depend on latent_log_var mode
        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def __call__(self, sample: mx.array) -> mx.array:
        """
        Encode video frames into normalized latent representation.

        Args:
            sample: Input video (B, C, F, H, W). F must be 1 + 8*k.

        Returns:
            Normalized latent means (B, 128, F', H', W').
        """
        # Validate frame count
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                f"Invalid number of frames: {frames_count}. "
                "Encode input must have 1 + 8*k frames (e.g., 1, 9, 17, ...)."
            )

        # Initial spatial compression via patchify
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample, causal=True)

        # Down blocks
        for down_block in self.down_blocks:
            if hasattr(down_block, "__call__"):
                # Check if it accepts causal argument
                if isinstance(down_block, (UNetMidBlock3D, ResnetBlock3D, SpaceToDepthDownsample)):
                    sample = down_block(sample, causal=True)
                else:
                    sample = down_block(sample, causal=True)
            else:
                sample = down_block(sample)

        # Output normalization
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample, causal=True)

        # Handle log variance modes
        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Uniform: N means + 1 shared logvar -> expand to 2*N
            means = sample[:, :-1, ...]
            logvar = sample[:, -1:, ...]
            num_channels = means.shape[1]
            repeated_logvar = mx.repeat(logvar, num_channels, axis=1)
            sample = mx.concatenate([means, repeated_logvar], axis=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # Minimal clamp value
            sample = mx.concatenate(
                [sample, mx.ones_like(sample) * approx_ln_0],
                axis=1,
            )

        # Split into means and logvar, return normalized means
        means = sample[:, :self.latent_channels, ...]
        return self.per_channel_statistics.normalize(means)
