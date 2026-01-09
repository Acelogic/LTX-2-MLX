"""Video VAE Decoder for LTX-2."""

from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from LTX_2_MLX.types import SpatioTemporalScaleFactors

from .convolution import make_conv_nd, PaddingModeType, NormLayerType
from .ops import unpatchify, PerChannelStatistics
from .resnet import ResnetBlock3D, UNetMidBlock3D, PixelNorm
from .sampling import DepthToSpaceUpsample


def _make_decoder_block(
    block_name: str,
    block_config: Dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """
    Create a decoder block based on the block name and configuration.

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
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Unknown decoder block: {block_name}")

    return block, out_channels


class VideoDecoder(nn.Module):
    """
    Video VAE Decoder.

    Decodes latent representation into video frames. The decoder upsamples
    latents through a series of upsampling operations (inverse of encoder).

    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard config.

    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, Union[int, Dict[str, Any]]]] = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        """
        Initialize VideoDecoder.

        Args:
            convolution_dimensions: Number of dimensions for convolutions.
            in_channels: Number of latent channels (128 for LTX-2).
            out_channels: Number of output channels (3 for RGB).
            decoder_blocks: List of (block_name, params) tuples.
            patch_size: Final spatial expansion factor.
            norm_layer: Normalization layer type.
            causal: Whether to use causal convolutions.
            timestep_conditioning: Whether to condition on timestep.
            decoder_spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()

        if decoder_blocks is None:
            decoder_blocks = []

        self.video_downscale_factors = SpatioTemporalScaleFactors(
            time=8, width=32, height=32
        )

        self.patch_size = patch_size
        out_channels = out_channels * patch_size ** 2  # For unpatchify
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        # Compute initial feature_channels by going through blocks in reverse
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = []
        for block_name, block_params in list(reversed(decoder_blocks)):
            # Convert int to dict format
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )
            self.up_blocks.append(block)

        # Output normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_groups=self._norm_num_groups, dims=feature_channels, eps=1e-6
            )
        else:  # PIXEL_NORM
            self.conv_norm_out = PixelNorm()

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        # Timestep conditioning parameters
        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array([1000.0])
            self.last_scale_shift_table = mx.zeros((2, feature_channels))

    def __call__(
        self,
        sample: mx.array,
        timestep: Optional[mx.array] = None,
        key: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Decode latent representation into video frames.

        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True).
            key: Random key for noise injection.

        Returns:
            Decoded video (B, 3, F, H, W).
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            if key is not None:
                noise = mx.random.normal(sample.shape, key=key) * self.decode_noise_scale
            else:
                noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning and timestep is not None:
            scaled_timestep = timestep * float(self.timestep_scale_multiplier)

        # Up blocks
        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                sample = up_block(
                    sample,
                    causal=self.causal,
                    timestep=scaled_timestep if self.timestep_conditioning else None,
                    key=key,
                )
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, key=key)
            else:
                sample = up_block(sample, causal=self.causal)

        # Output normalization
        sample = self.conv_norm_out(sample)

        # Timestep conditioning at output
        if self.timestep_conditioning and scaled_timestep is not None:
            # Simplified AdaLN - full implementation would include timestep embedding
            ada_values = self.last_scale_shift_table[None, ..., None, None, None]
            shift = ada_values[:, 0, ...]
            scale = ada_values[:, 1, ...]
            sample = sample * (1 + scale) + shift

        sample = nn.silu(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Final spatial expansion via unpatchify
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample


def decode_video(
    latent: mx.array,
    video_decoder: VideoDecoder,
) -> mx.array:
    """
    Decode a video latent tensor with the given decoder.

    Args:
        latent: Tensor (B, C, F, H, W) or (C, F, H, W).
        video_decoder: Decoder module.

    Returns:
        Decoded video in uint8 format (F, H, W, C) in [0, 255].
    """
    # Add batch dimension if needed
    if latent.ndim == 4:
        latent = latent[None, ...]

    decoded_video = video_decoder(latent)

    # Convert to uint8: [-1, 1] -> [0, 255]
    frames = mx.clip((decoded_video + 1.0) / 2.0, 0.0, 1.0) * 255.0
    frames = frames.astype(mx.uint8)

    # Rearrange from (B, C, F, H, W) to (F, H, W, C)
    frames = frames[0]  # Remove batch dim
    frames = frames.transpose(1, 2, 3, 0)  # (C, F, H, W) -> (F, H, W, C)

    return frames
