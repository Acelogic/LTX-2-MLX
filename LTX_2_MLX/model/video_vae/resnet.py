"""ResNet blocks for Video VAE."""

from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .convolution import CausalConv3d, PaddingModeType, NormLayerType, make_conv_nd, make_linear_nd


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.

    For each element along the chosen dimension, normalizes the tensor
    by the root-mean-square of its values across that dimension:
        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        """
        Initialize PixelNorm.

        Args:
            dim: Dimension along which to compute the RMS (typically channels).
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply pixel normalization."""
        rms = mx.sqrt(mx.mean(x * x, axis=self.dim, keepdims=True) + self.eps)
        return x / rms


class ResnetBlock3D(nn.Module):
    """
    A 3D ResNet block with optional timestep conditioning and noise injection.

    Parameters:
        dims: Dimension specification (2, 3, or (2,1) for DualConv).
        in_channels: Number of input channels.
        out_channels: Number of output channels (defaults to in_channels).
        dropout: Dropout probability.
        groups: Number of groups for group normalization.
        eps: Epsilon for normalization layers.
        norm_layer: Type of normalization (GROUP_NORM or PIXEL_NORM).
        inject_noise: Whether to inject spatial noise (StyleGAN-style).
        timestep_conditioning: Whether to condition on timestep embeddings.
        spatial_padding_mode: Padding mode for convolutions.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise
        self.timestep_conditioning = timestep_conditioning
        self.dropout_rate = dropout

        # First normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm1 = nn.GroupNorm(num_groups=groups, dims=in_channels, eps=eps)
        else:  # PIXEL_NORM
            self.norm1 = PixelNorm()

        # First convolution
        self.conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        # Noise injection scales (StyleGAN-style)
        if inject_noise:
            self.per_channel_scale1 = mx.zeros((in_channels, 1, 1))

        # Second normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm2 = nn.GroupNorm(num_groups=groups, dims=out_channels, eps=eps)
        else:  # PIXEL_NORM
            self.norm2 = PixelNorm()

        # Second convolution
        self.conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = mx.zeros((in_channels, 1, 1))

        # Shortcut connection
        if in_channels != out_channels:
            self.conv_shortcut = make_linear_nd(
                dims=dims,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            # LayerNorm equivalent using GroupNorm with 1 group
            self.norm3 = nn.GroupNorm(num_groups=1, dims=in_channels, eps=eps)
        else:
            self.conv_shortcut = None
            self.norm3 = None

        # Timestep conditioning
        if timestep_conditioning:
            self.scale_shift_table = mx.random.normal((4, in_channels)) / (in_channels ** 0.5)

    def _feed_spatial_noise(
        self,
        hidden_states: mx.array,
        per_channel_scale: mx.array,
        key: Optional[mx.array] = None,
    ) -> mx.array:
        """Inject spatial noise (StyleGAN-style explicit noise inputs)."""
        spatial_shape = hidden_states.shape[-2:]

        if key is not None:
            spatial_noise = mx.random.normal(spatial_shape, key=key)
        else:
            spatial_noise = mx.random.normal(spatial_shape)

        # Shape: (1, C, 1, H, W) for broadcasting
        scaled_noise = (spatial_noise[None] * per_channel_scale)[None, :, None, ...]
        return hidden_states + scaled_noise

    def __call__(
        self,
        input_tensor: mx.array,
        causal: bool = True,
        timestep: Optional[mx.array] = None,
        key: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through the ResNet block.

        Args:
            input_tensor: Input tensor of shape (B, C, D, H, W).
            causal: Whether to use causal convolution.
            timestep: Optional timestep embedding for conditioning.
            key: Optional random key for noise injection.

        Returns:
            Output tensor of shape (B, C_out, D, H, W).
        """
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        # First norm + activation
        hidden_states = self.norm1(hidden_states)

        # Timestep conditioning (AdaLN-style)
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("timestep must be provided when timestep_conditioning is True")

            # Reshape scale_shift_table for broadcasting
            ada_values = self.scale_shift_table[None, ..., None, None, None]
            ada_values = ada_values + timestep.reshape(
                batch_size, 4, -1,
                timestep.shape[-3], timestep.shape[-2], timestep.shape[-1]
            )
            shift1, scale1, shift2, scale2 = mx.split(ada_values, 4, axis=1)
            shift1 = shift1.squeeze(1)
            scale1 = scale1.squeeze(1)
            shift2 = shift2.squeeze(1)
            scale2 = scale2.squeeze(1)

            hidden_states = hidden_states * (1 + scale1) + shift1

        hidden_states = nn.silu(hidden_states)

        # First convolution
        hidden_states = self.conv1(hidden_states, causal=causal)

        # Noise injection
        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states, self.per_channel_scale1, key
            )

        # Second norm + activation
        hidden_states = self.norm2(hidden_states)

        if self.timestep_conditioning:
            hidden_states = hidden_states * (1 + scale2) + shift2

        hidden_states = nn.silu(hidden_states)

        # Dropout
        if self.dropout_rate > 0:
            hidden_states = nn.Dropout(self.dropout_rate)(hidden_states)

        # Second convolution
        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states, self.per_channel_scale2, key
            )

        # Shortcut connection
        shortcut = input_tensor
        if self.norm3 is not None:
            shortcut = self.norm3(shortcut)
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)

        return shortcut + hidden_states


class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block with multiple residual blocks.

    Args:
        dims: Dimension specification.
        in_channels: Number of input channels.
        dropout: Dropout rate.
        num_layers: Number of residual blocks.
        resnet_eps: Epsilon for ResNet normalization.
        resnet_groups: Number of groups for group normalization.
        norm_layer: Normalization layer type.
        inject_noise: Whether to inject noise.
        timestep_conditioning: Whether to use timestep conditioning.
        spatial_padding_mode: Padding mode for convolutions.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: NormLayerType = NormLayerType.GROUP_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.timestep_conditioning = timestep_conditioning

        # Timestep embedder would be added here if needed
        # For now, we assume timestep embeddings come from outside

        self.res_blocks = [
            ResnetBlock3D(
                dims=dims,
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                norm_layer=norm_layer,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        causal: bool = True,
        timestep: Optional[mx.array] = None,
        key: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through the mid block.

        Args:
            hidden_states: Input tensor.
            causal: Whether to use causal convolution.
            timestep: Optional timestep embedding.
            key: Optional random key for noise injection.

        Returns:
            Output tensor.
        """
        for resnet in self.res_blocks:
            hidden_states = resnet(
                hidden_states,
                causal=causal,
                timestep=timestep,
                key=key,
            )

        return hidden_states
