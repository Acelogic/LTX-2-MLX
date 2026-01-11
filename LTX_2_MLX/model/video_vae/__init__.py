"""Video VAE encoder and decoder."""

from .convolution import (
    DualConv3d,
    CausalConv3d,
    Conv3d,
    PointwiseConv3d,
    make_conv_nd,
    make_linear_nd,
    PaddingModeType,
    NormLayerType,
)
from .ops import (
    patchify,
    unpatchify,
    PerChannelStatistics,
    pixel_shuffle_3d,
    pixel_unshuffle_3d,
)
from .resnet import ResnetBlock3D, UNetMidBlock3D, PixelNorm
from .sampling import (
    SpaceToDepthDownsample,
    DepthToSpaceUpsample,
    space_to_depth,
    depth_to_space,
)
from .encoder import VideoEncoder, LogVarianceType
from .decoder import VideoDecoder, decode_video
from .simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    decode_latent,
)
from .simple_encoder import (
    SimpleVideoEncoder,
    load_vae_encoder_weights,
    encode_video,
)
from .tiling import (
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
    decode_tiled,
    compute_trapezoidal_mask_1d,
)

__all__ = [
    # Convolution
    "DualConv3d",
    "CausalConv3d",
    "Conv3d",
    "PointwiseConv3d",
    "make_conv_nd",
    "make_linear_nd",
    "PaddingModeType",
    "NormLayerType",
    # Ops
    "patchify",
    "unpatchify",
    "PerChannelStatistics",
    "pixel_shuffle_3d",
    "pixel_unshuffle_3d",
    # ResNet
    "ResnetBlock3D",
    "UNetMidBlock3D",
    "PixelNorm",
    # Sampling
    "SpaceToDepthDownsample",
    "DepthToSpaceUpsample",
    "space_to_depth",
    "depth_to_space",
    # Encoder/Decoder
    "VideoEncoder",
    "VideoDecoder",
    "LogVarianceType",
    "decode_video",
    # Simple decoder (for weight loading)
    "SimpleVideoDecoder",
    "load_vae_decoder_weights",
    "decode_latent",
    # Simple encoder (for weight loading)
    "SimpleVideoEncoder",
    "load_vae_encoder_weights",
    "encode_video",
    # Tiling
    "TilingConfig",
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "decode_tiled",
    "compute_trapezoidal_mask_1d",
]
