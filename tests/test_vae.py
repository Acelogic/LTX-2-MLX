"""Tests for LTX-2 MLX Video VAE."""

import mlx.core as mx
import numpy as np


def test_vae_ops():
    """Test VAE basic operations."""
    from LTX_2_MLX.model.video_vae import patchify, unpatchify

    # Test patchify/unpatchify roundtrip
    # Input: (B, C, F, H, W) with H, W divisible by patch_size
    video = mx.random.normal(shape=(1, 3, 9, 64, 64))  # 9 frames, 64x64

    # Patchify with patch_size=4
    patchified = patchify(video, patch_size_hw=4, patch_size_t=1)
    # Expected shape: (1, C*16, F, H/4, W/4) = (1, 48, 9, 16, 16)
    assert patchified.shape == (1, 48, 9, 16, 16), f"Got {patchified.shape}"
    print("  patchify: OK")

    # Unpatchify
    unpatchified = unpatchify(patchified, patch_size_hw=4, patch_size_t=1)
    assert unpatchified.shape == video.shape, f"Got {unpatchified.shape}"
    print("  unpatchify: OK")

    # Roundtrip should preserve values
    diff = float(mx.abs(video - unpatchified).max())
    assert diff < 1e-5, f"Roundtrip error: {diff}"
    print("  patchify_roundtrip: OK")


def test_pixel_shuffle():
    """Test pixel shuffle operations."""
    from LTX_2_MLX.model.video_vae import pixel_shuffle_3d, pixel_unshuffle_3d

    # Test pixel_shuffle_3d: (B, C*r², F, H, W) -> (B, C, F, H*r, W*r)
    # Note: Only spatial shuffle, temporal dimension unchanged
    x = mx.random.normal(shape=(1, 4, 2, 4, 4))  # 4 channels = 1 * 2²
    shuffled = pixel_shuffle_3d(x, upscale_factor=2)
    assert shuffled.shape == (1, 1, 2, 8, 8), f"Got {shuffled.shape}"
    print("  pixel_shuffle_3d: OK")

    # Test pixel_unshuffle_3d: (B, C, F, H, W) -> (B, C*r², F, H/r, W/r)
    x = mx.random.normal(shape=(1, 1, 2, 8, 8))
    unshuffled = pixel_unshuffle_3d(x, downscale_factor=2)
    assert unshuffled.shape == (1, 4, 2, 4, 4), f"Got {unshuffled.shape}"
    print("  pixel_unshuffle_3d: OK")


def test_per_channel_statistics():
    """Test per-channel statistics normalization."""
    from LTX_2_MLX.model.video_vae import PerChannelStatistics

    stats = PerChannelStatistics(latent_channels=128)

    # Create test latent
    latent = mx.random.normal(shape=(1, 128, 4, 16, 16))

    # Normalize and unnormalize should be inverse operations
    normalized = stats.normalize(latent)
    assert normalized.shape == latent.shape
    print("  normalize: OK")

    unnormalized = stats.un_normalize(normalized)
    assert unnormalized.shape == latent.shape
    # Should be approximately equal
    diff = float(mx.abs(latent - unnormalized).max())
    assert diff < 1e-3, f"Normalize roundtrip error: {diff}"
    print("  un_normalize: OK")


def test_resnet_block():
    """Test ResNet 3D block."""
    from LTX_2_MLX.model.video_vae import ResnetBlock3D, NormLayerType

    block = ResnetBlock3D(
        dims=3,
        in_channels=64,
        out_channels=64,
        eps=1e-6,
        groups=32,
        norm_layer=NormLayerType.PIXEL_NORM,
    )

    # Test forward pass
    x = mx.random.normal(shape=(1, 64, 4, 8, 8))
    out = block(x, causal=True)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("  resnet_block_3d: OK")


def test_dual_conv3d():
    """Test DualConv3d (2D+1D decomposition of 3D conv)."""
    from LTX_2_MLX.model.video_vae import DualConv3d

    conv = DualConv3d(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        bias=True,
    )

    x = mx.random.normal(shape=(1, 64, 4, 8, 8))
    out = conv(x)  # No causal parameter for DualConv3d
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("  dual_conv3d: OK")


def test_causal_conv3d():
    """Test CausalConv3d."""
    from LTX_2_MLX.model.video_vae import CausalConv3d

    conv = CausalConv3d(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
    )

    x = mx.random.normal(shape=(1, 64, 4, 8, 8))
    out = conv(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("  causal_conv3d: OK")


def test_sampling_blocks():
    """Test upsampling and downsampling blocks."""
    from LTX_2_MLX.model.video_vae import SpaceToDepthDownsample, DepthToSpaceUpsample

    # Test downsampling: spatial 2x, temporal 1x
    down = SpaceToDepthDownsample(
        dims=3,
        in_channels=64,
        out_channels=128,
        stride=(1, 2, 2),
    )
    x = mx.random.normal(shape=(1, 64, 4, 16, 16))
    out = down(x, causal=True)
    assert out.shape == (1, 128, 4, 8, 8), f"Got {out.shape}"
    print("  space_to_depth_downsample: OK")

    # Test upsampling: spatial 2x, temporal 1x
    up = DepthToSpaceUpsample(
        dims=3,
        in_channels=128,
        stride=(1, 2, 2),
    )
    x = mx.random.normal(shape=(1, 128, 4, 8, 8))
    out = up(x, causal=True)
    assert out.shape == (1, 128, 4, 16, 16), f"Got {out.shape}"
    print("  depth_to_space_upsample: OK")


def test_video_decoder_minimal():
    """Test VideoDecoder with minimal configuration."""
    from LTX_2_MLX.model.video_vae import VideoDecoder

    # Minimal decoder - just conv_in and conv_out, no upsampling blocks
    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=[],  # No blocks
        patch_size=4,
        causal=True,
    )

    # Test forward pass with small latent
    # Input: (B, 128, F, H, W) where F, H, W are latent dimensions
    latent = mx.random.normal(shape=(1, 128, 2, 4, 4))

    out = decoder(latent)
    # Output shape: (B, 3, F, H*patch_size, W*patch_size)
    # With no blocks: F unchanged, H*4=16, W*4=16
    expected_shape = (1, 3, 2, 16, 16)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print("  video_decoder_minimal: OK")


def test_video_encoder_minimal():
    """Test VideoEncoder with minimal configuration."""
    from LTX_2_MLX.model.video_vae import VideoEncoder

    # Minimal encoder - just patchify, conv_in, and conv_out
    encoder = VideoEncoder(
        convolution_dimensions=3,
        in_channels=3,
        out_channels=128,
        encoder_blocks=[],  # No blocks
        patch_size=4,
    )

    # Test forward pass with small video
    # Input must have 1 + 8*k frames
    video = mx.random.normal(shape=(1, 3, 9, 64, 64))  # 9 = 1 + 8*1

    out = encoder(video)
    # After patchify: H'=16, W'=16
    # No blocks means F stays at 9
    expected_shape = (1, 128, 9, 16, 16)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print("  video_encoder_minimal: OK")


def run_vae_tests():
    """Run all VAE tests."""
    print("\n=== VAE Tests ===\n")

    print("Testing VAE ops...")
    test_vae_ops()

    print("\nTesting pixel shuffle...")
    test_pixel_shuffle()

    print("\nTesting per-channel statistics...")
    test_per_channel_statistics()

    print("\nTesting ResNet block...")
    test_resnet_block()

    print("\nTesting DualConv3d...")
    test_dual_conv3d()

    print("\nTesting CausalConv3d...")
    test_causal_conv3d()

    print("\nTesting sampling blocks...")
    test_sampling_blocks()

    print("\nTesting VideoDecoder (minimal)...")
    test_video_decoder_minimal()

    print("\nTesting VideoEncoder (minimal)...")
    test_video_encoder_minimal()

    print("\n=== All VAE Tests Passed! ===\n")


if __name__ == "__main__":
    run_vae_tests()
