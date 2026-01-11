"""Unit tests for LTX-2 upscaler components.

Tests spatial and temporal upscalers without requiring pre-trained weights.

Run with: pytest tests/test_upscalers.py -v
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from LTX_2_MLX.model.upscaler.spatial import (
    ResBlock3d,
    PixelShuffle2d,
    SpatialUpscaler,
)
from LTX_2_MLX.model.upscaler.temporal import (
    TemporalPixelShuffle,
    TemporalUpscaler,
)


class TestResBlock3d:
    """Test 3D residual block."""

    def test_basic_resblock(self):
        """Test basic 3D residual block."""
        block = ResBlock3d(channels=1024, num_groups=32)
        # NCTHW format
        x = mx.random.normal((1, 1024, 4, 8, 8))

        output = block(x)

        # Output should have same shape as input
        assert output.shape == x.shape
        assert np.all(np.isfinite(np.array(output)))

    def test_resblock_stability(self):
        """Test that ResBlock doesn't cause exponential amplification."""
        block1 = ResBlock3d(channels=512, num_groups=32)
        block2 = ResBlock3d(channels=512, num_groups=32)
        block3 = ResBlock3d(channels=512, num_groups=32)

        x = mx.random.normal((1, 512, 4, 8, 8))

        # Track variance through blocks
        initial_var = float(mx.var(x))

        x1 = block1(x)
        var1 = float(mx.var(x1))

        x2 = block2(x1)
        var2 = float(mx.var(x2))

        x3 = block3(x2)
        var3 = float(mx.var(x3))

        # Variance should stay bounded
        assert var1 < initial_var * 10, f"Variance exploded: {initial_var} -> {var1}"
        assert var2 < initial_var * 10, f"Variance exploded: {initial_var} -> {var2}"
        assert var3 < initial_var * 10, f"Variance exploded: {initial_var} -> {var3}"

    def test_resblock_different_sizes(self):
        """Test ResBlock with different channel sizes."""
        for channels in [256, 512, 1024]:
            block = ResBlock3d(channels=channels, num_groups=min(32, channels))
            x = mx.random.normal((1, channels, 4, 8, 8))  # NCTHW format

            output = block(x)
            assert output.shape == x.shape


class TestPixelShuffle2d:
    """Test pixel shuffle upsampler."""

    def test_basic_upsampling(self):
        """Test basic 2x spatial upsampling."""
        shuffle = PixelShuffle2d(upscale_factor=2)
        # Input: (B, H, W, C*r*r) where C*r*r = 2048, r=2, so C=512
        x = mx.random.normal((4, 8, 8, 2048))

        output = shuffle(x)

        # Output should be (B, H*2, W*2, C) = (4, 16, 16, 512)
        assert output.shape == (4, 16, 16, 512)
        assert np.all(np.isfinite(np.array(output)))

    def test_pixel_shuffle_rearrangement(self):
        """Test the pixel shuffle rearrangement."""
        shuffle = PixelShuffle2d(upscale_factor=2)

        # Create input with known pattern
        b, h, w = 1, 2, 2
        c_out = 4
        r = 2
        c_in = c_out * r * r  # 16

        x = mx.arange(b * h * w * c_in).reshape(b, h, w, c_in).astype(mx.float32)
        output = shuffle(x)

        # Output should be (1, 4, 4, 4)
        assert output.shape == (b, h * r, w * r, c_out)


class TestSpatialUpscaler:
    """Test complete spatial upscaler."""

    def test_basic_spatial_upscaling(self):
        """Test basic 2x spatial upscaling."""
        upscaler = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,  # Use smaller for testing
            num_blocks_per_stage=2,  # Fewer blocks for faster testing
        )
        # Input: (B, C, F, H, W) - NCFHW format
        x = mx.random.normal((1, 128, 4, 8, 8))

        output = upscaler(x)

        # Spatial dimensions should be doubled
        assert output.shape == (1, 128, 4, 16, 16)
        assert np.all(np.isfinite(np.array(output)))

    def test_spatial_upscaler_different_sizes(self):
        """Test spatial upscaler with different input sizes."""
        upscaler = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,
            num_blocks_per_stage=1,  # Fewer blocks for faster testing
        )

        test_shapes = [
            (1, 128, 2, 4, 4),
            (1, 128, 4, 8, 8),
        ]

        for shape in test_shapes:
            x = mx.random.normal(shape)
            output = upscaler(x)

            # Check spatial 2x upsampling
            b, c, f, h, w = shape
            assert output.shape == (b, c, f, h * 2, w * 2)

    def test_spatial_upscaler_numerical_stability(self):
        """Test that spatial upscaler produces stable outputs."""
        upscaler = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,
            num_blocks_per_stage=2,
        )
        x = mx.random.normal((1, 128, 4, 8, 8))

        output = upscaler(x)

        # Check for NaN or Inf
        output_np = np.array(output)
        assert np.all(np.isfinite(output_np)), "Output should not contain NaN or Inf"

        # Variance should be reasonable (not exploded)
        input_var = float(mx.var(x))
        output_var = float(mx.var(output))

        # Allow up to 100x variance increase
        assert output_var < input_var * 100, \
            f"Variance exploded: {input_var} -> {output_var}"


class TestTemporalPixelShuffle:
    """Test temporal pixel shuffle upsampler."""

    def test_basic_temporal_upsampling(self):
        """Test basic 2x temporal upsampling."""
        upsampler = TemporalPixelShuffle(in_channels=512, scale_factor=2)
        x = mx.random.normal((1, 512, 4, 8, 8))

        output = upsampler(x)

        # Temporal dimension should be doubled
        assert output.shape == (1, 512, 8, 8, 8)
        assert np.all(np.isfinite(np.array(output)))

    def test_temporal_pixel_shuffle_operation(self):
        """Test the temporal pixel shuffle rearrangement."""
        upsampler = TemporalPixelShuffle(in_channels=512, scale_factor=2)

        b, c, t, h, w = 1, 512, 4, 8, 8
        x = mx.random.normal((b, c, t, h, w))

        output = upsampler(x)

        # Output should be 2x larger temporally
        assert output.shape == (b, c, t * 2, h, w)


class TestTemporalUpscaler:
    """Test complete temporal upscaler."""

    def test_basic_temporal_upscaling(self):
        """Test basic 2x temporal upscaling."""
        upscaler = TemporalUpscaler(
            latent_channels=128,
            hidden_channels=512,
            num_res_blocks=2,  # Fewer blocks for faster testing
        )
        x = mx.random.normal((1, 128, 4, 8, 8))

        output = upscaler(x)

        # Temporal dimension: doubled then first frame removed (per PyTorch reference)
        # 4 frames -> 8 frames -> 7 frames (first frame encodes one pixel frame)
        assert output.shape == (1, 128, 7, 8, 8)
        assert np.all(np.isfinite(np.array(output)))

    def test_temporal_upscaler_different_frame_counts(self):
        """Test temporal upscaler with different frame counts."""
        upscaler = TemporalUpscaler(
            latent_channels=128,
            hidden_channels=512,
            num_res_blocks=1,  # Fewer blocks for faster testing
        )

        test_frames = [2, 4, 8]

        for frames in test_frames:
            x = mx.random.normal((1, 128, frames, 8, 8))
            output = upscaler(x)

            # Temporal: doubled then first frame removed (per PyTorch)
            # frames -> 2*frames -> 2*frames - 1
            assert output.shape == (1, 128, frames * 2 - 1, 8, 8)

    def test_temporal_upscaler_numerical_stability(self):
        """Test that temporal upscaler produces stable outputs."""
        upscaler = TemporalUpscaler(
            latent_channels=128,
            hidden_channels=512,
            num_res_blocks=2,
        )
        x = mx.random.normal((1, 128, 4, 8, 8))

        output = upscaler(x)

        # Check for NaN or Inf
        output_np = np.array(output)
        assert np.all(np.isfinite(output_np)), "Output should not contain NaN or Inf"

        # Variance should be reasonable
        input_var = float(mx.var(x))
        output_var = float(mx.var(output))

        # Allow reasonable variance change
        assert output_var < input_var * 100, \
            f"Variance exploded: {input_var} -> {output_var}"


class TestUpscalerIntegration:
    """Integration tests combining spatial and temporal upscaling."""

    def test_spatial_then_temporal(self):
        """Test applying spatial upscaling followed by temporal."""
        spatial = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,
            num_blocks_per_stage=1,
        )
        temporal = TemporalUpscaler(
            latent_channels=128,
            hidden_channels=512,
            num_res_blocks=1,
        )

        x = mx.random.normal((1, 128, 4, 8, 8))

        # Apply spatial (2x spatial)
        x_spatial = spatial(x)
        assert x_spatial.shape == (1, 128, 4, 16, 16)

        # Apply temporal (2x temporal - first frame removed per PyTorch)
        x_both = temporal(x_spatial)
        assert x_both.shape == (1, 128, 7, 16, 16)

        # Should produce finite values
        assert np.all(np.isfinite(np.array(x_both)))

    def test_temporal_then_spatial(self):
        """Test applying temporal upscaling followed by spatial."""
        temporal = TemporalUpscaler(
            latent_channels=128,
            hidden_channels=512,
            num_res_blocks=1,
        )
        spatial = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,
            num_blocks_per_stage=1,
        )

        x = mx.random.normal((1, 128, 4, 8, 8))

        # Apply temporal (2x temporal - first frame removed per PyTorch)
        x_temporal = temporal(x)
        assert x_temporal.shape == (1, 128, 7, 8, 8)

        # Apply spatial (2x spatial)
        x_both = spatial(x_temporal)
        assert x_both.shape == (1, 128, 7, 16, 16)

        # Should produce finite values
        assert np.all(np.isfinite(np.array(x_both)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
