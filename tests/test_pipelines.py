"""Tests for pipeline module: configuration, utilities, and generation flow."""

import os
import pytest
import mlx.core as mx
from unittest.mock import MagicMock
import tempfile
from PIL import Image

# Import the modules under test
from LTX_2_MLX.pipelines.text_to_video import (
    GenerationConfig,
    TextToVideoPipeline,
)
from LTX_2_MLX.pipelines.distilled import (
    DistilledConfig,
)
from LTX_2_MLX.pipelines.common import (
    ImageCondition,
    load_image_tensor,
    post_process_latent,
    timesteps_from_mask,
)
from LTX_2_MLX.types import VideoLatentShape


# ============================================================================
# GenerationConfig Tests
# ============================================================================

class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.height == 480
        assert config.width == 704
        assert config.num_frames == 121
        assert config.num_inference_steps == 50
        assert config.cfg_scale == 7.5
        assert config.seed is None
        assert config.use_distilled == False
        assert config.precision == "float16"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            height=720,
            width=1280,
            num_frames=65,
            num_inference_steps=30,
            cfg_scale=5.0,
            seed=42,
        )
        assert config.height == 720
        assert config.width == 1280
        assert config.num_frames == 65
        assert config.num_inference_steps == 30
        assert config.cfg_scale == 5.0
        assert config.seed == 42

    def test_valid_frame_counts(self):
        """Test valid frame counts (8k + 1)."""
        valid_frames = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]
        for frames in valid_frames:
            config = GenerationConfig(num_frames=frames)
            assert config.num_frames == frames

    def test_invalid_frame_count_raises(self):
        """Test invalid frame counts raise ValueError."""
        invalid_frames = [2, 8, 10, 16, 24, 32, 100, 120]
        for frames in invalid_frames:
            with pytest.raises(ValueError) as exc_info:
                GenerationConfig(num_frames=frames)
            assert "8*k + 1" in str(exc_info.value)


class TestDistilledConfig:
    """Tests for DistilledConfig dataclass."""

    def test_valid_config(self):
        """Test valid distilled configuration (resolution divisible by 64)."""
        # DistilledConfig requires resolution divisible by 64
        config = DistilledConfig(height=512, width=768, num_frames=97)
        assert config.height == 512
        assert config.width == 768
        assert config.num_frames == 97
        assert config.seed == 42
        assert config.fps == 24.0

    def test_invalid_frame_count_raises(self):
        """Test invalid frame counts raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DistilledConfig(height=512, width=768, num_frames=100)
        assert "8*k + 1" in str(exc_info.value)

    def test_resolution_must_be_divisible_by_64(self):
        """Test resolution validation for two-stage pipeline."""
        # Valid resolutions (divisible by 64)
        config = DistilledConfig(height=512, width=768, num_frames=97)
        assert config.height == 512

        # Invalid resolutions
        with pytest.raises(ValueError) as exc_info:
            DistilledConfig(height=480, width=700, num_frames=97)  # Neither divisible by 64
        assert "divisible by 64" in str(exc_info.value)


# ============================================================================
# ImageCondition Tests
# ============================================================================

class TestImageCondition:
    """Tests for ImageCondition dataclass."""

    def test_default_strength(self):
        """Test default conditioning strength."""
        cond = ImageCondition(image_path="/path/to/image.png", frame_index=0)
        assert cond.strength == 0.95

    def test_custom_strength(self):
        """Test custom conditioning strength."""
        cond = ImageCondition(
            image_path="/path/to/image.png",
            frame_index=5,
            strength=0.8,
        )
        assert cond.frame_index == 5
        assert cond.strength == 0.8


# ============================================================================
# Common Pipeline Utility Tests
# ============================================================================

class TestLoadImageTensor:
    """Tests for load_image_tensor function."""

    def test_load_valid_rgb_image(self):
        """Test loading a valid RGB image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            img_path = os.path.join(tmpdir, "test.png")
            img = Image.new("RGB", (256, 256), color=(128, 64, 32))
            img.save(img_path)

            # Load and validate
            tensor = load_image_tensor(img_path, height=128, width=128)

            assert tensor.shape == (1, 3, 1, 128, 128)
            assert tensor.dtype == mx.float32
            # Values should be in [-1, 1] range
            assert float(mx.min(tensor)) >= -1.0
            assert float(mx.max(tensor)) <= 1.0

    def test_load_valid_rgba_image(self):
        """Test loading an RGBA image (converted to RGB)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            img = Image.new("RGBA", (256, 256), color=(128, 64, 32, 255))
            img.save(img_path)

            tensor = load_image_tensor(img_path, height=64, width=64)
            assert tensor.shape == (1, 3, 1, 64, 64)

    def test_load_valid_grayscale_image(self):
        """Test loading a grayscale image (converted to RGB)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            img = Image.new("L", (256, 256), color=128)
            img.save(img_path)

            tensor = load_image_tensor(img_path, height=64, width=64)
            assert tensor.shape == (1, 3, 1, 64, 64)

    def test_nonexistent_file_raises(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_tensor("/nonexistent/path.png", height=64, width=64)

    def test_custom_dtype(self):
        """Test loading with custom dtype."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            img = Image.new("RGB", (64, 64))
            img.save(img_path)

            tensor = load_image_tensor(img_path, 32, 32, dtype=mx.float16)
            assert tensor.dtype == mx.float16


class TestPostProcessLatent:
    """Tests for post_process_latent function."""

    def test_full_denoise(self):
        """Test with full denoise mask (all 1s)."""
        denoised = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
        clean = mx.array([[[5.0, 6.0], [7.0, 8.0]]])
        mask = mx.ones_like(denoised)

        result = post_process_latent(denoised, mask, clean)
        assert mx.allclose(result, denoised)

    def test_no_denoise(self):
        """Test with no denoise mask (all 0s)."""
        denoised = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
        clean = mx.array([[[5.0, 6.0], [7.0, 8.0]]])
        mask = mx.zeros_like(denoised)

        result = post_process_latent(denoised, mask, clean)
        assert mx.allclose(result, clean)

    def test_partial_denoise(self):
        """Test with partial denoise mask."""
        denoised = mx.array([[[1.0, 2.0]]])
        clean = mx.array([[[5.0, 6.0]]])
        mask = mx.array([[[1.0, 0.0]]])  # First element denoised, second clean

        result = post_process_latent(denoised, mask, clean)
        expected = mx.array([[[1.0, 6.0]]])
        assert mx.allclose(result, expected)

    def test_preserves_dtype(self):
        """Test that output dtype matches input."""
        denoised = mx.array([[[1.0]]]).astype(mx.float16)
        clean = mx.array([[[2.0]]]).astype(mx.float16)
        mask = mx.ones_like(denoised)

        result = post_process_latent(denoised, mask, clean)
        assert result.dtype == mx.float16


class TestTimestepsFromMask:
    """Tests for timesteps_from_mask function."""

    def test_full_mask(self):
        """Test with full denoise mask."""
        mask = mx.ones((1, 10))
        sigma = 0.5

        result = timesteps_from_mask(mask, sigma)
        expected = mx.full((1, 10), 0.5)
        assert mx.allclose(result, expected)

    def test_zero_mask(self):
        """Test with zero mask."""
        mask = mx.zeros((1, 10))
        sigma = 0.5

        result = timesteps_from_mask(mask, sigma)
        expected = mx.zeros((1, 10))
        assert mx.allclose(result, expected)

    def test_partial_mask(self):
        """Test with partial mask."""
        mask = mx.array([[1.0, 0.5, 0.0]])
        sigma = 2.0

        result = timesteps_from_mask(mask, sigma)
        expected = mx.array([[2.0, 1.0, 0.0]])
        assert mx.allclose(result, expected)


# ============================================================================
# VideoLatentShape Tests
# ============================================================================

class TestVideoLatentShape:
    """Tests for VideoLatentShape class."""

    def test_from_pixel_shape(self):
        """Test creating latent shape from pixel shape."""
        from LTX_2_MLX.types import VideoPixelShape

        pixel_shape = VideoPixelShape(
            batch=1,
            frames=97,
            height=480,
            width=704,
            fps=24.0,
        )

        latent_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=128
        )

        # VAE compression: 32x spatial, 8x temporal
        assert latent_shape.batch == 1
        assert latent_shape.channels == 128
        assert latent_shape.frames == (97 - 1) // 8 + 1  # 13
        assert latent_shape.height == 480 // 32  # 15
        assert latent_shape.width == 704 // 32  # 22


# ============================================================================
# TextToVideoPipeline Tests (with mocks)
# ============================================================================

class TestTextToVideoPipelineConfig:
    """Tests for TextToVideoPipeline configuration and shape calculations."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a pipeline with mock components."""
        from LTX_2_MLX.components.guiders import CFGGuider

        mock_transformer = MagicMock()
        mock_decoder = MagicMock()
        mock_patchifier = MagicMock()
        mock_guider = CFGGuider(scale=7.5)

        return TextToVideoPipeline(
            transformer=mock_transformer,
            decoder=mock_decoder,
            patchifier=mock_patchifier,
            guider=mock_guider,
        )

    def test_get_latent_shape(self, mock_pipeline):
        """Test latent shape calculation."""
        pipeline = mock_pipeline

        config = GenerationConfig(
            height=480,
            width=704,
            num_frames=121,
        )

        shape = pipeline.get_latent_shape(config)

        # VAE compression: 32x spatial, 8x temporal
        assert shape.batch == 1
        assert shape.channels == 128
        assert shape.frames == (121 - 1) // 8 + 1  # 16
        assert shape.height == 480 // 32  # 15
        assert shape.width == 704 // 32  # 22

    def test_initialize_latent_shape(self, mock_pipeline):
        """Test latent initialization produces correct shape."""
        pipeline = mock_pipeline

        shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=13,
            height=15,
            width=22,
        )

        latent = pipeline.initialize_latent(shape, seed=42)

        assert latent.shape == (1, 128, 13, 15, 22)

    def test_initialize_latent_with_seed(self, mock_pipeline):
        """Test latent initialization is deterministic with seed."""
        pipeline = mock_pipeline

        shape = VideoLatentShape(
            batch=1,
            channels=4,
            frames=2,
            height=4,
            width=4,
        )

        latent1 = pipeline.initialize_latent(shape, seed=42)
        mx.eval(latent1)

        latent2 = pipeline.initialize_latent(shape, seed=42)
        mx.eval(latent2)

        assert mx.allclose(latent1, latent2)

    def test_prepare_text_context_with_cfg(self, mock_pipeline):
        """Test text context preparation with CFG."""
        pipeline = mock_pipeline

        text_encoding = mx.ones((1, 10, 256))
        text_mask = mx.ones((1, 10))

        context, mask = pipeline.prepare_text_context(
            text_encoding, text_mask, cfg_scale=7.5
        )

        # With CFG, batch dimension should double
        assert context.shape == (2, 10, 256)
        assert mask.shape == (2, 10)

        # First half should be original, second half zeros (uncond)
        assert mx.allclose(context[0], text_encoding[0])
        assert mx.allclose(context[1], mx.zeros((10, 256)))

    def test_prepare_text_context_without_cfg(self, mock_pipeline):
        """Test text context preparation without CFG."""
        pipeline = mock_pipeline

        text_encoding = mx.ones((1, 10, 256))
        text_mask = mx.ones((1, 10))

        context, mask = pipeline.prepare_text_context(
            text_encoding, text_mask, cfg_scale=1.0
        )

        # Without CFG (scale <= 1), should return unchanged
        assert context.shape == (1, 10, 256)
        assert mask.shape == (1, 10)


# ============================================================================
# Component Integration Tests
# ============================================================================

class TestPipelineComponents:
    """Tests for pipeline component integration."""

    def test_patchifier_import(self):
        """Test VideoLatentPatchifier can be imported and instantiated."""
        from LTX_2_MLX.components.patchifiers import VideoLatentPatchifier

        patchifier = VideoLatentPatchifier(patch_size=1)
        # patch_size is a tuple (1, 1, 1) for spatial dimensions
        assert patchifier.patch_size == (1, 1, 1)

    def test_noiser_import(self):
        """Test GaussianNoiser can be imported and instantiated."""
        from LTX_2_MLX.components.noisers import GaussianNoiser

        noiser = GaussianNoiser()
        assert noiser is not None

    def test_guider_import(self):
        """Test CFGGuider can be imported and instantiated."""
        from LTX_2_MLX.components.guiders import CFGGuider

        # CFGGuider is a dataclass with 'scale' attribute
        guider = CFGGuider(scale=7.5)
        assert guider.scale == 7.5

    def test_diffusion_step_import(self):
        """Test EulerDiffusionStep can be imported and instantiated."""
        from LTX_2_MLX.components.diffusion_steps import EulerDiffusionStep

        step = EulerDiffusionStep()
        assert step is not None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    def test_invalid_resolution_in_config(self):
        """Test that invalid resolutions don't crash during config creation."""
        # Very small resolution that should still work for config
        config = GenerationConfig(height=64, width=64, num_frames=9)
        assert config.height == 64

    def test_zero_cfg_scale(self):
        """Test zero CFG scale doesn't cause division errors."""
        from LTX_2_MLX.components.guiders import CFGGuider

        mock_transformer = MagicMock()
        mock_decoder = MagicMock()
        mock_patchifier = MagicMock()
        mock_guider = CFGGuider(scale=0.0)

        pipeline = TextToVideoPipeline(
            transformer=mock_transformer,
            decoder=mock_decoder,
            patchifier=mock_patchifier,
            guider=mock_guider,
        )

        text_encoding = mx.ones((1, 10, 256))
        text_mask = mx.ones((1, 10))

        # Zero CFG should not cause errors
        context, _ = pipeline.prepare_text_context(
            text_encoding, text_mask, cfg_scale=0.0
        )

        assert context.shape == (1, 10, 256)


# ============================================================================
# Scheduler Integration Tests
# ============================================================================

class TestSchedulerIntegration:
    """Tests for scheduler integration with pipelines."""

    def test_get_sigma_schedule(self):
        """Test sigma schedule generation."""
        from LTX_2_MLX.components.schedulers import get_sigma_schedule

        sigmas = get_sigma_schedule(num_steps=10, distilled=False)

        assert len(sigmas) == 11  # num_steps + 1
        assert float(sigmas[-1]) == 0.0  # Should end at 0

    def test_distilled_sigma_values(self):
        """Test distilled sigma values are available."""
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES

        assert len(DISTILLED_SIGMA_VALUES) > 0
        assert DISTILLED_SIGMA_VALUES[-1] == 0.0  # Should end at 0
