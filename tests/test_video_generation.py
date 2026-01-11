"""Full video generation tests with actual outputs.

These tests run complete video generation pipelines with real model weights,
generate actual video frames, and save them to disk for verification.

Run with: pytest tests/test_video_generation.py -v -s

Output videos are saved to: tests/outputs/
"""

import gc
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from conftest import VerboseTestLogger

# =============================================================================
# Constants
# =============================================================================

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
LTX2_WEIGHTS_DIR = WEIGHTS_DIR / "ltx-2"
GEMMA_WEIGHTS_DIR = WEIGHTS_DIR / "gemma-3-12b"
OUTPUT_DIR = Path(__file__).parent / "outputs"
CACHED_EMBEDDINGS_DIR = Path(__file__).parent / "fixtures" / "embeddings"

# Model paths
DISTILLED_WEIGHTS = LTX2_WEIGHTS_DIR / "ltx-2-19b-distilled.safetensors"
DISTILLED_FP8_WEIGHTS = LTX2_WEIGHTS_DIR / "ltx-2-19b-distilled-fp8.safetensors"
DEV_WEIGHTS = LTX2_WEIGHTS_DIR / "ltx-2-19b-dev.safetensors"
DEV_FP8_WEIGHTS = LTX2_WEIGHTS_DIR / "ltx-2-19b-dev-fp8.safetensors"
SPATIAL_UPSCALER_WEIGHTS = LTX2_WEIGHTS_DIR / "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# Text encoder system prompt
T2V_SYSTEM_PROMPT = "Describe the video in extreme detail, focusing on the visual content."

# Test prompts
TEST_PROMPTS = [
    "A serene mountain landscape at sunset with golden light reflecting on a lake",
    "A cat walking through a garden with colorful flowers",
    "Ocean waves crashing on a sandy beach under blue sky",
]


def get_available_weights() -> Optional[Path]:
    """Get the best available weights file."""
    candidates = [
        DISTILLED_FP8_WEIGHTS,
        DEV_FP8_WEIGHTS,
        DISTILLED_WEIGHTS,
        DEV_WEIGHTS,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def weights_available() -> bool:
    """Check if any model weights are available."""
    return get_available_weights() is not None


requires_weights = pytest.mark.skipif(
    not weights_available(),
    reason=f"Model weights not found in {LTX2_WEIGHTS_DIR}"
)


def gemma_weights_available() -> bool:
    """Check if Gemma 3 weights are available for text encoding."""
    return (GEMMA_WEIGHTS_DIR / "tokenizer.model").exists()


requires_gemma = pytest.mark.skipif(
    not gemma_weights_available(),
    reason=f"Gemma weights not found in {GEMMA_WEIGHTS_DIR}"
)


def ensure_output_dir():
    """Ensure output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_video_frames(frames: List[np.ndarray], output_path: str, fps: int = 24):
    """Save frames as video using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as images
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        # Use ffmpeg to create video
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False
    return True


def create_dummy_text_encoding(
    prompt: str,
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,
) -> tuple:
    """Create deterministic dummy text encoding based on prompt."""
    mx.random.seed(hash(prompt) % (2**31))
    text_encoding = mx.random.normal(shape=(batch_size, max_tokens, embed_dim)) * 0.1
    text_mask = mx.ones((batch_size, max_tokens))
    return text_encoding, text_mask


def create_position_grid_with_bounds(
    batch_size: int,
    frames: int,
    height: int,
    width: int,
) -> mx.array:
    """
    Create position grid with start/end bounds for the transformer.

    The model expects positions with shape (B, n_dims, T, 2) where
    the last dimension contains [start, end] bounds for each position.

    Args:
        batch_size: Batch size.
        frames: Number of latent frames.
        height: Latent height.
        width: Latent width.

    Returns:
        Position grid of shape (B, 3, T, 2).
    """
    from LTX_2_MLX.model.transformer import create_position_grid

    grid = create_position_grid(batch_size, frames, height, width)  # (B, 3, T)
    # Add bounds dimension - each position has (start, end) = (pos, pos+1)
    grid_start = grid[..., None]  # (B, 3, T, 1)
    grid_end = grid_start + 1
    return mx.concatenate([grid_start, grid_end], axis=-1)  # (B, 3, T, 2)


def encode_text_with_gemma(
    prompt: str,
    gemma_path: Optional[str] = None,
    ltx_weights_path: Optional[str] = None,
    max_length: int = 256,
    use_cache: bool = True,
) -> tuple:
    """
    Encode text using real Gemma 3 text encoder.

    This loads the full Gemma 3 model, tokenizes the prompt, runs it through
    the text encoder pipeline, and returns embeddings ready for the transformer.

    Args:
        prompt: Text prompt to encode.
        gemma_path: Path to Gemma 3 weights directory.
        ltx_weights_path: Path to LTX-2 weights for text encoder projection.
        max_length: Maximum sequence length (default 256).
        use_cache: Whether to use cached embeddings if available.

    Returns:
        Tuple of (text_encoding, text_mask) with shapes:
        - text_encoding: [1, max_length, 3840]
        - text_mask: [1, max_length]
    """
    from transformers import AutoTokenizer
    from LTX_2_MLX.model.text_encoder.gemma3 import Gemma3Config, Gemma3Model, load_gemma3_weights
    from LTX_2_MLX.model.text_encoder.encoder import create_text_encoder, load_text_encoder_weights

    gemma_path = gemma_path or str(GEMMA_WEIGHTS_DIR)
    ltx_weights_path = ltx_weights_path or str(get_available_weights())

    # Cache key based on prompt and max_length
    cache_key = hash((prompt, max_length)) % (2**31)
    cache_file = CACHED_EMBEDDINGS_DIR / f"{cache_key}.npz"

    # Check cache first
    if use_cache and cache_file.exists():
        data = np.load(cache_file)
        return mx.array(data["embedding"]), mx.array(data["mask"])

    # Load tokenizer (right padding is critical to avoid NaN!)
    tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    tokenizer.padding_side = "right"

    # Create chat format prompt (with <bos> token like official script)
    chat_prompt = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    # Tokenize
    encoding = tokenizer(
        chat_prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = mx.array(encoding["input_ids"])
    attention_mask = mx.array(encoding["attention_mask"])

    # Load Gemma 3 model
    gemma = Gemma3Model(Gemma3Config())
    load_gemma3_weights(gemma, gemma_path)
    mx.eval(gemma.parameters())

    # Load text encoder projection layers
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, ltx_weights_path)
    mx.eval(text_encoder.parameters())

    # Run through Gemma to get all hidden states
    _, hidden_states_tuple = gemma(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    if hidden_states_tuple is None:
        raise ValueError("Gemma must return hidden states when output_hidden_states=True")
    all_hidden_states: List[mx.array] = list(hidden_states_tuple)
    mx.eval(all_hidden_states)

    # Process through text encoder feature extractor
    encoded = text_encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=attention_mask,
        padding_side="right",
    )
    mx.eval(encoded)

    # Process through embeddings connector
    large_value = 1e9
    connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
    connector_mask = connector_mask.reshape(1, 1, 1, max_length)
    encoded, output_mask = text_encoder.embeddings_connector(encoded, connector_mask)
    mx.eval(encoded)
    mx.eval(output_mask)

    # Convert mask back to binary
    binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)

    # Apply mask to zero out padded positions (like official script)
    encoded = encoded * binary_mask[:, :, None].astype(encoded.dtype)
    mx.eval(encoded)

    # Cache for future use
    if use_cache:
        CACHED_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            embedding=np.array(encoded),
            mask=np.array(binary_mask),
            prompt=prompt,
        )

    # Cleanup Gemma (it's huge!)
    del gemma, text_encoder, tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return encoded, binary_mask


def encode_null_text_for_cfg(
    gemma_path: Optional[str] = None,
    ltx_weights_path: Optional[str] = None,
    max_length: int = 256,
    use_cache: bool = True,
) -> tuple:
    """
    Encode empty string through Gemma for proper CFG unconditional pass.

    For CFG to work correctly, the unconditional embedding should be the
    text encoder's output for an empty string, NOT zeros. This ensures
    the unconditional path has meaningful structure that CFG can push away from.

    Args:
        gemma_path: Path to Gemma 3 weights directory.
        ltx_weights_path: Path to LTX-2 weights for text encoder projection.
        max_length: Maximum sequence length (default 256).
        use_cache: Whether to use cached embeddings if available.

    Returns:
        Tuple of (null_encoding, null_mask) with shapes:
        - null_encoding: [1, max_length, 3840]
        - null_mask: [1, max_length]
    """
    # Encode empty string through Gemma (same pipeline as regular text)
    return encode_text_with_gemma(
        prompt="",  # Empty string for unconditional
        gemma_path=gemma_path,
        ltx_weights_path=ltx_weights_path,
        max_length=max_length,
        use_cache=use_cache,
    )


def create_null_text_encoding_fallback(
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,
) -> tuple:
    """
    Create zero-based null encoding as fallback when Gemma is not available.

    WARNING: This is NOT semantically correct for CFG. For proper CFG,
    use encode_null_text_for_cfg() which encodes an empty string through
    the text encoder. This fallback produces suboptimal results.

    Returns:
        Tuple of (null_encoding, null_mask).
    """
    null_encoding = mx.zeros((batch_size, max_tokens, embed_dim))
    null_mask = mx.zeros((batch_size, max_tokens))  # All masked out
    return null_encoding, null_mask


# =============================================================================
# Test Configuration Classes
# =============================================================================

@dataclass
class TestVideoConfig:
    """Configuration for a test video generation."""
    name: str
    prompt: str
    height: int = 256  # Small for faster testing
    width: int = 384
    num_frames: int = 9  # Minimum valid (8k+1)
    seed: int = 42


# =============================================================================
# Simple Denoising Loop Test
# =============================================================================

class TestSimpleGeneration:
    """Test simple video generation with real weights."""

    @requires_weights
    def test_generate_short_video(self, verbose_logger):
        """Generate a short video with real weights."""
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import (
            DISTILLED_SIGMA_VALUES,
            VideoLatentPatchifier,
            EulerDiffusionStep,
        )
        from LTX_2_MLX.types import VideoLatentShape

        ensure_output_dir()
        verbose_logger.log_step("Generating short video with distilled model")

        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()
        verbose_logger.log_step(f"Using weights: {weights_path.name}")

        # Test config
        config = TestVideoConfig(
            name="simple_test",
            prompt="A calm ocean with gentle waves under blue sky",
            height=256,
            width=384,
            num_frames=9,  # 1 + 8 = 9 frames
            seed=42,
        )

        # Compute latent dimensions
        latent_h = config.height // 32
        latent_w = config.width // 32
        latent_f = (config.num_frames - 1) // 8 + 1  # 1 for 9 frames

        verbose_logger.log_step(f"Config: {config.width}x{config.height}, {config.num_frames} frames")
        verbose_logger.log_step(f"Latent: {latent_f}x{latent_h}x{latent_w}")

        # Load transformer
        verbose_logger.log_step("Loading transformer...")
        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        # Load VAE decoder
        verbose_logger.log_step("Loading VAE decoder...")
        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        # Initialize components
        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        # Create dummy text encoding
        verbose_logger.log_step("Creating text encoding...")
        text_encoding, text_mask = create_dummy_text_encoding(config.prompt)

        # Initialize latent noise
        mx.random.seed(config.seed)
        latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))
        verbose_logger.log_step(f"Initial latent shape: {latent.shape}")

        # Create position grid with start/end bounds
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)

        output_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=latent_f,
            height=latent_h,
            width=latent_w,
        )

        # Denoising loop (7 steps for distilled)
        num_steps = len(sigmas) - 1
        verbose_logger.log_step(f"Running {num_steps} denoising steps...")

        start_time = time.time()
        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            verbose_logger.log_step(f"  Step {step_idx + 1}/{num_steps}, sigma={sigma:.4f}")

            # Patchify
            latent_patchified = patchifier.patchify(latent)

            # Create modality
            modality = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )

            # Forward pass (velocity prediction)
            velocity_patchified = model(modality)

            # Unpatchify
            velocity = patchifier.unpatchify(velocity_patchified, output_shape=output_shape)

            # Euler step
            latent = euler.step(
                sample=latent,
                denoised_sample=velocity,
                sigmas=sigmas,
                step_index=step_idx,
            )
            mx.eval(latent)

        denoise_time = time.time() - start_time
        verbose_logger.log_step(f"Denoising took {denoise_time:.2f}s")

        # Decode to video
        verbose_logger.log_step("Decoding with VAE...")
        start_time = time.time()
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)
        decode_time = time.time() - start_time
        verbose_logger.log_step(f"VAE decode took {decode_time:.2f}s")

        verbose_logger.log_step(f"Video shape: {video.shape}")

        # Save video
        frames = [np.array(video[f]) for f in range(video.shape[0])]
        output_path = str(OUTPUT_DIR / f"test_{config.name}.mp4")
        verbose_logger.log_step(f"Saving {len(frames)} frames to {output_path}")

        success = save_video_frames(frames, output_path)
        assert success, "Failed to save video"

        verbose_logger.log_info(f"Video saved to {output_path}")

        # Verify output
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Cleanup
        del model, vae_decoder, latent, video
        mx.metal.clear_cache()
        gc.collect()


# =============================================================================
# Multi-Resolution Tests
# =============================================================================

class TestMultiResolution:
    """Test video generation at multiple resolutions."""

    @requires_weights
    @pytest.mark.parametrize("resolution", [
        (256, 384),   # Low res, fast
        (320, 512),   # Medium res
    ])
    def test_resolution(self, resolution, verbose_logger):
        """Test generation at specific resolution."""
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, EulerDiffusionStep
        from LTX_2_MLX.types import VideoLatentShape

        ensure_output_dir()

        height, width = resolution
        verbose_logger.log_step(f"Testing resolution: {width}x{height}")

        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()

        # Compute dimensions
        latent_h = height // 32
        latent_w = width // 32
        latent_f = 1  # Single frame for speed

        # Load models
        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        # Components
        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        # Create inputs
        text_encoding, text_mask = create_dummy_text_encoding("A beautiful landscape")
        mx.random.seed(42)
        latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)

        output_shape = VideoLatentShape(1, 128, latent_f, latent_h, latent_w)

        # Quick denoising (3 steps for speed)
        for step_idx in range(3):
            sigma = float(sigmas[step_idx])
            latent_patchified = patchifier.patchify(latent)

            modality = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )

            velocity_patchified = model(modality)
            velocity = patchifier.unpatchify(velocity_patchified, output_shape=output_shape)

            latent = euler.step(latent, velocity, sigmas, step_idx)
            mx.eval(latent)

        # Decode
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)

        verbose_logger.log_step(f"Output video shape: {video.shape}")

        # Verify dimensions
        expected_h = latent_h * 32
        expected_w = latent_w * 32
        # Due to causal trimming, output frames may differ
        assert video.shape[1] == expected_h, f"Height mismatch: {video.shape[1]} vs {expected_h}"
        assert video.shape[2] == expected_w, f"Width mismatch: {video.shape[2]} vs {expected_w}"

        verbose_logger.log_info(f"Resolution {width}x{height} test passed")

        # Cleanup
        del model, vae_decoder
        mx.metal.clear_cache()
        gc.collect()


# =============================================================================
# Multiple Prompts Test
# =============================================================================

class TestMultiplePrompts:
    """Test generation with different prompts."""

    @requires_weights
    def test_generate_with_prompts(self, verbose_logger):
        """Generate videos with multiple prompts."""
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, EulerDiffusionStep
        from LTX_2_MLX.types import VideoLatentShape

        ensure_output_dir()
        verbose_logger.log_step("Testing multiple prompts")

        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()

        # Load models once
        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        # Test prompts
        prompts = [
            "A calm lake at sunset with reflections",
            "A bustling city street with cars",
            "A forest path with sunlight through trees",
        ]

        latent_f, latent_h, latent_w = 1, 8, 12  # 256x384
        output_shape = VideoLatentShape(1, 128, latent_f, latent_h, latent_w)
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)

        for i, prompt in enumerate(prompts):
            verbose_logger.log_step(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

            # Create encoding for this prompt
            text_encoding, text_mask = create_dummy_text_encoding(prompt)

            # Initialize noise with different seed per prompt
            mx.random.seed(42 + i)
            latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))

            # Quick denoising (2 steps)
            for step_idx in range(2):
                sigma = float(sigmas[step_idx])
                latent_patchified = patchifier.patchify(latent)

                modality = Modality(
                    latent=latent_patchified,
                    context=text_encoding,
                    context_mask=text_mask,
                    timesteps=mx.array([sigma]),
                    positions=positions,
                    enabled=True,
                )

                velocity = patchifier.unpatchify(model(modality), output_shape=output_shape)
                latent = euler.step(latent, velocity, sigmas, step_idx)
                mx.eval(latent)

            # Decode
            video = decode_latent(latent, vae_decoder)
            mx.eval(video)

            # Save
            frames = [np.array(video[f]) for f in range(video.shape[0])]
            output_path = str(OUTPUT_DIR / f"prompt_test_{i+1}.mp4")
            save_video_frames(frames, output_path)

            verbose_logger.log_step(f"  Saved to {output_path}")

        verbose_logger.log_info("Multiple prompts test passed")

        # Cleanup
        del model, vae_decoder
        mx.metal.clear_cache()
        gc.collect()


# =============================================================================
# Longer Video Test
# =============================================================================

class TestLongerVideo:
    """Test generation of longer videos."""

    @requires_weights
    @pytest.mark.slow
    def test_longer_video(self, verbose_logger):
        """Generate a longer video (17 frames)."""
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, EulerDiffusionStep
        from LTX_2_MLX.types import VideoLatentShape

        ensure_output_dir()
        verbose_logger.log_step("Testing longer video (17 frames)")

        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()

        # Config for longer video
        num_frames = 17  # 2 latent frames -> 17 output frames
        height, width = 256, 384
        latent_f = (num_frames - 1) // 8 + 1  # = 3
        latent_h = height // 32
        latent_w = width // 32

        verbose_logger.log_step(f"Frames: {num_frames}, Latent frames: {latent_f}")

        # Load models
        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        text_encoding, text_mask = create_dummy_text_encoding("A flowing river through mountains")

        mx.random.seed(42)
        latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)
        output_shape = VideoLatentShape(1, 128, latent_f, latent_h, latent_w)

        # Full denoising loop
        start_time = time.time()
        for step_idx in range(len(sigmas) - 1):
            sigma = float(sigmas[step_idx])
            verbose_logger.log_step(f"  Step {step_idx + 1}/{len(sigmas)-1}")

            latent_patchified = patchifier.patchify(latent)
            modality = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )

            velocity = patchifier.unpatchify(model(modality), output_shape=output_shape)
            latent = euler.step(latent, velocity, sigmas, step_idx)
            mx.eval(latent)

        denoise_time = time.time() - start_time

        # Decode
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)

        verbose_logger.log_step(f"Output video: {video.shape}")
        verbose_logger.log_step(f"Denoise time: {denoise_time:.2f}s")

        # Save
        frames = [np.array(video[f]) for f in range(video.shape[0])]
        output_path = str(OUTPUT_DIR / "longer_video_test.mp4")
        save_video_frames(frames, output_path)

        assert os.path.exists(output_path)
        verbose_logger.log_info(f"Longer video saved to {output_path}")

        # Cleanup
        del model, vae_decoder
        mx.metal.clear_cache()
        gc.collect()


# =============================================================================
# Quality Verification Tests
# =============================================================================

class TestVideoQuality:
    """Tests to verify video output quality."""

    @requires_weights
    def test_output_pixel_range(self, verbose_logger):
        """Verify output pixels are in valid range."""
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, EulerDiffusionStep
        from LTX_2_MLX.types import VideoLatentShape

        verbose_logger.log_step("Testing output pixel range")

        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()

        # Minimal config
        latent_f, latent_h, latent_w = 1, 8, 12

        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        text_encoding, text_mask = create_dummy_text_encoding("Test prompt")
        mx.random.seed(42)
        latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)
        output_shape = VideoLatentShape(1, 128, latent_f, latent_h, latent_w)

        # Run 3 steps
        for step_idx in range(3):
            latent_patchified = patchifier.patchify(latent)
            modality = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([float(sigmas[step_idx])]),
                positions=positions,
                enabled=True,
            )
            velocity = patchifier.unpatchify(model(modality), output_shape=output_shape)
            latent = euler.step(latent, velocity, sigmas, step_idx)
            mx.eval(latent)

        # Decode
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)

        # Check pixel values
        min_val = int(video.min())
        max_val = int(video.max())
        mean_val = float(video.astype(mx.float32).mean())

        verbose_logger.log_step(f"Pixel range: [{min_val}, {max_val}]")
        verbose_logger.log_step(f"Mean value: {mean_val:.2f}")

        # Verify valid range
        assert min_val >= 0, f"Min pixel value {min_val} < 0"
        assert max_val <= 255, f"Max pixel value {max_val} > 255"
        assert video.dtype == mx.uint8, f"Expected uint8, got {video.dtype}"

        # Verify not all same value (degenerate output)
        assert max_val > min_val, "All pixels have same value (degenerate output)"

        verbose_logger.log_info("Pixel range test passed")

        del model, vae_decoder
        mx.metal.clear_cache()
        gc.collect()


# =============================================================================
# Real Prompt Generation Tests (Requires Gemma Weights)
# =============================================================================

# Prompts designed to produce visually distinct outputs
REAL_TEST_PROMPTS = [
    "A calm blue ocean with gentle waves under a clear sky",
    "A red sports car driving on a mountain road at sunset",
    "A white cat sitting on a green garden bench surrounded by flowers",
    "Rain falling on a city street with reflections and umbrellas",
    "A golden wheat field swaying in the wind under dramatic clouds",
]


class TestRealPromptGeneration:
    """Test video generation with real Gemma text encoding.

    These tests use the full Gemma 3 text encoder to generate semantically
    meaningful text embeddings. Videos should reflect the prompt content.
    """

    @requires_weights
    @requires_gemma
    def test_generate_with_real_prompt(self, verbose_logger):
        """Generate video with actual text encoding from Gemma and CFG."""
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, EulerDiffusionStep
        from LTX_2_MLX.types import VideoLatentShape

        ensure_output_dir()

        prompt = "A calm blue ocean with gentle waves under a clear sky"
        cfg_scale = 3.0  # CFG scale for guidance
        verbose_logger.log_step(f"Real prompt test: {prompt[:50]}...")
        verbose_logger.log_step(f"CFG scale: {cfg_scale}")

        # Encode with Gemma
        verbose_logger.log_step("Encoding text with Gemma 3...")
        start_time = time.time()
        text_encoding, text_mask = encode_text_with_gemma(prompt)
        encode_time = time.time() - start_time
        verbose_logger.log_step(f"Text encoding took {encode_time:.2f}s")
        verbose_logger.log_step(f"Text encoding shape: {text_encoding.shape}")

        # Create proper null encoding for CFG (encoded empty string, not zeros)
        verbose_logger.log_step("Encoding empty string for CFG unconditional...")
        null_encoding, null_mask = encode_null_text_for_cfg(
            max_length=text_encoding.shape[1],
        )

        # Load models
        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()

        verbose_logger.log_step("Loading transformer...")
        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        verbose_logger.log_step("Loading VAE decoder...")
        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        # Config
        latent_f, latent_h, latent_w = 1, 8, 12  # 256x384, single frame
        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        # Initialize latent
        mx.random.seed(42)
        latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)
        output_shape = VideoLatentShape(1, 128, latent_f, latent_h, latent_w)

        # Full denoising loop with CFG
        num_steps = len(sigmas) - 1
        verbose_logger.log_step(f"Running {num_steps} denoising steps with CFG...")

        start_time = time.time()
        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            latent_patchified = patchifier.patchify(latent)

            # Unconditional pass (null encoding)
            modality_uncond = Modality(
                latent=latent_patchified,
                context=null_encoding,
                context_mask=null_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )
            velocity_uncond = patchifier.unpatchify(model(modality_uncond), output_shape=output_shape)
            mx.eval(velocity_uncond)

            # Conditional pass (text encoding)
            modality_cond = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )
            velocity_cond = patchifier.unpatchify(model(modality_cond), output_shape=output_shape)
            mx.eval(velocity_cond)

            # CFG formula: v = v_uncond + scale * (v_cond - v_uncond)
            velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)

            latent = euler.step(latent, velocity, sigmas, step_idx)
            mx.eval(latent)

        denoise_time = time.time() - start_time
        verbose_logger.log_step(f"Denoising took {denoise_time:.2f}s")

        # Decode
        verbose_logger.log_step("Decoding with VAE...")
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)

        # Save
        frames = [np.array(video[f]) for f in range(video.shape[0])]
        output_path = str(OUTPUT_DIR / "real_prompt_ocean.mp4")
        save_video_frames(frames, output_path)

        assert os.path.exists(output_path)
        verbose_logger.log_info(f"Real prompt video saved to {output_path}")

        # Cleanup
        del model, vae_decoder
        mx.metal.clear_cache()
        gc.collect()

    @requires_weights
    @requires_gemma
    def test_multiple_real_prompts(self, verbose_logger):
        """Generate videos with multiple different real prompts using CFG.

        Tests that different prompts produce distinct outputs.
        """
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, EulerDiffusionStep
        from LTX_2_MLX.types import VideoLatentShape

        ensure_output_dir()
        cfg_scale = 3.0
        verbose_logger.log_step(f"Testing multiple real prompts with CFG={cfg_scale}")

        # Load models once
        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()

        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        latent_f, latent_h, latent_w = 1, 8, 12
        output_shape = VideoLatentShape(1, 128, latent_f, latent_h, latent_w)
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)

        # Create proper null encoding for CFG (encoded empty string)
        verbose_logger.log_step("Encoding empty string for CFG unconditional...")
        null_encoding, null_mask = encode_null_text_for_cfg(max_length=256)

        # Test subset of prompts
        prompts = REAL_TEST_PROMPTS[:3]
        outputs = []

        for i, prompt in enumerate(prompts):
            verbose_logger.log_step(f"Prompt {i+1}/{len(prompts)}: {prompt[:40]}...")

            # Encode with Gemma
            text_encoding, text_mask = encode_text_with_gemma(prompt)
            verbose_logger.log_step(f"  Encoded shape: {text_encoding.shape}")

            # Initialize noise with same seed for fair comparison
            mx.random.seed(42)
            latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))

            # Full denoising with CFG (7 steps for proper results)
            num_steps = len(sigmas) - 1
            for step_idx in range(num_steps):
                sigma = float(sigmas[step_idx])
                latent_patchified = patchifier.patchify(latent)

                # Unconditional pass
                modality_uncond = Modality(
                    latent=latent_patchified,
                    context=null_encoding,
                    context_mask=null_mask,
                    timesteps=mx.array([sigma]),
                    positions=positions,
                    enabled=True,
                )
                velocity_uncond = patchifier.unpatchify(model(modality_uncond), output_shape=output_shape)
                mx.eval(velocity_uncond)

                # Conditional pass
                modality_cond = Modality(
                    latent=latent_patchified,
                    context=text_encoding,
                    context_mask=text_mask,
                    timesteps=mx.array([sigma]),
                    positions=positions,
                    enabled=True,
                )
                velocity_cond = patchifier.unpatchify(model(modality_cond), output_shape=output_shape)
                mx.eval(velocity_cond)

                # CFG formula
                velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
                latent = euler.step(latent, velocity, sigmas, step_idx)
                mx.eval(latent)

            # Decode
            video = decode_latent(latent, vae_decoder)
            mx.eval(video)

            # Save
            frames = [np.array(video[f]) for f in range(video.shape[0])]
            output_path = str(OUTPUT_DIR / f"real_prompt_{i+1}.mp4")
            save_video_frames(frames, output_path)
            verbose_logger.log_step(f"  Saved to {output_path}")

            # Store mean color for comparison
            mean_color = np.array(video).mean(axis=(0, 1, 2))
            outputs.append(mean_color)
            verbose_logger.log_step(f"  Mean RGB: {mean_color}")

        # Verify outputs are different (prompts should produce distinct colors)
        # Note: With the same seed, differences are smaller since noise initialization dominates
        # The threshold of 1.0 ensures outputs are not identical while accounting for
        # the limited semantic influence of text encoding with 7 distilled steps
        total_diff = 0.0
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diff = np.abs(outputs[i] - outputs[j]).sum()
                verbose_logger.log_step(f"  Color diff {i+1} vs {j+1}: {diff:.2f}")
                total_diff += diff

        # At minimum, total difference across all pairs should show prompt influence
        verbose_logger.log_step(f"  Total color diff: {total_diff:.2f}")
        assert total_diff > 1.0, f"All outputs are too similar (total_diff={total_diff:.2f})"

        verbose_logger.log_info("Multiple real prompts test passed")

        # Cleanup
        del model, vae_decoder
        mx.metal.clear_cache()
        gc.collect()

    @requires_weights
    @requires_gemma
    @pytest.mark.slow
    def test_real_prompt_longer_video(self, verbose_logger):
        """Generate longer video (9 frames) with real text encoding."""
        from LTX_2_MLX.model.transformer import LTXModel, Modality
        from LTX_2_MLX.model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
        from LTX_2_MLX.loader import load_transformer_weights
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, EulerDiffusionStep
        from LTX_2_MLX.types import VideoLatentShape

        ensure_output_dir()

        prompt = "A golden wheat field swaying in the wind under dramatic clouds"
        verbose_logger.log_step(f"Longer real video: {prompt[:50]}...")

        # Encode
        verbose_logger.log_step("Encoding text with Gemma 3...")
        text_encoding, text_mask = encode_text_with_gemma(prompt)

        # Load models
        weights_path = get_available_weights()
        assert weights_path is not None
        use_fp8 = "fp8" in weights_path.name.lower()

        model = LTXModel(compute_dtype=mx.float16, low_memory=True)
        load_transformer_weights(model, str(weights_path), use_fp8=use_fp8)

        vae_decoder = SimpleVideoDecoder(compute_dtype=mx.float16)
        load_vae_decoder_weights(vae_decoder, str(weights_path))

        # Config for longer video
        num_frames = 9
        height, width = 256, 384
        latent_f = (num_frames - 1) // 8 + 1  # = 2
        latent_h = height // 32
        latent_w = width // 32

        verbose_logger.log_step(f"Generating {num_frames} frames (latent_f={latent_f})")

        patchifier = VideoLatentPatchifier(patch_size=1)
        euler = EulerDiffusionStep()
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        mx.random.seed(42)
        latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))
        positions = create_position_grid_with_bounds(1, latent_f, latent_h, latent_w)
        output_shape = VideoLatentShape(1, 128, latent_f, latent_h, latent_w)

        # Full denoising
        num_steps = len(sigmas) - 1
        verbose_logger.log_step(f"Running {num_steps} denoising steps...")

        start_time = time.time()
        for step_idx in range(num_steps):
            latent_patchified = patchifier.patchify(latent)

            modality = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([float(sigmas[step_idx])]),
                positions=positions,
                enabled=True,
            )

            velocity = patchifier.unpatchify(model(modality), output_shape=output_shape)
            latent = euler.step(latent, velocity, sigmas, step_idx)
            mx.eval(latent)

        denoise_time = time.time() - start_time
        verbose_logger.log_step(f"Denoising took {denoise_time:.2f}s")

        # Decode
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)
        verbose_logger.log_step(f"Output video shape: {video.shape}")

        # Save
        frames = [np.array(video[f]) for f in range(video.shape[0])]
        output_path = str(OUTPUT_DIR / "real_prompt_wheat_field.mp4")
        save_video_frames(frames, output_path)

        assert os.path.exists(output_path)
        verbose_logger.log_info(f"Longer real video saved to {output_path}")

        # Cleanup
        del model, vae_decoder
        mx.metal.clear_cache()
        gc.collect()


# =============================================================================
# Run Function
# =============================================================================

def run_video_generation_tests():
    """Run all video generation tests manually."""
    print("\n=== Video Generation Tests ===\n")

    logger = VerboseTestLogger("video_generation")

    if not weights_available():
        print(f"SKIPPED: Model weights not found in {LTX2_WEIGHTS_DIR}")
        return

    ensure_output_dir()
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Simple generation test
    print("--- Simple Generation Test ---")
    simple_tests = TestSimpleGeneration()
    simple_tests.test_generate_short_video(logger)

    # Multiple prompts test
    print("\n--- Multiple Prompts Test ---")
    prompts_tests = TestMultiplePrompts()
    prompts_tests.test_generate_with_prompts(logger)

    # Quality test
    print("\n--- Quality Verification Test ---")
    quality_tests = TestVideoQuality()
    quality_tests.test_output_pixel_range(logger)

    # Real prompt tests (require Gemma)
    if gemma_weights_available():
        print("\n--- Real Prompt Generation Tests ---")
        real_tests = TestRealPromptGeneration()
        real_tests.test_generate_with_real_prompt(logger)
        real_tests.test_multiple_real_prompts(logger)
    else:
        print(f"\n--- Skipping real prompt tests (Gemma not found in {GEMMA_WEIGHTS_DIR}) ---")

    print("\n=== All Video Generation Tests Passed! ===\n")
    print(f"Generated videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_video_generation_tests()
