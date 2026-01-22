#!/usr/bin/env python3
"""Generate video from text prompt using LTX-2 MLX."""

import argparse
import gc
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.transformer import (
    LTXAVModel,
    LTXModel,
    LTXModelType,
    Modality,
    MultiModalTransformerArgsPreprocessor,
    X0Model,
)
from LTX_2_MLX.components.patchifiers import get_pixel_coords
from LTX_2_MLX.types import SpatioTemporalScaleFactors
from LTX_2_MLX.model.audio_vae import (
    AudioDecoder,
    Vocoder,
    load_audio_decoder_weights,
    load_vocoder_weights,
)
from LTX_2_MLX.model.video_vae import VideoDecoder, NormLayerType
from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, get_sigma_schedule
from LTX_2_MLX.components.guiders import LtxAPGGuider, LegacyStatefulAPGGuider, STGGuider
from LTX_2_MLX.components.perturbations import create_batched_stg_config
from LTX_2_MLX.types import VideoLatentShape
from LTX_2_MLX.loader import load_transformer_weights, load_av_transformer_weights, LoRAConfig
from LTX_2_MLX.loader.lora_loader import fuse_lora_into_weights
from mlx.utils import tree_flatten
from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    decode_latent,
)
from LTX_2_MLX.core_utils import to_velocity


def batched_cfg_forward(
    model,
    latent_patchified: mx.array,
    text_encoding: mx.array,
    text_mask: mx.array,
    null_encoding: mx.array,
    null_mask: mx.array,
    sigma: float,
    positions: mx.array,
) -> tuple:
    """
    Run CFG forward pass with batched inputs (2x speedup).

    Instead of two separate forward passes for cond and uncond,
    we stack them along the batch dimension and do a single pass.

    Returns:
        Tuple of (cond_output, uncond_output) both shape [1, T, C]
    """
    # Stack along batch dimension: [1, T, C] -> [2, T, C]
    batched_latent = mx.concatenate([latent_patchified, latent_patchified], axis=0)
    batched_context = mx.concatenate([text_encoding, null_encoding], axis=0)
    batched_mask = mx.concatenate([text_mask, null_mask], axis=0)
    batched_positions = mx.concatenate([positions, positions], axis=0)
    batched_timesteps = mx.array([sigma, sigma])

    # Single batched modality
    # NOTE: context_mask=None matches PyTorch behavior - they don't use text masks
    batched_modality = Modality(
        latent=batched_latent,
        context=batched_context,
        context_mask=None,
        timesteps=batched_timesteps,
        positions=batched_positions,
        enabled=True,
    )

    # Single forward pass (2x faster than two separate passes)
    batched_output = model(batched_modality)

    # Split back: [2, T, C] -> two [1, T, C]
    cond_output = batched_output[0:1]
    uncond_output = batched_output[1:2]

    return cond_output, uncond_output
from LTX_2_MLX.model.text_encoder.gemma3 import (
    Gemma3Config,
    Gemma3Model,
    load_gemma3_weights,
)
from LTX_2_MLX.model.text_encoder.encoder import (
    create_text_encoder,
    load_text_encoder_weights,
    create_av_text_encoder,
    load_av_text_encoder_weights,
)
from LTX_2_MLX.model.upscaler import (
    SpatialUpscaler,
    load_spatial_upscaler_weights,
    TemporalUpscaler,
    load_temporal_upscaler_weights,
)
from LTX_2_MLX.pipelines.two_stage import (
    TwoStagePipeline,
    TwoStageCFGConfig,
)
from LTX_2_MLX.pipelines.one_stage import (
    OneStagePipeline,
    OneStageCFGConfig,
)
from LTX_2_MLX.pipelines.common import ImageCondition
from LTX_2_MLX.pipelines.ic_lora import (
    ControlType,
    VideoCondition,
    ICLoraPipeline,
    ICLoraConfig,
    preprocess_control_signal,
    load_control_signal_tensor,
)
from LTX_2_MLX.pipelines.keyframe_interpolation import (
    KeyframeInterpolationPipeline,
    KeyframeInterpolationConfig,
    Keyframe,
)
from LTX_2_MLX.model.video_vae.simple_encoder import (
    SimpleVideoEncoder,
    load_vae_encoder_weights,
)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


def progress_bar(iterable, desc=None, total=None):
    """Create a progress bar wrapper."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, ncols=80)
    else:
        return _simple_progress(iterable, desc, total)


def _simple_progress(iterable, desc, total):
    """Simple progress fallback when tqdm is not available."""
    items = list(iterable)
    total = len(items) if total is None else total
    for i, item in enumerate(items):
        print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
        yield item
    print()  # newline after completion


# LTX-2 system prompt for video generation (used during encoding)
T2V_SYSTEM_PROMPT = """Describe the video in extreme detail, focusing on the visual content, without any introductory phrases."""

# System prompt for prompt enhancement (used to expand short prompts into detailed descriptions)
ENHANCE_SYSTEM_PROMPT = """You are a creative assistant that transforms simple video descriptions into detailed, vivid prompts for video generation.

Given a brief description, expand it into a rich, detailed paragraph (100-200 words) that includes:
- Specific visual details (colors, textures, lighting)
- Motion and action descriptions (speed, direction, style)
- Environment and atmosphere details
- Camera perspective if relevant

Keep the description in natural, flowing language. Focus only on visual elements.
Do not add dialogue or sound descriptions unless specifically mentioned.
Output only the enhanced description, nothing else."""


def load_tokenizer(model_path: str):
    """Load the Gemma tokenizer from HuggingFace transformers."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer
    except ImportError as e:
        print(f"Error: transformers library required for tokenizer: {e}")
        print("Install with: pip install transformers")
        return None


def create_chat_prompt(user_prompt: str) -> str:
    """Create a chat-format prompt for Gemma 3."""
    # Gemma 3 instruction-tuned format
    chat = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    return chat


def enhance_prompt(
    prompt: str,
    gemma_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Enhance a short prompt into a detailed description using Gemma generation.

    The official LTX-2 pipeline uses this to expand simple prompts like "A blue ball"
    into rich descriptions with visual details, creating more differentiated embeddings.

    Args:
        prompt: Short user prompt to enhance.
        gemma_path: Path to Gemma 3 weights directory.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.7 recommended for creativity).

    Returns:
        Enhanced prompt string.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("    Warning: PyTorch/transformers not installed, skipping prompt enhancement")
        return prompt

    print(f"  Enhancing prompt with Gemma (via transformers)...")
    print(f"    Original: {prompt}")

    # Create enhancement chat prompt
    enhance_chat = f"<bos><start_of_turn>user\n{ENHANCE_SYSTEM_PROMPT}\n\nPrompt to enhance: {prompt}<end_of_turn>\n<start_of_turn>model\n"

    # Load model and tokenizer via transformers
    print(f"    Loading Gemma 3 for generation...")
    tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    model = AutoModelForCausalLM.from_pretrained(
        gemma_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Tokenize and generate
    inputs = tokenizer(enhance_chat, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip input)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    enhanced = tokenizer.decode(generated_ids, skip_special_tokens=True)
    enhanced = enhanced.strip()

    # Clean up model from memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc
    gc.collect()

    print(f"    Enhanced: {enhanced[:100]}..." if len(enhanced) > 100 else f"    Enhanced: {enhanced}")

    return enhanced if enhanced else prompt


def encode_with_gemma(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    max_length: int = 1024,
    use_early_layers_only: bool = False,
) -> tuple:
    """
    Encode a text prompt using the full Gemma 3 + LTX-2 text encoder pipeline.

    Args:
        prompt: Text prompt to encode.
        gemma_path: Path to Gemma 3 weights directory.
        ltx_weights_path: Path to LTX-2 weights (for text encoder projection).
        max_length: Maximum token length.
        use_early_layers_only: If True, use only Layer 0 (input embeddings) to
            preserve token differentiation. Gemma's self-attention homogenizes
            representations by Layer 4, making different prompts indistinguishable.
            Layer 0 preserves ~0.4 correlation at differing tokens vs ~0.999+ at Layer 4+.

    Returns:
        Tuple of (embedding, attention_mask) as MLX arrays.
    """
    print(f"  Loading tokenizer from {gemma_path}...")
    tokenizer = load_tokenizer(gemma_path)
    if tokenizer is None:
        return None, None

    # Match PyTorch tokenizer behavior: left padding with EOS as pad token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    # IMPORTANT: Use FP32 for Gemma - the model was trained with BF16 which has
    # the same exponent range as FP32. Using FP16 causes numerical overflow due to
    # the large RMSNorm weights (mean ~6.9x scale factor).
    load_gemma3_weights(gemma, gemma_path, use_fp16=False)

    print(f"  Loading text encoder projection...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, ltx_weights_path)

    # Tokenize prompt directly (skip chat template - it dilutes the signal)
    # Chat template adds ~28 shared tokens, diluting the actual content
    # Without template: 0.71 correlation for blue vs red (good)
    # With template: 0.98 correlation (bad - template tokens dominate)
    print(f"  Tokenizing prompt...")
    encoding = tokenizer(
        prompt,  # Use raw prompt, not chat template
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = mx.array(encoding["input_ids"])
    attention_mask = mx.array(encoding["attention_mask"])

    num_tokens = int(attention_mask.sum())
    print(f"  Token count: {num_tokens}/{max_length}")

    # Run through Gemma to get hidden states
    print(f"  Running Gemma 3 forward pass (48 layers)...")
    last_hidden, all_hidden_states = gemma(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mx.eval(last_hidden)

    if all_hidden_states is None:
        print("  Error: Gemma model did not return hidden states")
        return None, None

    print(f"  Got {len(all_hidden_states)} hidden states")

    # EXPERIMENTAL: Use only early layers to preserve differentiation
    if use_early_layers_only:
        print("  [EXPERIMENTAL] Using only Layer 0 (input embeddings)...")
        # Layer 0 is the input embeddings before any self-attention
        # This preserves ~0.4 correlation at differing tokens instead of ~0.999+
        encoded = all_hidden_states[0]  # [B, T, 3840]

        # Zero out padding positions
        mask_expanded = attention_mask[:, :, None].astype(encoded.dtype)
        encoded = encoded * mask_expanded

        original_mask = attention_mask.astype(mx.int32)
        mx.eval(encoded)
        mx.eval(original_mask)

        print(f"  Output embedding shape: {encoded.shape}")  # [B, T, 3840]

    else:
        # Run through text encoder pipeline
        # Note: We skip caption_projection here because the transformer has its own
        print(f"  Processing through text encoder pipeline...")

        # Feature extraction (uses Layer 48 only for best differentiation)
        encoded = text_encoder.feature_extractor.extract_from_hidden_states(
            hidden_states=all_hidden_states,
            attention_mask=attention_mask,
            padding_side="left",
        )

        # Use connector (1D transformer with learnable registers)
        # Earlier testing showed connector homogenizes embeddings, but the model
        # may have been trained to expect connector output format
        print(f"  Processing through connector...")

        # Convert mask to additive format for connector attention
        connector_mask = (attention_mask.astype(encoded.dtype) - 1) * 1e9
        connector_mask = connector_mask.reshape(
            attention_mask.shape[0], 1, 1, attention_mask.shape[-1]
        )

        encoded, output_mask = text_encoder.embeddings_connector(encoded, connector_mask)
        mx.eval(encoded)

        # Convert mask back to binary for cross-attention
        original_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)

        # Zero out padding positions
        encoded = encoded * original_mask[:, :, None]
        mx.eval(encoded)
        mx.eval(original_mask)

        print(f"  Output embedding shape: {encoded.shape}")  # Should be [B, T+registers, 3840]

    # === MEMORY OPTIMIZATION ===
    # Clear Gemma and text encoder from memory after encoding
    # These are large models (~12GB for Gemma FP16) that are no longer needed
    print(f"  Clearing Gemma from memory...")
    del gemma
    del text_encoder
    del all_hidden_states
    del last_hidden
    del tokenizer
    gc.collect()
    # Force MLX to release memory
    mx.metal.clear_cache()

    return encoded, original_mask


def encode_with_av_gemma(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    max_length: int = 1024,
) -> tuple:
    """
    Encode a text prompt using the AudioVideo Gemma 3 + LTX-2 text encoder pipeline.

    This returns both video and audio encodings, which are processed through
    separate embedding connectors.

    Args:
        prompt: Text prompt to encode.
        gemma_path: Path to Gemma 3 weights directory.
        ltx_weights_path: Path to LTX-2 AudioVideo weights (for text encoder projection).
        max_length: Maximum token length.

    Returns:
        Tuple of (video_encoding, audio_encoding, attention_mask) as MLX arrays.
    """
    print(f"  Loading tokenizer from {gemma_path}...")
    tokenizer = load_tokenizer(gemma_path)
    if tokenizer is None:
        return None, None, None

    # Match PyTorch tokenizer behavior: left padding with EOS as pad token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    # IMPORTANT: Use FP32 for Gemma - the model was trained with BF16 which has
    # the same exponent range as FP32. Using FP16 causes numerical overflow.
    load_gemma3_weights(gemma, gemma_path, use_fp16=False)

    print(f"  Loading AV text encoder projection...")
    text_encoder = create_av_text_encoder()
    load_av_text_encoder_weights(text_encoder, ltx_weights_path)

    # Tokenize prompt directly (skip chat template - it dilutes the signal)
    print(f"  Tokenizing prompt...")
    encoding = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = mx.array(encoding["input_ids"])
    attention_mask = mx.array(encoding["attention_mask"])

    num_tokens = int(attention_mask.sum())
    print(f"  Token count: {num_tokens}/{max_length}")

    # Run through Gemma to get hidden states
    print(f"  Running Gemma 3 forward pass (48 layers)...")
    last_hidden, all_hidden_states = gemma(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mx.eval(last_hidden)

    if all_hidden_states is None:
        print("  Error: Gemma model did not return hidden states")
        return None, None, None

    print(f"  Got {len(all_hidden_states)} hidden states")

    # Run through AV text encoder pipeline
    print(f"  Processing through AV text encoder pipeline...")
    av_output = text_encoder.encode_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=attention_mask,
        padding_side="left",
    )
    mx.eval(av_output.video_encoding)
    mx.eval(av_output.audio_encoding)

    print(f"  Video encoding shape: {av_output.video_encoding.shape}")
    print(f"  Audio encoding shape: {av_output.audio_encoding.shape}")

    # === MEMORY OPTIMIZATION ===
    # Clear Gemma and text encoder from memory after encoding
    print(f"  Clearing Gemma from memory...")
    del gemma
    del text_encoder
    del all_hidden_states
    del last_hidden
    del tokenizer
    gc.collect()
    # Force MLX to release memory
    mx.metal.clear_cache()

    return av_output.video_encoding, av_output.audio_encoding, av_output.attention_mask


def create_dummy_text_encoding(
    prompt: str,
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,  # Pre-projection dimension (transformer does its own projection)
) -> tuple:
    """
    Create dummy text encoding for testing.

    In production, this should be replaced with actual Gemma encoding.
    Note: Output is 3840-dim because the transformer has its own caption_projection.
    """
    # For now, use random but deterministic encoding based on prompt
    mx.random.seed(hash(prompt) % (2**31))

    # Create text embeddings in pre-projection dimension
    text_encoding = mx.random.normal(shape=(batch_size, max_tokens, embed_dim)) * 0.1
    text_mask = mx.ones((batch_size, max_tokens))

    return text_encoding, text_mask


def create_null_text_encoding(
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,
) -> tuple:
    """
    Create null/empty text encoding for CFG unconditional pass.

    WARNING: This creates zero embeddings which is NOT semantically correct
    for CFG. For proper CFG, the unconditional embedding should be the
    encoding of an empty string through the text encoder. Use
    encode_with_gemma("") when the encoder is available.

    Returns:
        Tuple of (null_encoding, null_mask).
    """
    # Zero embeddings - NOTE: not ideal for CFG, but works as a fallback
    # Proper CFG should use encoded empty string from text encoder
    null_encoding = mx.zeros((batch_size, max_tokens, embed_dim))
    null_mask = mx.zeros((batch_size, max_tokens))  # All masked out

    return null_encoding, null_mask


def rescale_noise_cfg(
    noise_cfg: mx.array,
    noise_pred_text: mx.array,
    guidance_rescale: float = 0.7,
) -> mx.array:
    """
    Rescale CFG output to prevent variance explosion.

    Based on Section 3.4 of "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    (https://arxiv.org/abs/2305.08891). This rescales the CFG output to match the
    variance of the conditional prediction, preventing over-saturation.

    Args:
        noise_cfg: The CFG-combined output (uncond + scale * (cond - uncond)).
        noise_pred_text: The conditional prediction (before CFG).
        guidance_rescale: Factor for blending rescaled vs original CFG.
                         0.0 = no rescaling (original CFG), 1.0 = full rescaling.

    Returns:
        Rescaled CFG output.
    """
    # Per-channel rescaling to fix per-channel biases
    # Shape: [B, C, F, H, W] - compute stats per channel
    # This is more aggressive than standard guidance rescale

    # Compute per-channel mean and std for both predictions
    # Using RMS (root mean square) for std to avoid issues with zero mean
    cfg_mean = mx.mean(noise_cfg, axis=(2, 3, 4), keepdims=True)  # [B, C, 1, 1, 1]
    cfg_std = mx.sqrt(mx.mean((noise_cfg - cfg_mean) ** 2, axis=(2, 3, 4), keepdims=True) + 1e-8)

    text_mean = mx.mean(noise_pred_text, axis=(2, 3, 4), keepdims=True)
    text_std = mx.sqrt(mx.mean((noise_pred_text - text_mean) ** 2, axis=(2, 3, 4), keepdims=True) + 1e-8)

    # Normalize CFG to have same per-channel mean and std as conditional
    noise_pred_rescaled = (noise_cfg - cfg_mean) / cfg_std * text_std + text_mean

    # Blend between original and rescaled based on guidance_rescale factor
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

    return noise_cfg


def load_text_embedding(embedding_path: str) -> tuple:
    """
    Load pre-computed text embedding from file.

    Args:
        embedding_path: Path to .npz file with embedding and attention_mask.

    Returns:
        Tuple of (embedding, attention_mask).
    """
    data = np.load(embedding_path)
    embedding = mx.array(data["embedding"])
    mask = mx.array(data["attention_mask"])

    print(f"  Loaded embedding from {embedding_path}")
    print(f"  Shape: {embedding.shape}")

    if "prompt" in data:
        print(f"  Original prompt: {data['prompt']}")

    return embedding, mask


def load_vae_decoder(weights_path: str) -> VideoDecoder:
    """Load VAE decoder with weights."""
    print("Loading VAE decoder...")

    # LTX-2 decoder configuration
    decoder_blocks = [
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 4}),
        ("compress_time", {}),
        ("res_x", {"num_layers": 4}),
        ("compress_space", {}),
        ("res_x", {"num_layers": 4}),
        ("compress_space", {}),
    ]

    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
    )

    # TODO: Load VAE weights from file
    print("  VAE decoder created (weights not loaded yet)")

    return decoder


def load_transformer(
    weights_path: str,
    num_layers: int = 48,
    compute_dtype: mx.Dtype = mx.float32,
    use_fp8: bool = False,
    low_memory: bool = False,
    fast_mode: bool = False,
) -> LTXModel:
    """Load transformer with weights.

    Args:
        weights_path: Path to safetensors weights file.
        num_layers: Number of transformer layers.
        compute_dtype: Dtype for computation.
        use_fp8: If True, load FP8 weights and dequantize.
        low_memory: If True, use aggressive memory optimization.
        fast_mode: If True, skip intermediate evaluations.
    """
    dtype_name = "FP16" if compute_dtype == mx.float16 else "FP32"
    fp8_str = " (FP8 dequantized)" if use_fp8 else ""
    mem_str = " (low memory)" if low_memory else ""
    fast_str = " (fast mode)" if fast_mode else ""
    print(f"Loading transformer ({dtype_name}{fp8_str}{mem_str}{fast_str})...")

    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=num_layers,
        cross_attention_dim=4096,
        caption_channels=3840,
        positional_embedding_theta=10000.0,
        compute_dtype=compute_dtype,
        low_memory=low_memory,
        fast_mode=fast_mode,
    )

    # Load weights
    if weights_path and os.path.exists(weights_path):
        target_dtype = "float16" if compute_dtype == mx.float16 else "float32"
        load_transformer_weights(model, weights_path, use_fp8=use_fp8, target_dtype=target_dtype)
    else:
        print(f"  Warning: Weights not found at {weights_path}, using random init")

    return model


def load_av_transformer(
    weights_path: str,
    num_layers: int = 48,
    compute_dtype: mx.Dtype = mx.float32,
    use_fp8: bool = False,
    low_memory: bool = False,
) -> LTXAVModel:
    """Load AudioVideo transformer with weights."""
    dtype_name = "FP16" if compute_dtype == mx.float16 else "FP32"
    fp8_str = " (FP8 dequantized)" if use_fp8 else ""
    mem_str = " (low memory)" if low_memory else ""
    print(f"Loading AudioVideo transformer ({dtype_name}{fp8_str}{mem_str})...")

    model = LTXAVModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=num_layers,
        cross_attention_dim=4096,
        caption_channels=3840,
        positional_embedding_theta=10000.0,
        compute_dtype=compute_dtype,
        low_memory=low_memory,
    )

    # Load weights (including audio components)
    if weights_path and os.path.exists(weights_path):
        target_dtype = "float16" if compute_dtype == mx.float16 else "float32"
        load_av_transformer_weights(model, weights_path, use_fp8=use_fp8, target_dtype=target_dtype)
    else:
        print(f"  Warning: Weights not found at {weights_path}, using random init")

    return model


def euler_step(
    latent: mx.array,
    velocity: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """
    Simple Euler step with direct velocity (for placeholder/testing only).

    NOTE: For proper inference, use euler_step_x0 with denoised prediction.
    """
    dt = sigma_next - sigma
    return latent + dt * velocity


def euler_step_x0(
    sample: mx.array,
    denoised: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """
    Perform one Euler diffusion step using X0 (denoised) prediction.

    This matches the PyTorch EulerDiffusionStep.step() which:
    1. Takes denoised sample (not velocity)
    2. Converts to velocity: v = (sample - denoised) / sigma
    3. Applies Euler: x_next = x + dt * v

    Args:
        sample: Current noisy sample
        denoised: Predicted denoised sample (x0)
        sigma: Current noise level
        sigma_next: Next noise level
    """
    # Convert denoised to velocity (matches PyTorch reference)
    velocity = to_velocity(sample, sigma, denoised)

    # Euler step
    dt = sigma_next - sigma
    return sample.astype(mx.float32) + velocity.astype(mx.float32) * dt


def generate_video(
    prompt: str,
    height: int = 480,
    width: int = 704,
    num_frames: int = 97,  # 12 seconds at 8fps after VAE (97 = 1 + 12*8)
    num_steps: int = 7,  # Distilled model uses 7 steps
    cfg_scale: float = 5.0,  # Updated default for better semantic quality
    guidance_rescale: float = 0.7,  # Rescale CFG output to prevent variance explosion
    # Two-stage pipeline parameters
    steps_stage1: int = 15,
    steps_stage2: int = 3,
    cfg_stage1: float | None = None,  # Defaults to cfg_scale if not specified
    seed: int = 42,
    weights_path: str | None = None,
    output_path: str = "gens/output.mp4",
    use_placeholder: bool = False,
    skip_vae: bool = False,
    embedding_path: str | None = None,
    gemma_path: str = "weights/gemma-3-12b",
    use_gemma: bool = True,
    use_fp16: bool = True,  # FP16 by default for memory efficiency
    use_fp8: bool = False,
    model_variant: str = "distilled",
    upscale_spatial: bool = False,
    spatial_upscaler_weights: str = None,
    upscale_temporal: bool = False,
    temporal_upscaler_weights: str = None,
    generate_audio: bool = False,
    low_memory: bool = False,
    fast_mode: bool = False,
    # New parameters
    image_path: str = None,
    image_strength: float = 0.95,
    lora_path: str = None,
    lora_strength: float = 1.0,
    tiled_vae: bool = False,
    pipeline_type: str = "text-to-video",
    early_layers_only: bool = False,
    enhance_prompt_flag: bool = False,
    cross_attn_scale: float = 1.0,
    distilled_lora: str = None,
    distilled_lora_scale: float = 1.0,
    stg_scale: float = 0.0,
    stg_mode: str = "video",
    apg_scale: float = 1.0,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    apg_momentum: float = 0.0,
    control_video: str = None,
    control_type: str = "raw",
    canny_low: int = 100,
    canny_high: int = 200,
    control_strength: float = 0.95,
    save_control: bool = False,
    ge_gamma: float = 0.0,
    output_fps: int = 24,
    output_speed: float = 1.0,
    # IC-LoRA and Keyframe Interpolation
    keyframes: list = None,
    ic_lora_weights: str = None,
):
    """Generate video from text prompt."""

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine compute dtype - FP16 by default for memory efficiency (~50% reduction)
    compute_dtype = mx.float16 if use_fp16 else mx.float32

    print(f"\n{'='*50}")
    print(f"LTX-2 MLX Video Generation")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}, {num_frames} frames")
    print(f"Steps: {num_steps}, CFG: {cfg_scale}, Seed: {seed}")
    print(f"Model variant: {model_variant}")
    if use_fp8:
        print(f"Weights: FP8 quantized (dequantized at load time)")
    if use_fp16:
        print(f"Compute dtype: FP16 (memory optimized)")
    if skip_vae:
        print(f"VAE decoding: SKIPPED")
    if upscale_spatial:
        print(f"Spatial upscaling: 2x (output will be {width*2}x{height*2})")
    if upscale_temporal:
        print(f"Temporal upscaling: 2x (frames will be ~{num_frames*2})")
    if generate_audio:
        print(f"Audio generation: ENABLED (stereo 24kHz)")
    if low_memory:
        print(f"Low memory mode: ENABLED (sequential CFG, aggressive eval)")
    if fast_mode:
        print(f"Fast mode: ENABLED (no intermediate evals)")
    if stg_scale > 0:
        print(f"STG guidance: scale={stg_scale}, mode={stg_mode}")
    if apg_scale != 1.0:
        print(f"APG guidance: scale={apg_scale}, eta={apg_eta}, norm_threshold={apg_norm_threshold}")
        if apg_momentum > 0:
            print(f"  Using stateful APG with momentum={apg_momentum}")
    if control_video:
        print(f"Control video: {control_video} (type={control_type}, strength={control_strength})")
        if control_type == "canny":
            print(f"  Canny thresholds: low={canny_low}, high={canny_high}")
    if ge_gamma > 0:
        print(f"GE denoising: gamma={ge_gamma} (velocity correction enabled)")
    if embedding_path:
        print(f"Using pre-computed embedding: {embedding_path}")
    elif use_gemma:
        print(f"Text encoder: Gemma 3 at {gemma_path}")
    else:
        print(f"Text encoder: DUMMY (testing mode)")

    # Set seed
    mx.random.seed(seed)

    # Compute latent dimensions
    # VAE: 32x spatial, 8x temporal compression
    latent_height = height // 32
    latent_width = width // 32
    latent_frames = (num_frames - 1) // 8 + 1

    print(f"\nLatent shape: {latent_frames}x{latent_height}x{latent_width}")

    # Enhance prompt if requested (expands short prompts to detailed descriptions)
    if enhance_prompt_flag and use_gemma:
        print("\n[0/5] Enhancing prompt...")
        prompt = enhance_prompt(prompt, gemma_path)
        print(f"  Using enhanced prompt for generation")

    # Get text encoding
    # Initialize audio encodings (used only when generate_audio=True)
    text_audio_encoding = None
    null_audio_encoding = None

    print("\n[1/5] Encoding prompt...")
    if embedding_path:
        text_encoding, text_mask = load_text_embedding(embedding_path)
        if generate_audio:
            print("  WARNING: Pre-computed embeddings don't include audio encoding. Audio quality may be degraded.")
            text_audio_encoding = text_encoding  # Fallback: use video encoding for audio
    elif use_gemma:
        # Check if Gemma weights exist
        if not os.path.exists(gemma_path):
            print(f"\n  ERROR: Gemma weights not found at {gemma_path}")
            print(f"\n  To download Gemma 3 12B:")
            print(f"    python scripts/download_gemma.py")
            print(f"\n  Or use --no-gemma flag to use dummy embeddings for testing")
            return

        if generate_audio:
            # Use AudioVideo Gemma encoder (returns both video and audio encodings)
            text_encoding, text_audio_encoding, text_mask = encode_with_av_gemma(
                prompt=prompt,
                gemma_path=gemma_path,
                ltx_weights_path=weights_path,
            )
            if text_encoding is None:
                print("  ERROR: Failed to encode prompt with AV encoder")
                return
            print(f"  Encoded with Gemma 3 (AudioVideo)")
        else:
            # Use video-only Gemma encoding
            text_encoding, text_mask = encode_with_gemma(
                prompt=prompt,
                gemma_path=gemma_path,
                ltx_weights_path=weights_path,
                use_early_layers_only=early_layers_only,
            )
            if text_encoding is None:
                print("  ERROR: Failed to encode prompt")
                return
            print(f"  Encoded with Gemma 3")
    else:
        text_encoding, text_mask = create_dummy_text_encoding(prompt)
        if generate_audio:
            text_audio_encoding = text_encoding  # Fallback for dummy mode
        print("  Using DUMMY encoding (test mode - output will be random)")

    # Create null encoding for CFG (unconditional)
    # For proper CFG, encode empty string through text encoder (not zeros)
    if use_gemma and gemma_path and os.path.exists(gemma_path):
        print("  Encoding empty prompt for CFG unconditional...")
        if generate_audio:
            # Use AudioVideo encoder for null encoding too
            null_encoding, null_audio_encoding, null_mask = encode_with_av_gemma(
                prompt="",  # Empty string for unconditional
                gemma_path=gemma_path,
                ltx_weights_path=weights_path,
                max_length=text_encoding.shape[1],
            )
            if null_encoding is None:
                print("  WARNING: Failed to encode empty prompt with AV encoder, using zeros fallback")
                null_encoding, null_mask = create_null_text_encoding(
                    batch_size=1,
                    max_tokens=text_encoding.shape[1],
                    embed_dim=text_encoding.shape[2],
                )
                null_audio_encoding = null_encoding  # Fallback
        else:
            null_encoding, null_mask = encode_with_gemma(
                prompt="",  # Empty string for unconditional
                gemma_path=gemma_path,
                ltx_weights_path=weights_path,
                max_length=text_encoding.shape[1],
                use_early_layers_only=early_layers_only,
            )
            if null_encoding is None:
                print("  WARNING: Failed to encode empty prompt, using zeros fallback")
                null_encoding, null_mask = create_null_text_encoding(
                    batch_size=1,
                    max_tokens=text_encoding.shape[1],
                    embed_dim=text_encoding.shape[2],
                )
    else:
        # Fallback to zeros when Gemma is not available
        null_encoding, null_mask = create_null_text_encoding(
            batch_size=1,
            max_tokens=text_encoding.shape[1],
            embed_dim=text_encoding.shape[2],
        )
        if generate_audio:
            null_audio_encoding = null_encoding  # Fallback

    # Load model
    if generate_audio:
        print("\n[2/5] Loading AudioVideo transformer...")
        if not use_placeholder and weights_path:
            model = load_av_transformer(weights_path, num_layers=48, compute_dtype=compute_dtype, use_fp8=use_fp8, low_memory=low_memory)
        else:
            model = None
            print("  Skipping model load (placeholder mode)")
    else:
        print("\n[2/5] Loading transformer...")
        if not use_placeholder and weights_path:
            velocity_model = load_transformer(weights_path, num_layers=48, compute_dtype=compute_dtype, use_fp8=use_fp8, low_memory=low_memory, fast_mode=fast_mode)

            # Apply cross-attention scaling if specified (improves text conditioning)
            if cross_attn_scale != 1.0:
                velocity_model.set_cross_attn_scale(cross_attn_scale, start_layer=40)
                print(f"  Applied cross-attention scale {cross_attn_scale}x for layers 40-47")

            # Wrap in X0Model to convert velocity predictions to denoised (X0)
            # The raw LTXModel outputs velocity, but denoising expects X0 predictions
            model = X0Model(velocity_model)
            print("  Wrapped model with X0Model for denoised predictions")
        else:
            model = None
            print("  Skipping model load (placeholder mode)")

    # Apply LoRA if provided
    if lora_path and model is not None:
        print(f"\n  Applying LoRA from {lora_path} (strength={lora_strength})")
        lora_config = LoRAConfig(path=lora_path, strength=lora_strength)

        # Get target model (handle X0Model wrapper)
        if hasattr(model, 'velocity_model'):
            target_model = model.velocity_model
        else:
            target_model = model

        # Fuse LoRA weights into model
        flat_params = dict(tree_flatten(target_model.parameters()))
        fused_weights = fuse_lora_into_weights(flat_params, [lora_config])
        target_model.load_weights(list(fused_weights.items()))
        mx.eval(target_model.parameters())
        print(f"  LoRA applied successfully")

    # Whether to use CFG
    # Distilled models (LTX-2 distilled) are trained without CFG and produce artifacts if CFG > 1.0
    # HOWEVER: Two-stage pipeline specifically uses CFG in Stage 1 (at low res), so we allow it there.
    if model_variant == "distilled" and cfg_scale > 1.2 and pipeline_type != "two-stage":
        print(f"  WARNING: Distilled model requires CFG=1.0 (no guidance). You requested {cfg_scale}.")
        print(f"  Forcing CFG=1.0 to prevent visual artifacts.")
        cfg_scale = 1.0
        # Disable guidance rescale too since it's irrelevant without CFG
        if guidance_rescale > 0:
            guidance_rescale = 0.0

    use_cfg = cfg_scale > 1.0 and model is not None
    if use_cfg:
        print(f"  CFG enabled with scale {cfg_scale}")
        if guidance_rescale > 0:
            print(f"  Guidance rescale: {guidance_rescale}")
    else:
        print(f"  CFG disabled (scale {cfg_scale}) - Running optimized single-pass inference")

    # Create APG guider if enabled (replaces CFG when active)
    apg_guider = None
    if apg_scale != 1.0:
        if apg_momentum > 0:
            apg_guider = LegacyStatefulAPGGuider(
                scale=apg_scale,
                eta=apg_eta,
                norm_threshold=apg_norm_threshold,
                momentum=apg_momentum,
            )
        else:
            apg_guider = LtxAPGGuider(
                scale=apg_scale,
                eta=apg_eta,
                norm_threshold=apg_norm_threshold,
            )
        print(f"  APG guidance enabled (replaces standard CFG)")

    # Create STG guider if enabled
    stg_guider = None
    if stg_scale > 0:
        stg_guider = STGGuider(scale=stg_scale)
        print(f"  STG guidance enabled (scale={stg_scale})")

    # Load VAE decoder
    vae_decoder = None
    if not skip_vae:
        print(f"\n[3/5] Loading VAE decoder...")
        vae_decoder = SimpleVideoDecoder(compute_dtype=compute_dtype)
        if weights_path and not use_placeholder:
             load_vae_decoder_weights(vae_decoder, weights_path)
        elif use_placeholder:
             print("  Skipping weights load (placeholder)")
    else:
        print("\n[3/5] VAE decoder skipped by user")

    # === TWO-STAGE PIPELINE ===
    # Use dedicated two-stage pipeline for higher quality generation
    if pipeline_type == "two-stage":
        # Validate requirements
        if generate_audio:
            raise ValueError("Two-stage pipeline does not support audio generation mode. Use --pipeline text-to-video instead.")
        
        if model is None:
            if use_placeholder:
                print("  Creating dummy model for placeholder Two-Stage pipeline...")
                class MockModel:
                    def __init__(self):
                        self.velocity_model = self
                    def parameters(self):
                        return {}
                    def __call__(self, *args, **kwargs):
                        return mx.zeros((1))
                    def load_weights(self, *args):
                        pass
                model = MockModel()
            else:
                raise ValueError("Two-stage pipeline requires a loaded model (cannot use placeholder mode)")

        if vae_decoder is None and not use_placeholder:
            raise ValueError("Two-stage pipeline requires VAE decoder")
        
        if (not spatial_upscaler_weights or not weights_path) and not use_placeholder:
            raise ValueError("Two-stage pipeline requires --spatial-upscaler-weights")

        # Two-stage pipeline requires resolution divisible by 64 (for stage 1 half-res to be divisible by 32)
        if height % 64 != 0 or width % 64 != 0:
            new_height = ((height + 63) // 64) * 64
            new_width = ((width + 63) // 64) * 64
            print(f"  WARNING: Two-stage pipeline requires resolution divisible by 64.")
            print(f"  Adjusting resolution from {height}x{width} to {new_height}x{new_width}")
            height = new_height
            width = new_width

        print("\n=== Using Two-Stage Pipeline ===")
        print(f"  Stage 1: {steps_stage1} steps at {height//2}x{width//2} with CFG {cfg_stage1 or cfg_scale}")
        if guidance_rescale > 0:
            print(f"  Guidance rescale: {guidance_rescale}")
        print(f"  Stage 2: 3 steps at {height}x{width} (distilled refinement)")

        # Load spatial upscaler
        print("\n[3.5/5] Loading spatial upscaler...")
        spatial_upscaler = SpatialUpscaler()
        if not use_placeholder:
            load_spatial_upscaler_weights(spatial_upscaler, spatial_upscaler_weights)
        else:
            print("  Skipping weights load (placeholder)")

        # DEBUG: Check if weights are loaded
        print(f"DEBUG: initial_conv_weight stats - mean: {float(mx.mean(spatial_upscaler.initial_conv_weight)):.6f}, std: {float(mx.std(spatial_upscaler.initial_conv_weight.astype(mx.float32))):.6f}")
        print(f"DEBUG: final_conv_weight stats - mean: {float(mx.mean(spatial_upscaler.final_conv_weight)):.6f}, std: {float(mx.std(spatial_upscaler.final_conv_weight.astype(mx.float32))):.6f}")

        # Load video encoder
        print("[3.5/5] Loading VAE encoder...")
        video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
        if not use_placeholder:
            load_vae_encoder_weights(video_encoder, weights_path)
        else:
            print("  Skipping weights load (placeholder)")

        # Create two-stage pipeline
        print("\n[4/5] Creating two-stage pipeline...")
        pipeline = TwoStagePipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            spatial_upscaler=spatial_upscaler,
        )

        # Create distilled LoRA config if provided
        distilled_lora_config = None
        if distilled_lora:
            print(f"  Distilled LoRA: {distilled_lora} (scale {distilled_lora_scale})")
            distilled_lora_config = LoRAConfig(path=distilled_lora, strength=distilled_lora_scale)
        elif pipeline_type == "two-stage":
             print("  WARNING: No distilled LoRA provided for two-stage pipeline. Stage 2 quality may be degraded.")

        # Create config
        config = TwoStageCFGConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=24.0,
            num_inference_steps=steps_stage1,
            cfg_scale=cfg_stage1 if cfg_stage1 is not None else cfg_scale,
            guidance_rescale=guidance_rescale,
            dtype=compute_dtype,
            distilled_lora_config=distilled_lora_config,
        )

        # Create image conditionings if provided
        images = []
        if image_path:
            print(f"  Image conditioning: {image_path} (strength={image_strength})")
            images = [ImageCondition(
                image_path=image_path,
                frame_index=0,
                strength=image_strength,
            )]

        # Run pipeline
        print(f"\n[5/5] Running two-stage generation...")
        video, _ = pipeline(
            positive_encoding=text_encoding,
            negative_encoding=null_encoding,
            config=config,
            images=images,
        )

        # Convert to frames list for save_video
        # decode_latent returns (T, H, W, C) in uint8, so just convert to numpy list
        video_np = np.array(video)  # (T, H, W, C)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        # Save video
        print(f"\nSaving video to {output_path}...")
        save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
        return

    # === IC-LORA PIPELINE ===
    # Use ICLoraPipeline for video-to-video or image-to-video generation with control signals
    if pipeline_type == "ic-lora":
        if not control_video and not image_path:
            raise ValueError("IC-LoRA pipeline requires --control-video or --image argument")

        print("\n=== Using IC-LoRA Pipeline ===")
        if control_video:
            print(f"  Control video: {control_video}")
            print(f"  Control type: {control_type}")
            print(f"  Control strength: {control_strength}")
        if image_path:
            print(f"  Image conditioning: {image_path} (strength={image_strength})")

        if model is None:
            if use_placeholder:
                print("  IC-LoRA requires model - cannot use placeholder mode")
                return
            raise ValueError("IC-LoRA pipeline requires a loaded model")

        if vae_decoder is None and not use_placeholder:
            raise ValueError("IC-LoRA pipeline requires VAE decoder")

        # Load VAE encoder
        print("[3.5/5] Loading VAE encoder...")
        video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
        if weights_path and not use_placeholder:
            load_vae_encoder_weights(video_encoder, weights_path)
        else:
            print("  Skipping weights load (placeholder)")

        # Load spatial upscaler
        print("[3.6/5] Loading spatial upscaler...")
        spatial_upscaler = SpatialUpscaler()
        upscaler_path = spatial_upscaler_weights or "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        if os.path.exists(upscaler_path):
            load_spatial_upscaler_weights(spatial_upscaler, upscaler_path)
        else:
            print(f"  Warning: Spatial upscaler weights not found at {upscaler_path}")

        # Get base transformer weights for restoration after stage 1
        if hasattr(model, 'velocity_model'):
            base_weights = dict(tree_flatten(model.velocity_model.parameters()))
        else:
            base_weights = dict(tree_flatten(model.parameters()))

        # Prepare LoRA configs if provided
        lora_configs = None
        if ic_lora_weights:
            print(f"  IC-LoRA weights: {ic_lora_weights}")
            lora_configs = [LoRAConfig(path=ic_lora_weights, strength=1.0)]

        # Create IC-LoRA pipeline
        print("\n[4/5] Creating IC-LoRA pipeline...")
        ic_pipeline = ICLoraPipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            spatial_upscaler=spatial_upscaler,
            base_transformer_weights=base_weights,
            lora_configs=lora_configs,
        )

        # Create config
        config = ICLoraConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=24.0,
            stage_1_steps=num_steps,
            dtype=compute_dtype,
        )

        # Create video conditioning if control video provided
        video_conditioning = []
        if control_video:
            ctrl_type = ControlType.CANNY if control_type == "canny" else ControlType.RAW
            video_cond = VideoCondition(
                video_path=control_video,
                strength=control_strength,
                control_type=ctrl_type,
                canny_low=canny_low,
                canny_high=canny_high,
                save_control=save_control,
            )
            video_conditioning = [video_cond]

        # Create image conditioning if provided (IC-LoRA supports both video and image conditioning)
        images = []
        if image_path:
            images = [ImageCondition(
                image_path=image_path,
                frame_index=0,
                strength=image_strength,
            )]

        # Run pipeline
        print(f"\n[5/5] Running IC-LoRA generation...")
        video = ic_pipeline(
            text_encoding=text_encoding,
            text_mask=mx.ones((1, text_encoding.shape[1]), dtype=mx.int32),
            config=config,
            images=images,
            video_conditioning=video_conditioning,
        )

        # Convert to frames
        video_np = np.array(video)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        # Save video
        print(f"\nSaving video to {output_path}...")
        save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
        return

    # === KEYFRAME INTERPOLATION PIPELINE ===
    # Use KeyframeInterpolationPipeline for interpolating between keyframe images
    if pipeline_type == "keyframe-interpolation":
        if not keyframes:
            raise ValueError("Keyframe interpolation pipeline requires --keyframe arguments")

        print("\n=== Using Keyframe Interpolation Pipeline ===")

        # Parse keyframes
        parsed_keyframes = []
        for kf_str in keyframes:
            parts = kf_str.split(":")
            if len(parts) < 2:
                raise ValueError(f"Invalid keyframe format: {kf_str}. Use 'path:frame_index' or 'path:frame_index:strength'")
            path = parts[0]
            frame_idx = int(parts[1])
            strength = float(parts[2]) if len(parts) > 2 else 0.95
            parsed_keyframes.append(Keyframe(image_path=path, frame_index=frame_idx, strength=strength))
            print(f"  Keyframe: {path} at frame {frame_idx} (strength={strength})")

        if model is None:
            if use_placeholder:
                print("  Keyframe interpolation requires model - cannot use placeholder mode")
                return
            raise ValueError("Keyframe interpolation pipeline requires a loaded model")

        if vae_decoder is None and not use_placeholder:
            raise ValueError("Keyframe interpolation pipeline requires VAE decoder")

        # Load VAE encoder
        print("[3.5/5] Loading VAE encoder...")
        video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
        if weights_path and not use_placeholder:
            load_vae_encoder_weights(video_encoder, weights_path)
        else:
            print("  Skipping weights load (placeholder)")

        # Load spatial upscaler for two-stage
        print("[3.6/5] Loading spatial upscaler...")
        spatial_upscaler = SpatialUpscaler()
        upscaler_path = spatial_upscaler_weights or "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        if os.path.exists(upscaler_path):
            load_spatial_upscaler_weights(spatial_upscaler, upscaler_path)
        else:
            print(f"  Warning: Spatial upscaler weights not found at {upscaler_path}")

        # Create keyframe interpolation pipeline
        print("\n[4/5] Creating keyframe interpolation pipeline...")
        kf_pipeline = KeyframeInterpolationPipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            spatial_upscaler=spatial_upscaler,
        )

        # Create config
        config = KeyframeInterpolationConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=24.0,
            num_inference_steps=num_steps,
            cfg_scale=cfg_scale,
            dtype=compute_dtype,
        )

        # Run pipeline
        print(f"\n[5/5] Running keyframe interpolation ({num_steps} steps)...")
        video = kf_pipeline(
            text_encoding=text_encoding,
            text_mask=mx.ones((1, text_encoding.shape[1]), dtype=mx.int32),
            keyframes=parsed_keyframes,
            config=config,
            negative_text_encoding=null_encoding,
            negative_text_mask=mx.ones((1, null_encoding.shape[1]), dtype=mx.int32),
        )

        # Convert to frames
        video_np = np.array(video)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        # Save video
        print(f"\nSaving video to {output_path}...")
        save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
        return

    # === AUDIO-VIDEO PIPELINE ===
    # Use OneStagePipeline for joint audio-video generation
    if generate_audio:
        print("\n=== Using Audio-Video Pipeline ===")

        if model is None:
            if use_placeholder:
                print("  Audio generation requires model - cannot use placeholder mode")
                return
            raise ValueError("Audio generation requires a loaded AudioVideo model")

        if vae_decoder is None and not use_placeholder:
            raise ValueError("Audio generation requires VAE decoder")

        # Load VAE encoder (needed for image conditioning)
        print("[3.5/5] Loading VAE encoder...")
        video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
        if weights_path and not use_placeholder:
            load_vae_encoder_weights(video_encoder, weights_path)
        else:
            print("  Skipping weights load (placeholder)")

        # Load audio components
        print("  Loading Audio VAE decoder...")
        audio_decoder = AudioDecoder(compute_dtype=compute_dtype)
        if weights_path:
            load_audio_decoder_weights(audio_decoder, weights_path)

        print("  Loading Vocoder...")
        vocoder = Vocoder(compute_dtype=compute_dtype)
        if weights_path:
            load_vocoder_weights(vocoder, weights_path)

        # Create one-stage pipeline with audio support
        print("\n[4/5] Creating audio-video pipeline...")
        av_pipeline = OneStagePipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
        )

        # Create config with audio enabled
        av_config = OneStageCFGConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=24.0,
            num_inference_steps=num_steps,
            cfg_scale=cfg_scale,
            dtype=compute_dtype,
            audio_enabled=True,
        )

        # Create image conditionings if provided
        images = []
        if image_path:
            print(f"  Image conditioning: {image_path} (strength={image_strength})")
            images = [ImageCondition(
                image_path=image_path,
                frame_index=0,
                strength=image_strength,
            )]

        # Run pipeline with audio
        print(f"\n[5/5] Running audio-video generation ({num_steps} steps)...")

        def progress_callback(step: int, total: int):
            print(f"\r  Denoising: {step}/{total}", end="", flush=True)

        video, audio_waveform = av_pipeline(
            positive_encoding=text_encoding,
            negative_encoding=null_encoding,
            config=av_config,
            images=images,
            callback=progress_callback,
            positive_audio_encoding=text_audio_encoding,
            negative_audio_encoding=null_audio_encoding,
        )
        print()  # newline after progress

        # Convert to frames list for save_video
        video_np = np.array(video)  # (T, H, W, C)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        if audio_waveform is not None:
            print(f"  Generated audio: {audio_waveform.shape}")

        # Save video with audio
        print(f"\nSaving video to {output_path}...")
        if audio_waveform is not None:
            save_video_with_audio(frames, audio_waveform, output_path, fps=output_fps, speed=output_speed)
        else:
            save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
        return

    # === STANDARD PIPELINE (one-stage, distilled, video-only) ===
    # Note: Audio generation is handled by the AUDIO-VIDEO PIPELINE above
    # Initialize noise
    print("\n[4/5] Initializing latent noise...")
    latent = mx.random.normal(shape=(1, 128, latent_frames, latent_height, latent_width))

    # Get sigma schedule based on model variant
    # The distilled model was trained with specific sigma values
    use_linear_schedule = False  # Use distilled values for distilled model
    if use_linear_schedule:
        # Linear schedule: evenly spaced from 1.0 to 0.0
        # Better for spatial coherence preservation
        sigmas = mx.array(np.linspace(1.0, 0.0, num_steps + 1).tolist())
        print(f"  Sigma schedule (linear): {[f'{float(s):.3f}' for s in sigmas]}")
    elif model_variant == "distilled":
        sigmas = mx.array(DISTILLED_SIGMA_VALUES[:num_steps + 1])
        print(f"  Sigma schedule (distilled): {[f'{float(s):.3f}' for s in sigmas]}")
    else:
        # Dev model uses LTX2Scheduler for dynamic schedule
        sigmas = get_sigma_schedule(num_steps=num_steps, distilled=False, latent=latent)
        print(f"  Sigma schedule (dev): {[f'{float(s):.3f}' for s in sigmas]}")

    # Create patchifier
    patchifier = VideoLatentPatchifier(patch_size=1)

    print(f"\n[5/5] Denoising ({num_steps} steps)...")

    # Denoising loop with progress bar
    step_iterator = progress_bar(range(len(sigmas) - 1), desc="Denoising", total=num_steps)

    # GE (Gradient Estimation) velocity tracking
    prev_velocity = None

    for i in step_iterator:
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        if model is not None and not use_placeholder:
            # === Actual model inference ===
            # Patchify video latent: [B, C, F, H, W] -> [B, T, C]
            latent_patchified = patchifier.patchify(latent)

            # Create video position grid with proper pixel-space coordinates
            # (matching the distilled pipeline format)
            output_shape = VideoLatentShape(
                batch=1,
                channels=128,
                frames=latent_frames,
                height=latent_height,
                width=latent_width,
            )
            latent_coords = patchifier.get_patch_grid_bounds(output_shape=output_shape)
            scale_factors = SpatioTemporalScaleFactors.default()  # time=8, height=32, width=32
            positions = get_pixel_coords(
                latent_coords=latent_coords,
                scale_factors=scale_factors,
                causal_fix=True,
            ).astype(mx.float32)
            # Convert temporal positions from frames to seconds
            fps = 24.0
            temporal_positions = positions[:, 0:1, ...] / fps
            other_positions = positions[:, 1:, ...]
            positions = mx.concatenate([temporal_positions, other_positions], axis=1)

            # === Video-only mode with X0 prediction ===
            # Note: Audio mode is handled by the AUDIO-VIDEO PIPELINE section above
            # The distilled LTX-2 model directly outputs X0 (denoised samples),
            # NOT velocity. So we:
            # 1. Get X0 directly from model
            # 2. Apply CFG on X0 samples
            # 3. Euler step with X0
            # (output_shape already defined above for position creation)

            # Apply CFG if enabled
            if use_cfg:
                if low_memory:
                    # MEMORY OPTIMIZATION: Sequential CFG passes
                    # Run unconditional first, eval, then conditional

                    # Unconditional (null text) pass first
                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_uncond = Modality(
                        latent=latent_patchified,
                        context=null_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_uncond_patchified = model(modality_uncond)
                    denoised_uncond = patchifier.unpatchify(x0_uncond_patchified, output_shape=output_shape)
                    mx.eval(denoised_uncond)
                    del x0_uncond_patchified

                    # Conditional (text-guided) pass
                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_cond = Modality(
                        latent=latent_patchified,
                        context=text_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_cond_patchified = model(modality_cond)
                    denoised_cond = patchifier.unpatchify(x0_cond_patchified, output_shape=output_shape)
                    mx.eval(denoised_cond)
                    del x0_cond_patchified

                    # Apply guidance: APG if enabled, otherwise standard CFG
                    if apg_guider is not None and apg_guider.enabled():
                        denoised = apg_guider.guide(denoised_cond, denoised_uncond)
                    else:
                        # CFG formula on X0: x0 = x0_uncond + scale * (x0_cond - x0_uncond)
                        denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

                    # Apply guidance rescale to prevent variance explosion
                    if guidance_rescale > 0:
                        denoised = rescale_noise_cfg(denoised, denoised_cond, guidance_rescale)

                    # Apply STG (Spatio-Temporal Guidance) if enabled
                    if stg_guider is not None and stg_guider.enabled():
                        # Run perturbed forward pass (skip video self-attention)
                        x0_perturbed_patchified = model(modality_cond, perturbations=create_batched_stg_config(batch_size=1))
                        denoised_perturbed = patchifier.unpatchify(x0_perturbed_patchified, output_shape=output_shape)
                        denoised = stg_guider.guide(denoised, denoised_perturbed)
                        del denoised_perturbed, x0_perturbed_patchified

                    del denoised_uncond, denoised_cond
                else:
                    # Standard CFG: Sequential forward passes
                    # NOTE: Batched CFG (stacking cond+uncond) was tested but found SLOWER
                    # for 19B models because GPU is already fully utilized with batch=1.
                    # Doubling batch just doubles compute time with no throughput gain.
                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_cond = Modality(
                        latent=latent_patchified,
                        context=text_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_cond_patchified = model(modality_cond)
                    denoised_cond = patchifier.unpatchify(x0_cond_patchified, output_shape=output_shape)

                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_uncond = Modality(
                        latent=latent_patchified,
                        context=null_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_uncond_patchified = model(modality_uncond)
                    denoised_uncond = patchifier.unpatchify(x0_uncond_patchified, output_shape=output_shape)

                    # Apply guidance: APG if enabled, otherwise standard CFG
                    if apg_guider is not None and apg_guider.enabled():
                        denoised = apg_guider.guide(denoised_cond, denoised_uncond)
                    else:
                        # CFG formula on X0: x0 = x0_uncond + scale * (x0_cond - x0_uncond)
                        denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

                    # Apply guidance rescale to prevent variance explosion
                    if guidance_rescale > 0:
                        denoised = rescale_noise_cfg(denoised, denoised_cond, guidance_rescale)

                    # Apply STG (Spatio-Temporal Guidance) if enabled
                    if stg_guider is not None and stg_guider.enabled():
                        # Run perturbed forward pass (skip video self-attention)
                        x0_perturbed_patchified = model(modality_cond, perturbations=create_batched_stg_config(batch_size=1))
                        denoised_perturbed = patchifier.unpatchify(x0_perturbed_patchified, output_shape=output_shape)
                        denoised = stg_guider.guide(denoised, denoised_perturbed)
            else:
                # No CFG - just conditional pass
                # NOTE: context_mask=None matches PyTorch behavior
                modality_cond = Modality(
                    latent=latent_patchified,
                    context=text_encoding,
                    context_mask=None,
                    timesteps=mx.array([sigma]),
                    positions=positions,
                    enabled=True,
                )
                x0_cond_patchified = model(modality_cond)
                denoised = patchifier.unpatchify(x0_cond_patchified, output_shape=output_shape)

                # Apply STG (Spatio-Temporal Guidance) if enabled (works without CFG)
                if stg_guider is not None and stg_guider.enabled():
                    # Run perturbed forward pass (skip video self-attention)
                    x0_perturbed_patchified = model(modality_cond, perturbations=create_batched_stg_config(batch_size=1))
                    denoised_perturbed = patchifier.unpatchify(x0_perturbed_patchified, output_shape=output_shape)
                    denoised = stg_guider.guide(denoised, denoised_perturbed)

            # Apply GE (Gradient Estimation) velocity correction if enabled
            if ge_gamma > 0:
                # Compute current velocity: v = (x - x0) / sigma
                current_velocity = (latent - denoised) / sigma

                if prev_velocity is not None:
                    # Apply velocity correction using momentum-like update
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    # Reconstruct corrected denoised: x0 = x - v * sigma
                    denoised = latent - total_velocity * sigma

                # Update velocity for next iteration
                prev_velocity = current_velocity

            # Euler step using X0 (denoised) prediction
            latent = euler_step_x0(latent, denoised, sigma, sigma_next)

            # Force evaluation for memory efficiency
            mx.eval(latent)
        else:
            # Placeholder: random velocity
            velocity = mx.random.normal(shape=latent.shape) * 0.1
            latent = euler_step(latent, velocity, sigma, sigma_next)
            mx.eval(latent)

    # === MEMORY OPTIMIZATION ===
    # Clear transformer from memory - no longer needed after denoising
    print("\n  Clearing transformer from memory...")
    del model
    gc.collect()
    mx.metal.clear_cache()

    # Save denoised latent
    latent_path = output_path.replace('.mp4', '_latent.npz')
    print(f"\nSaving denoised latent to {latent_path}...")
    np.savez(latent_path, latent=np.array(latent))
    print(f"  Latent shape: {latent.shape}")

    # Apply spatial upscaling if requested
    if upscale_spatial and spatial_upscaler_weights:
        print("\nApplying 2x spatial upscaling...")
        print(f"  Input latent: {latent.shape}")

        # Load upscaler
        spatial_upscaler = SpatialUpscaler()
        load_spatial_upscaler_weights(spatial_upscaler, spatial_upscaler_weights)

        # CRITICAL: Un-normalize before upsampling, re-normalize after
        # The upsampler model is trained on raw (un-normalized) latents
        # Reference: PyTorch upsample_video() in ltx_core/model/upsampler/model.py
        if vae_decoder is not None:
            # Un-normalize: latent_raw = latent * std + mean
            std = vae_decoder.std_of_means.reshape(1, -1, 1, 1, 1)
            mean = vae_decoder.mean_of_means.reshape(1, -1, 1, 1, 1)
            latent_unnorm = latent * std + mean
            print(f"  Un-normalized: std={float(mx.std(latent_unnorm)):.3f}")

            # Upscale the un-normalized latent
            latent_upscaled = spatial_upscaler(latent_unnorm)
            mx.eval(latent_upscaled)

            # Re-normalize: latent = (latent_raw - mean) / std
            latent = (latent_upscaled - mean) / std
            mx.eval(latent)
            print(f"  Re-normalized: std={float(mx.std(latent)):.3f}")
        else:
            # Fallback: upscale directly (may have incorrect dynamic range)
            print("  WARNING: No VAE decoder for normalization - output may have wrong range")
            latent = spatial_upscaler(latent)
            mx.eval(latent)

        print(f"  Upscaled latent: {latent.shape}")

        # Clear upscaler from memory
        del spatial_upscaler
        gc.collect()
        mx.metal.clear_cache()

    # Apply temporal upscaling if requested
    if upscale_temporal and temporal_upscaler_weights:
        print("\nApplying 2x temporal upscaling...")
        print(f"  Input latent: {latent.shape}")

        # Load upscaler
        temporal_upscaler = TemporalUpscaler()
        load_temporal_upscaler_weights(temporal_upscaler, temporal_upscaler_weights)

        # CRITICAL: Un-normalize before upsampling, re-normalize after
        # The upsampler model is trained on raw (un-normalized) latents
        # Reference: PyTorch upsample_video() in ltx_core/model/upsampler/model.py
        if vae_decoder is not None:
            # Un-normalize: latent_raw = latent * std + mean
            std = vae_decoder.std_of_means.reshape(1, -1, 1, 1, 1)
            mean = vae_decoder.mean_of_means.reshape(1, -1, 1, 1, 1)
            latent_unnorm = latent * std + mean
            print(f"  Un-normalized: std={float(mx.std(latent_unnorm)):.3f}")

            # Upscale the un-normalized latent
            latent_upscaled = temporal_upscaler(latent_unnorm)
            mx.eval(latent_upscaled)

            # Re-normalize: latent = (latent_raw - mean) / std
            latent = (latent_upscaled - mean) / std
            mx.eval(latent)
            print(f"  Re-normalized: std={float(mx.std(latent)):.3f}")
        else:
            # Fallback: upscale directly (may have incorrect dynamic range)
            print("  WARNING: No VAE decoder for normalization - output may have wrong range")
            latent = temporal_upscaler(latent)
            mx.eval(latent)

        print(f"  Upscaled latent: {latent.shape}")

        # Clear upscaler from memory
        del temporal_upscaler
        gc.collect()
        mx.metal.clear_cache()

    # Decode with VAE or create placeholder
    if vae_decoder is not None:
        print("\nDecoding with VAE...")
        print(f"  Input latent: {latent.shape}")

        # VAE decode
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)
        print(f"  Output video: {video.shape}")

        # Convert to frames list for save_video
        frames = [np.array(video[f]) for f in range(video.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

    else:
        print("\nCreating placeholder video (VAE not loaded)...")

        # Placeholder output - create simple visualization based on latent statistics
        frames = []
        latent_np = np.array(latent[0])  # (C, F, H, W)
        latent_mean = latent_np.mean(axis=0)  # (F, H, W)
        latent_std = latent_np.std(axis=0)

        frame_iterator = progress_bar(range(num_frames), desc="Creating frames", total=num_frames)

        for f in frame_iterator:
            # Map latent frame to visualization
            lat_f = f * latent_frames // num_frames
            lat_f = min(lat_f, latent_frames - 1)

            # Get latent statistics for this frame
            lat_slice_mean = latent_mean[lat_f]  # (H, W)
            lat_slice_std = latent_std[lat_f]

            # Create frame based on latent (upscale from latent to output resolution)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Simple bilinear-ish upscale of latent visualization
            for y in range(height):
                for x in range(width):
                    lat_y = y * latent_height // height
                    lat_x = x * latent_width // width
                    lat_y = min(lat_y, latent_height - 1)
                    lat_x = min(lat_x, latent_width - 1)

                    # Use latent values to create RGB
                    val_mean = float(lat_slice_mean[lat_y, lat_x])
                    val_std = float(lat_slice_std[lat_y, lat_x])

                    # Normalize and convert to color
                    r = int(np.clip((val_mean + 2) / 4 * 255, 0, 255))
                    g = int(np.clip((val_std) / 2 * 255, 0, 255))
                    b = int(np.clip((val_mean * val_std + 1) / 2 * 128, 0, 255))

                    frame[y, x] = [r, g, b]

            frames.append(frame)

    # Save video
    # Note: Audio generation is handled by the AUDIO-VIDEO PIPELINE section above
    print(f"\nSaving video to {output_path}...")
    save_video(frames, output_path, fps=output_fps, speed=output_speed)

    print(f"\nDone! Video saved to {output_path}")

    if use_placeholder:
        print("\nNote: This is a placeholder output. Full inference requires:")
        print("  1. Proper weight loading (use --weights flag)")
        print("  2. Gemma text encoder integration")

    if vae_decoder is None and not skip_vae:
        print("\nNote: VAE decoder was not loaded - output is placeholder visualization.")


def save_video(frames: list, output_path: str, fps: int = 24, speed: float = 1.0):
    """Save frames as video using ffmpeg with optional interpolation and speed adjustment.

    Args:
        frames: List of frame arrays (H, W, C) in uint8.
        output_path: Output video file path.
        fps: Target output frame rate. If >24, uses motion interpolation.
        speed: Playback speed multiplier (0.5=slow-mo, 1.0=normal, 2.0=fast).
    """
    import subprocess
    import tempfile
    from PIL import Image

    NATIVE_FPS = 24  # Model generates motion at 24fps

    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as images with progress
        print("  Writing frames...")
        if HAS_TQDM:
            iterator = tqdm(enumerate(frames), desc="  Saving frames", total=len(frames), ncols=80)
        else:
            iterator = enumerate(frames)

        for i, frame in iterator:
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        # Build ffmpeg filter chain
        filters = []

        # Speed adjustment (applied first, before interpolation)
        # setpts: lower value = faster, higher value = slower
        if speed != 1.0:
            # speed=2.0 means 2x faster, so PTS should be halved
            pts_multiplier = 1.0 / speed
            filters.append(f"setpts={pts_multiplier}*PTS")

        # Frame interpolation if target fps > native
        if fps > NATIVE_FPS:
            # minterpolate creates smooth intermediate frames
            # mi_mode=mci: motion compensated interpolation
            # mc_mode=aobmc: adaptive overlapped block motion compensation
            # me_mode=bidir: bidirectional motion estimation
            filters.append(f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")

        # Build ffmpeg command
        print("\n  Encoding video...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(NATIVE_FPS),  # Input is always at native 24fps
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
        ]

        # Add filter chain if needed
        if filters:
            filter_str = ",".join(filters)
            cmd.extend(["-vf", filter_str])
            if fps > NATIVE_FPS:
                print(f"  Interpolating {NATIVE_FPS}fps → {fps}fps (speed: {speed}x)")
            elif speed != 1.0:
                print(f"  Applying speed: {speed}x")

        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-loglevel", "error",
            output_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def save_video_with_audio(
    frames: list,
    audio_waveform: mx.array,
    output_path: str,
    fps: int = 24,
    speed: float = 1.0,
    audio_sample_rate: int = 24000,
):
    """Save frames as video with audio using ffmpeg with optional interpolation and speed.

    Args:
        frames: List of frame arrays (H, W, C) in uint8.
        audio_waveform: Audio waveform tensor (B, 2, samples).
        output_path: Output video file path.
        fps: Target output frame rate. If >24, uses motion interpolation.
        speed: Playback speed multiplier (0.5=slow-mo, 1.0=normal, 2.0=fast).
        audio_sample_rate: Audio sample rate in Hz.
    """
    import subprocess
    import tempfile
    import wave
    from PIL import Image

    NATIVE_FPS = 24  # Model generates motion at 24fps

    # Create temp directory for frames and audio
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as images with progress
        print("  Writing frames...")
        if HAS_TQDM:
            iterator = tqdm(enumerate(frames), desc="  Saving frames", total=len(frames), ncols=80)
        else:
            iterator = enumerate(frames)

        for i, frame in iterator:
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        # Save audio as WAV file
        audio_path = os.path.join(tmpdir, "audio.wav")
        print("\n  Writing audio...")

        # audio_waveform shape: (B, 2, samples) - stereo
        audio_np = np.array(audio_waveform[0])  # (2, samples)

        # Convert from float [-1, 1] to int16
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

        # Interleave stereo channels: (2, samples) -> (samples, 2) -> flat
        audio_interleaved = audio_int16.T  # (samples, 2)
        audio_flat = audio_interleaved.flatten()

        # Write WAV file
        with wave.open(audio_path, 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(audio_sample_rate)
            wav_file.writeframes(audio_flat.tobytes())

        print(f"    Audio: {len(audio_flat) // 2} samples, {len(audio_flat) // 2 / audio_sample_rate:.2f}s")

        # Build video filter chain
        video_filters = []

        # Speed adjustment for video (applied first, before interpolation)
        if speed != 1.0:
            pts_multiplier = 1.0 / speed
            video_filters.append(f"setpts={pts_multiplier}*PTS")

        # Frame interpolation if target fps > native
        if fps > NATIVE_FPS:
            video_filters.append(f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")

        # Build audio filter chain for speed adjustment
        # atempo filter range is 0.5-2.0, so chain multiple for extreme speeds
        audio_filters = []
        if speed != 1.0:
            remaining_speed = speed
            while remaining_speed > 2.0:
                audio_filters.append("atempo=2.0")
                remaining_speed /= 2.0
            while remaining_speed < 0.5:
                audio_filters.append("atempo=0.5")
                remaining_speed /= 0.5
            if remaining_speed != 1.0:
                audio_filters.append(f"atempo={remaining_speed}")

        # Build ffmpeg command
        print("\n  Encoding video with audio...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(NATIVE_FPS),  # Input is always at native 24fps
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-i", audio_path,
        ]

        # Add video filter chain if needed
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])
            if fps > NATIVE_FPS:
                print(f"  Interpolating {NATIVE_FPS}fps → {fps}fps (speed: {speed}x)")
            elif speed != 1.0:
                print(f"  Applying speed: {speed}x")

        # Add audio filter chain if needed
        if audio_filters:
            cmd.extend(["-af", ",".join(audio_filters)])

        cmd.extend([
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-shortest",  # Use shortest duration (video or audio)
            "-loglevel", "error",
            output_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Generate video with LTX-2 MLX")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=704, help="Video width")
    parser.add_argument("--frames", type=int, default=97, help="Number of frames")
    parser.add_argument("--steps", type=int, default=8, help="Denoising steps (8 for distilled, 15+ for two-stage)")
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale (default 5.0 for better semantic quality)")
    parser.add_argument("--guidance-rescale", type=float, default=0.7, help="Guidance rescale factor (0.0=off, 0.7=default, 1.0=full)")
    parser.add_argument("--steps-stage1", type=int, default=15, help="Stage 1 steps for two-stage pipeline")
    parser.add_argument("--steps-stage2", type=int, default=3, help="Stage 2 refinement steps for two-stage pipeline")
    parser.add_argument("--cfg-stage1", type=float, default=None, help="Stage 1 CFG (defaults to --cfg value)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate (default: 24). If >24, uses frame interpolation.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (0.5=slow-mo, 1.0=normal, 2.0=fast)")
    parser.add_argument("--output", type=str, default="outputs/output.mp4", help="Output path")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to weights"
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Use placeholder inference (skip model loading)"
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip VAE decoding (output latent visualization instead)"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Path to pre-computed text embedding (.npz)"
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
        help="Path to Gemma 3 weights directory"
    )
    parser.add_argument(
        "--no-gemma",
        action="store_true",
        help="Use dummy embeddings instead of real Gemma encoding (for testing)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 computation (default, ~50%% memory reduction)"
    )
    parser.add_argument(
        "--fp32", "--no-fp16",
        action="store_true",
        dest="fp32",
        help="Use FP32 computation instead of FP16 (higher memory usage)"
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Load FP8-quantized weights (auto-selects distilled-fp8 or dev-fp8)"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["distilled", "dev"],
        default="distilled",
        help="Model variant: 'distilled' (fast, 3-7 steps) or 'dev' (quality, 25-50 steps)"
    )
    parser.add_argument(
        "--distilled-lora",
        type=str,
        default=None,
        help="Path to distilled LoRA weights (required for high-quality two-stage generation)"
    )
    parser.add_argument(
        "--distilled-lora-scale",
        type=float,
        default=1.0,
        help="Scale for distilled LoRA (default 1.0)"
    )
    parser.add_argument(
        "--upscale-spatial",
        action="store_true",
        help="Apply 2x spatial upscaling to output (256->512, etc.)"
    )
    parser.add_argument(
        "--spatial-upscaler-weights",
        type=str,
        default="weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        help="Path to spatial upscaler weights"
    )
    parser.add_argument(
        "--upscale-temporal",
        action="store_true",
        help="Apply 2x temporal upscaling to output (17->33 frames, etc.)"
    )
    parser.add_argument(
        "--temporal-upscaler-weights",
        type=str,
        default="weights/ltx-2/ltx-2-temporal-upscaler-x2-1.0.safetensors",
        help="Path to temporal upscaler weights"
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate synchronized audio with video (requires AudioVideo model weights)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable aggressive memory optimization (slower but uses ~30%% less VRAM)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Experimental: Skip intermediate evaluations during denoising. "
             "May increase memory usage. Not recommended for 19B models - "
             "the GPU is already fully utilized, so this typically doesn't help."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to conditioning image for image-to-video generation"
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=0.95,
        help="Conditioning strength for --image (0.0-1.0, default 0.95)"
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to LoRA weights (.safetensors)"
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="LoRA strength (-2.0 to 2.0, default 1.0)"
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=0.0,
        help="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. (EXPERIMENTAL)"
    )
    parser.add_argument(
        "--stg-mode",
        type=str,
        choices=["video", "audio", "both"],
        default="video",
        help="STG perturbation mode: video, audio, or both (EXPERIMENTAL)"
    )
    # APG (Adaptive Projected Guidance) arguments
    parser.add_argument(
        "--apg-scale",
        type=float,
        default=1.0,
        help="APG (Adaptive Projected Guidance) scale. 1.0 disables APG, use values like 3.0-7.0"
    )
    parser.add_argument(
        "--apg-eta",
        type=float,
        default=1.0,
        help="APG parallel component weight (default 1.0)"
    )
    parser.add_argument(
        "--apg-norm-threshold",
        type=float,
        default=0.0,
        help="APG norm threshold for guidance clipping (0 = disabled)"
    )
    parser.add_argument(
        "--apg-momentum",
        type=float,
        default=0.0,
        help="APG momentum for stateful guidance (0 = disabled, try 0.5-0.9)"
    )
    # GE (Gradient Estimation) denoising argument
    parser.add_argument(
        "--ge-gamma",
        type=float,
        default=0.0,
        help="GE (Gradient Estimation) gamma. 0.0 disables GE, try 2.0 to reduce steps"
    )
    # IC-LoRA control signal arguments
    parser.add_argument(
        "--control-video",
        type=str,
        default=None,
        help="Path to control video for IC-LoRA conditioning (depth, pose, canny)"
    )
    parser.add_argument(
        "--control-type",
        type=str,
        choices=["canny", "raw"],
        default="raw",
        help="Control signal type: 'canny' applies edge detection, 'raw' uses video as-is"
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=100,
        help="Canny edge detection low threshold (0-255)"
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=200,
        help="Canny edge detection high threshold (0-255)"
    )
    parser.add_argument(
        "--control-strength",
        type=float,
        default=0.95,
        help="Control signal strength (0.0-1.0, default 0.95)"
    )
    parser.add_argument(
        "--save-control",
        action="store_true",
        help="Save the preprocessed control signal video for debugging"
    )
    parser.add_argument(
        "--tiled-vae",
        action="store_true",
        help="Use tiled VAE decoding for lower memory usage"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["text-to-video", "distilled", "one-stage", "two-stage", "ic-lora", "keyframe-interpolation"],
        default="text-to-video",
        help="Pipeline type (default: text-to-video)"
    )
    # Keyframe interpolation arguments
    parser.add_argument(
        "--keyframe",
        type=str,
        action="append",
        default=None,
        help="Keyframe image in format 'path:frame_index' or 'path:frame_index:strength'. Can be specified multiple times."
    )
    # IC-LoRA arguments
    parser.add_argument(
        "--ic-lora-weights",
        type=str,
        default=None,
        help="Path to IC-LoRA weights for video-to-video generation"
    )
    parser.add_argument(
        "--early-layers-only",
        action="store_true",
        help="[EXPERIMENTAL] Use only Layer 0 (input embeddings) from Gemma. "
             "Preserves text differentiation (~0.4 corr vs ~0.999+ with full pipeline)."
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Use Gemma to expand short prompts into detailed descriptions before encoding. "
             "This matches the official LTX-2 pipeline behavior and improves text differentiation."
    )
    parser.add_argument(
        "--cross-attn-scale",
        type=float,
        default=1.0,
        help="Scale factor for cross-attention in late transformer layers (40-47). "
             "Values 5-10 improve text conditioning for semantic content generation. "
             "Default 1.0 preserves original behavior."
    )

    args = parser.parse_args()

    # Auto-select weights based on model variant
    if args.model_variant == "dev":
        # Switch to dev weights
        args.weights = args.weights.replace("distilled", "dev")
        # Adjust default steps for dev model if not specified
        if args.steps == 7:  # default distilled value
            args.steps = 30
            print(f"Using dev model default: {args.steps} steps")

    # Auto-select FP8 weights if --fp8 flag is set
    if args.fp8:
        if ".safetensors" in args.weights and "-fp8" not in args.weights:
            args.weights = args.weights.replace(".safetensors", "-fp8.safetensors")
            print(f"Using FP8 weights: {args.weights}")

    generate_video(
        distilled_lora=args.distilled_lora,
        distilled_lora_scale=args.distilled_lora_scale,

        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_steps=args.steps,
        cfg_scale=args.cfg,
        guidance_rescale=getattr(args, 'guidance_rescale', 0.7),
        seed=args.seed,
        weights_path=args.weights,
        output_path=args.output,
        use_placeholder=args.placeholder,
        skip_vae=args.skip_vae,
        embedding_path=args.embedding,
        gemma_path=args.gemma_path,
        use_gemma=not args.no_gemma,
        use_fp16=not args.fp32,  # FP16 is default, --fp32 overrides
        use_fp8=args.fp8,
        model_variant=args.model_variant,
        upscale_spatial=args.upscale_spatial,
        spatial_upscaler_weights=args.spatial_upscaler_weights,
        upscale_temporal=args.upscale_temporal,
        temporal_upscaler_weights=args.temporal_upscaler_weights,
        generate_audio=args.generate_audio,
        low_memory=args.low_memory,
        fast_mode=args.fast_mode,
        # New parameters
        image_path=args.image,
        image_strength=args.image_strength,
        lora_path=args.lora,
        lora_strength=args.lora_strength,
        tiled_vae=args.tiled_vae,
        pipeline_type=args.pipeline,
        early_layers_only=args.early_layers_only,
        enhance_prompt_flag=args.enhance_prompt,
        cross_attn_scale=args.cross_attn_scale,
        # Two-stage pipeline parameters
        steps_stage1=args.steps_stage1,
        steps_stage2=args.steps_stage2,
        cfg_stage1=args.cfg_stage1,
        # STG parameters
        stg_scale=args.stg_scale,
        stg_mode=args.stg_mode,
        # APG parameters
        apg_scale=args.apg_scale,
        apg_eta=args.apg_eta,
        apg_norm_threshold=args.apg_norm_threshold,
        apg_momentum=args.apg_momentum,
        # IC-LoRA control parameters
        control_video=args.control_video,
        control_type=args.control_type,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        control_strength=args.control_strength,
        save_control=args.save_control,
        # GE (Gradient Estimation) parameter
        ge_gamma=args.ge_gamma,
        # Output FPS and speed
        output_fps=args.fps,
        output_speed=args.speed,
        # IC-LoRA and Keyframe Interpolation
        keyframes=args.keyframe,
        ic_lora_weights=args.ic_lora_weights,
    )


if __name__ == "__main__":
    main()
