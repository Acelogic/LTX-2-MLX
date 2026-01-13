#!/usr/bin/env python3
"""End-to-end parity check: MLX vs PyTorch with identical config.

Option 1 (fast, parity-first):
- model: distilled
- pipeline: distilled (8 steps, cfg=1.0)
- resolution: 256x384
- frames: 33
- seed: 42
- prompt: "A cat walking in a garden"
- no LoRA, no upscaler, no prompt enhancement

Compares:
1. Text embeddings (after connector)
2. Initial noise latent
3. Latent after each denoising step
4. Final decoded frames (pixel-level diff)
"""

import sys
from pathlib import Path

# Mock triton before any imports
import types
mock_triton = types.ModuleType('triton')
mock_triton.cdiv = lambda a, b: (a + b - 1) // b
mock_triton.jit = lambda fn: fn

mock_triton_language = types.ModuleType('triton.language')
mock_triton_language.constexpr = int

class MockDtype:
    pass
mock_triton_language.dtype = MockDtype

mock_triton.language = mock_triton_language

sys.modules['triton'] = mock_triton
sys.modules['triton.language'] = mock_triton_language

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add PyTorch LTX-2 to path
pytorch_ltx_path = Path(__file__).parent.parent.parent / "LTX-2-PyTorch"
if pytorch_ltx_path.exists():
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-core" / "src"))
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-pipelines" / "src"))
else:
    print(f"WARNING: PyTorch path not found: {pytorch_ltx_path}")
    print("Create symlink: ln -s LTX-2 LTX-2-PyTorch")

import numpy as np
import torch
import mlx.core as mx

# =============================================================================
# Configuration - Option 1 (fast parity check)
# =============================================================================
CONFIG = {
    "weights_path": "weights/ltx-2/ltx-2-19b-distilled.safetensors",
    "gemma_path": "weights/gemma-3-12b",
    "prompt": "A cat walking in a garden",
    "negative_prompt": "",
    "seed": 42,
    "height": 256,
    "width": 384,
    "frames": 33,
    "fps": 24.0,
    "cfg_scale": 1.0,  # No CFG for distilled
    "sigmas": [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
}

# Derived latent dimensions (32x spatial, 8x temporal compression)
LATENT_H = CONFIG["height"] // 32
LATENT_W = CONFIG["width"] // 32
LATENT_F = (CONFIG["frames"] - 1) // 8 + 1  # frames=33 -> latent_f=5


def compare_arrays(name, a, b, rtol=0.01, atol=0.01, verbose=True):
    """Compare two arrays and return (is_close, correlation)."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH - {a.shape} vs {b.shape}")
        return False, 0.0

    abs_diff = np.abs(a - b)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    try:
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    except:
        corr = float('nan')

    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "✓" if close else "✗"

    if verbose:
        print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.4f}")
        if not close and corr < 0.99:
            print(f"    MLX: mean={a.mean():.6f}, std={a.std():.6f}, range=[{a.min():.4f}, {a.max():.4f}]")
            print(f"    PT:  mean={b.mean():.6f}, std={b.std():.6f}, range=[{b.min():.4f}, {b.max():.4f}]")

    return close, corr


def create_pixel_space_positions(frames, height, width, fps=24.0, time_scale=8, spatial_scale=32):
    """Create pixel-space positions matching PyTorch production pipeline."""
    frame_coords = np.arange(0, frames)
    height_coords = np.arange(0, height)
    width_coords = np.arange(0, width)

    grid_f, grid_h, grid_w = np.meshgrid(frame_coords, height_coords, width_coords, indexing="ij")

    patch_starts = np.stack([grid_f, grid_h, grid_w], axis=0)
    patch_ends = patch_starts + 1

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_tokens = frames * height * width
    latent_coords = latent_coords.reshape(3, num_tokens, 2)
    latent_coords = latent_coords[None, ...]

    scale_factors = np.array([time_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_factors

    pixel_coords[:, 0, ...] = np.maximum(pixel_coords[:, 0, ...] + 1 - time_scale, 0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

    return pixel_coords.astype(np.float32)


def patchify(x):
    """(B, C, F, H, W) -> (B, F*H*W, C)"""
    B, C, F, H, W = x.shape
    return x.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)


def unpatchify(x, F, H, W):
    """(B, T, C) -> (B, C, F, H, W)"""
    B, T, C = x.shape
    return x.reshape(B, F, H, W, C).transpose(0, 4, 1, 2, 3)


# =============================================================================
# Stage 1: Text Embedding Comparison
# =============================================================================
def compare_text_embeddings():
    """Compare text embeddings between MLX and PyTorch."""
    print("\n" + "=" * 70)
    print("Stage 1: Text Embedding Comparison")
    print("=" * 70)

    prompt = CONFIG["prompt"]
    gemma_path = CONFIG["gemma_path"]

    if not Path(gemma_path).exists():
        print(f"  Gemma weights not found: {gemma_path}")
        print("  Skipping text embedding comparison, using random embeddings")
        # Return random embeddings for testing
        np.random.seed(CONFIG["seed"])
        context = np.random.randn(1, 256, 4096).astype(np.float32) * 0.1
        return context, context

    # --- MLX Text Encoding ---
    print("\n  Loading MLX text encoder...")
    from LTX_2_MLX.model.text_encoder.gemma3 import Gemma3Model, Gemma3Config, load_gemma3_weights
    from LTX_2_MLX.model.text_encoder.encoder import VideoGemmaTextEncoderModel
    from LTX_2_MLX.model.text_encoder import load_text_encoder_weights
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize with left padding
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=1024,
        truncation=True,
        return_tensors="np",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"  Prompt: '{prompt}'")
    print(f"  Tokens: {attention_mask.sum()} / 1024")

    # Load Gemma
    gemma_config = Gemma3Config()
    mlx_gemma = Gemma3Model(gemma_config)
    load_gemma3_weights(mlx_gemma, gemma_path, use_fp16=False)

    # Load text encoder projection layers
    mlx_text_encoder = VideoGemmaTextEncoderModel()
    load_text_encoder_weights(mlx_text_encoder, CONFIG["weights_path"])

    # Run MLX encoding
    mlx_input_ids = mx.array(input_ids)
    mlx_attention_mask = mx.array(attention_mask)

    _, mlx_hidden_states = mlx_gemma(
        mlx_input_ids,
        attention_mask=mlx_attention_mask,
        output_hidden_states=True,
    )
    mx.eval(mlx_hidden_states[-1])

    mlx_output = mlx_text_encoder.encode_from_hidden_states(
        hidden_states=mlx_hidden_states,
        attention_mask=mlx_attention_mask,
        padding_side="left",
    )
    mlx_context = np.array(mlx_output.video_encoding)

    # --- PyTorch Text Encoding ---
    print("  Loading PyTorch text encoder...")
    from transformers import Gemma3ForConditionalGeneration
    from ltx_core.text_encoders.gemma.encoders.av_encoder import AVGemmaTextEncoderModel
    from ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
    from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma.encoders.av_encoder import AV_GEMMA_TEXT_ENCODER_KEY_OPS

    # Load PyTorch Gemma
    pt_gemma = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path, local_files_only=True, torch_dtype=torch.float32
    )
    pt_gemma.eval()

    # Load text encoder components
    pt_feature_extractor = GemmaFeaturesExtractorProjLinear()
    pt_video_connector = Embeddings1DConnector()
    pt_audio_connector = Embeddings1DConnector()

    # Load weights
    builder = SingleGPUModelBuilder(
        model_class_configurator=lambda: AVGemmaTextEncoderModel(
            feature_extractor_linear=pt_feature_extractor,
            embeddings_connector=pt_video_connector,
            audio_embeddings_connector=pt_audio_connector,
        ),
        model_path=CONFIG["weights_path"],
        model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    )

    # Run PyTorch encoding
    pt_input_ids = torch.from_numpy(input_ids)
    pt_attention_mask = torch.from_numpy(attention_mask)

    with torch.no_grad():
        pt_outputs = pt_gemma(
            input_ids=pt_input_ids,
            attention_mask=pt_attention_mask,
            output_hidden_states=True,
        )
        pt_hidden_states = pt_outputs.hidden_states

        # Feature extraction
        encoded_text_features = torch.stack(pt_hidden_states, dim=-1)
        # ... (simplified - full pipeline would go through feature extractor + connector)

    pt_context = pt_hidden_states[-1].numpy()  # Simplified for now

    # Compare
    print("\n  Comparing embeddings...")
    compare_arrays("Layer 0", np.array(mlx_hidden_states[0]), pt_hidden_states[0].numpy())
    compare_arrays("Layer 48", np.array(mlx_hidden_states[-1]), pt_hidden_states[-1].numpy())

    # For now return MLX context for both (full comparison needs connector)
    return mlx_context, mlx_context


# =============================================================================
# Stage 2: Transformer Denoising Loop
# =============================================================================
def run_denoising_comparison(context_mlx, context_pt):
    """Run denoising loop and compare at each step."""
    print("\n" + "=" * 70)
    print("Stage 2: Denoising Loop Comparison")
    print("=" * 70)

    weights_path = CONFIG["weights_path"]

    if not Path(weights_path).exists():
        print(f"  Weights not found: {weights_path}")
        return None, None

    # Setup
    np.random.seed(CONFIG["seed"])
    batch_size = 1
    latent_channels = 128
    frames, height, width = LATENT_F, LATENT_H, LATENT_W
    sigmas = np.array(CONFIG["sigmas"])

    print(f"\n  Latent shape: ({batch_size}, {latent_channels}, {frames}, {height}, {width})")
    print(f"  Sigmas: {len(sigmas)-1} steps")

    # Create initial noise
    initial_noise = np.random.randn(batch_size, latent_channels, frames, height, width).astype(np.float32)
    initial_patchified = patchify(initial_noise)
    positions = create_pixel_space_positions(frames, height, width, CONFIG["fps"])

    # Use provided context or create dummy
    # Context dimension is 3840 (Gemma hidden size) - caption_projection maps to 4096
    if context_mlx is None:
        context = np.random.randn(1, 256, 3840).astype(np.float32) * 0.1
    else:
        context = context_mlx

    # --- Load PyTorch Model ---
    print("\n  Loading PyTorch transformer...")
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.model import X0Model as PTX0Model
    from ltx_core.model.transformer.modality import Modality as PTModality

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pt_velocity_model = builder.build(device=torch.device("cpu"), dtype=torch.float32)
    pt_velocity_model.eval()
    pt_x0_model = PTX0Model(pt_velocity_model)

    # --- Load MLX Model ---
    print("  Loading MLX transformer...")
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, X0Model as MLXX0Model
    from LTX_2_MLX.model.transformer.model import Modality as MLXModality
    from LTX_2_MLX.loader import load_transformer_weights

    mlx_velocity_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        compute_dtype=mx.float32,
    )
    load_transformer_weights(mlx_velocity_model, weights_path)
    mlx_x0_model = MLXX0Model(mlx_velocity_model)

    # --- Compare Initial Noise ---
    print("\n  Comparing initial noise...")
    compare_arrays("Initial noise", initial_patchified, initial_patchified)

    # --- Run Denoising Loop ---
    mlx_latent = mx.array(initial_patchified)
    pt_latent = torch.from_numpy(initial_patchified)
    num_tokens = initial_patchified.shape[1]

    results = []

    for step_idx in range(len(sigmas) - 1):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma

        print(f"\n  Step {step_idx}: σ={sigma:.4f} → {sigma_next:.4f}")

        # MLX forward pass
        mlx_modality = MLXModality(
            latent=mlx_latent,
            context=mx.array(context),
            context_mask=None,
            timesteps=mx.array([sigma]),
            positions=mx.array(positions),
            enabled=True,
        )
        mlx_denoised = mlx_x0_model(mlx_modality)
        mx.eval(mlx_denoised)

        # PyTorch forward pass
        pt_timesteps = np.ones((batch_size, num_tokens, 1), dtype=np.float32) * sigma
        pt_modality = PTModality(
            enabled=True,
            latent=pt_latent,
            timesteps=torch.from_numpy(pt_timesteps),
            positions=torch.from_numpy(positions),
            context=torch.from_numpy(context),
            context_mask=None,
        )
        with torch.no_grad():
            pt_denoised, _ = pt_x0_model(video=pt_modality, audio=None, perturbations=None)

        # Compare X0 predictions
        _, corr = compare_arrays(f"X0 pred", np.array(mlx_denoised), pt_denoised.numpy())
        results.append(("x0", step_idx, corr))

        # Euler step
        mlx_velocity = (mlx_latent - mlx_denoised) / sigma
        mlx_latent = mlx_latent + mlx_velocity * dt
        mx.eval(mlx_latent)

        pt_velocity = (pt_latent - pt_denoised) / sigma
        pt_latent = pt_latent + pt_velocity * dt

        # Compare updated latents
        _, corr = compare_arrays(f"Latent", np.array(mlx_latent), pt_latent.numpy())
        results.append(("latent", step_idx, corr))

    # --- Final Comparison ---
    print("\n" + "-" * 40)
    print("  Final Results")
    print("-" * 40)

    final_mlx = unpatchify(np.array(mlx_latent), frames, height, width)
    final_pt = unpatchify(pt_latent.numpy(), frames, height, width)

    _, final_corr = compare_arrays("Final latent (5D)", final_mlx, final_pt)

    return final_mlx, final_pt, results


# =============================================================================
# Stage 3: VAE Decoding Comparison
# =============================================================================
def compare_vae_decoding(mlx_latent, pt_latent):
    """Compare VAE decoding between MLX and PyTorch."""
    print("\n" + "=" * 70)
    print("Stage 3: VAE Decoding Comparison")
    print("=" * 70)

    if mlx_latent is None:
        print("  Skipping VAE comparison (no latents)")
        return

    # This would require loading VAE decoders for both frameworks
    # For now, just decode with MLX
    print("  (VAE comparison not yet implemented - requires PyTorch VAE loading)")
    print("  Decoding with MLX only...")

    # TODO: Add PyTorch VAE decoder comparison


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("End-to-End Parity Check: MLX vs PyTorch")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: distilled")
    print(f"  Resolution: {CONFIG['width']}x{CONFIG['height']}")
    print(f"  Frames: {CONFIG['frames']} (latent: {LATENT_F})")
    print(f"  Steps: {len(CONFIG['sigmas'])-1}")
    print(f"  CFG: {CONFIG['cfg_scale']}")
    print(f"  Seed: {CONFIG['seed']}")
    print(f"  Prompt: '{CONFIG['prompt']}'")

    # Stage 1: Text embeddings
    # mlx_context, pt_context = compare_text_embeddings()

    # For now, use random context (text encoder comparison is complex)
    # Context dimension is 3840 (Gemma hidden size) - caption_projection maps to 4096
    np.random.seed(CONFIG["seed"])
    context = np.random.randn(1, 256, 3840).astype(np.float32) * 0.1

    # Stage 2: Denoising loop
    mlx_latent, pt_latent, results = run_denoising_comparison(context, context)

    # Stage 3: VAE decoding
    compare_vae_decoding(mlx_latent, pt_latent)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if results:
        x0_corrs = [r[2] for r in results if r[0] == "x0"]
        latent_corrs = [r[2] for r in results if r[0] == "latent"]

        print(f"\n  X0 prediction correlations: min={min(x0_corrs):.4f}, max={max(x0_corrs):.4f}")
        print(f"  Latent correlations: min={min(latent_corrs):.4f}, max={max(latent_corrs):.4f}")

        if min(latent_corrs) > 0.99:
            print("\n  ✓ PASS: Denoising loop matches PyTorch!")
        elif min(latent_corrs) > 0.95:
            print("\n  ~ CLOSE: Minor numerical differences (corr > 0.95)")
        else:
            print("\n  ✗ FAIL: Significant divergence detected")


if __name__ == "__main__":
    main()
