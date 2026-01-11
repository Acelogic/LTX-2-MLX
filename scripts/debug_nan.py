#!/usr/bin/env python3
"""Debug script to identify NaN and zero values in the LTX-2 MLX pipeline."""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_array(name: str, arr: mx.array, verbose: bool = True) -> dict:
    """Check an array for NaN, inf, and zero values."""
    mx.eval(arr)
    np_arr = np.array(arr)

    stats = {
        "name": name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "has_nan": bool(np.isnan(np_arr).any()),
        "has_inf": bool(np.isinf(np_arr).any()),
        "all_zero": bool(np.allclose(np_arr, 0)),
        "min": float(np.nanmin(np_arr)),
        "max": float(np.nanmax(np_arr)),
        "mean": float(np.nanmean(np_arr)),
        "std": float(np.nanstd(np_arr)),
        "nan_count": int(np.isnan(np_arr).sum()),
        "inf_count": int(np.isinf(np_arr).sum()),
        "zero_count": int((np_arr == 0).sum()),
        "total_elements": int(np_arr.size),
    }

    if verbose:
        status = "OK"
        if stats["has_nan"]:
            status = "NaN DETECTED!"
        elif stats["has_inf"]:
            status = "INF DETECTED!"
        elif stats["all_zero"]:
            status = "ALL ZEROS!"

        print(f"\n{name}:")
        print(f"  Shape: {stats['shape']}, dtype: {stats['dtype']}")
        print(f"  Status: {status}")
        print(f"  Range: [{stats['min']:.6g}, {stats['max']:.6g}]")
        print(f"  Mean: {stats['mean']:.6g}, Std: {stats['std']:.6g}")
        if stats["has_nan"]:
            print(f"  NaN count: {stats['nan_count']} / {stats['total_elements']}")
        if stats["has_inf"]:
            print(f"  Inf count: {stats['inf_count']} / {stats['total_elements']}")
        if stats["zero_count"] > 0:
            pct = 100 * stats["zero_count"] / stats["total_elements"]
            print(f"  Zero count: {stats['zero_count']} ({pct:.1f}%)")

    return stats


def debug_vae_weights(weights_path: str):
    """Check VAE decoder weights for issues."""
    from safetensors import safe_open
    import torch

    print("\n" + "=" * 60)
    print("CHECKING VAE DECODER WEIGHTS")
    print("=" * 60)

    vae_keys = []
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("vae."):
                vae_keys.append(key)

    print(f"\nFound {len(vae_keys)} VAE keys")

    # Check critical VAE weights
    critical_keys = [
        "vae.per_channel_statistics.mean-of-means",
        "vae.per_channel_statistics.std-of-means",
        "vae.decoder.conv_in.conv.weight",
        "vae.decoder.conv_in.conv.bias",
        "vae.decoder.conv_out.conv.weight",
        "vae.decoder.conv_out.conv.bias",
    ]

    with safe_open(weights_path, framework="pt") as f:
        for key in critical_keys:
            if key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                arr = mx.array(tensor.numpy())
                check_array(f"VAE: {key}", arr)
            else:
                print(f"\n{key}: NOT FOUND!")


def debug_transformer_weights(weights_path: str):
    """Check transformer weights for issues."""
    from safetensors import safe_open
    import torch

    print("\n" + "=" * 60)
    print("CHECKING TRANSFORMER WEIGHTS")
    print("=" * 60)

    # Check critical transformer weights
    critical_keys = [
        "model.diffusion_model.patchify_proj.weight",
        "model.diffusion_model.patchify_proj.bias",
        "model.diffusion_model.proj_out.weight",
        "model.diffusion_model.proj_out.bias",
        "model.diffusion_model.scale_shift_table",
        "model.diffusion_model.adaln_single.linear.weight",
        "model.diffusion_model.caption_projection.linear_1.weight",
    ]

    with safe_open(weights_path, framework="pt") as f:
        for key in critical_keys:
            if key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                arr = mx.array(tensor.numpy())
                check_array(f"Transformer: {key}", arr)
            else:
                print(f"\n{key}: NOT FOUND!")

    # Check some transformer block weights
    print("\n--- Checking first transformer block ---")
    block_keys = [
        "model.diffusion_model.transformer_blocks.0.scale_shift_table",
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
        "model.diffusion_model.transformer_blocks.0.attn1.q_norm.weight",
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
    ]

    with safe_open(weights_path, framework="pt") as f:
        for key in block_keys:
            if key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                arr = mx.array(tensor.numpy())
                check_array(f"Block 0: {key.split('.')[-2]}.{key.split('.')[-1]}", arr)
            else:
                print(f"\n{key}: NOT FOUND!")


def euler_step_x0(sample: "mx.array", denoised: "mx.array", sigma: float, sigma_next: float) -> "mx.array":
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
    from LTX_2_MLX.core_utils import to_velocity

    # Convert denoised to velocity (matches PyTorch reference)
    velocity = to_velocity(sample, sigma, denoised)

    # Euler step
    dt = sigma_next - sigma
    return sample.astype(mx.float32) + velocity.astype(mx.float32) * dt


def encode_with_gemma(
    prompt: str,
    gemma_path: str = "weights/gemma-3-12b",
    ltx_weights_path: str = "weights/ltx-2/ltx-2-19b-distilled.safetensors",
    max_length: int = 256,
) -> tuple:
    """
    Encode a text prompt using real Gemma 3 + LTX-2 text encoder pipeline.
    """
    from LTX_2_MLX.model.text_encoder.gemma3 import Gemma3Config, Gemma3Model, load_gemma3_weights
    from LTX_2_MLX.model.text_encoder.encoder import create_text_encoder, load_text_encoder_weights

    # Load tokenizer
    print("  Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    except ImportError:
        print("  ERROR: transformers package not installed")
        return None, None

    # CRITICAL: Use right padding
    tokenizer.padding_side = "right"

    # Load Gemma 3
    print("  Loading Gemma 3 model (~25GB)...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, gemma_path)

    # Load text encoder
    print("  Loading text encoder projection...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, ltx_weights_path)

    # Create chat prompt format
    T2V_SYSTEM_PROMPT = """You are an assistant that generates detailed visual descriptions of video scenes.
When given a prompt, you will describe the scene in precise, vivid detail."""
    chat_prompt = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    # Tokenize
    print(f"  Tokenizing prompt: '{prompt[:50]}...'")
    encoded = tokenizer(
        chat_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )
    input_ids = mx.array(encoded["input_ids"])
    attention_mask = mx.array(encoded["attention_mask"])

    # Get Gemma hidden states
    print("  Running Gemma 3 forward pass...")
    last_hidden, all_hidden_states = gemma(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mx.eval(last_hidden)

    if all_hidden_states is None:
        print("  ERROR: Gemma did not return hidden states")
        return None, None

    # Run through text encoder pipeline (feature extraction + connector only)
    # Note: We skip caption_projection because the transformer has its own
    print("  Running text encoder pipeline (feature extractor + connector)...")

    # Feature extraction
    encoded = text_encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=attention_mask,
        padding_side="right",
    )

    # Convert mask to additive format
    large_value = 1e9
    connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
    connector_mask = connector_mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])

    # Process through connector
    encoded, output_mask = text_encoder.embeddings_connector(encoded, connector_mask)

    # Convert mask back to binary
    binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)

    # Apply mask to zero out padded positions
    text_encoding = encoded * binary_mask[:, :, None]

    mx.eval(text_encoding)
    mx.eval(binary_mask)

    print(f"  Output embedding shape: {text_encoding.shape}")  # Should be [B, T, 3840]

    # Cleanup Gemma
    print("  Clearing Gemma from memory...")
    del gemma
    del text_encoder
    del all_hidden_states
    del last_hidden
    del attention_mask
    del output_mask
    del connector_mask
    import gc
    gc.collect()

    return text_encoding, binary_mask


def create_null_encoding(shape, embed_dim=3840):
    """Create null/zero encoding for CFG unconditional pass."""
    batch_size, max_tokens = shape[0], shape[1]
    null_encoding = mx.zeros((batch_size, max_tokens, embed_dim))
    null_mask = mx.ones((batch_size, max_tokens), dtype=mx.int32)
    return null_encoding, null_mask


def debug_denoising_loop(
    weights_path: str,
    embedding_path: str = None,
    num_steps: int = 7,
    use_gemma: bool = False,
    gemma_path: str = "weights/gemma-3-12b",
    prompt: str = "A cat walking in a garden",
    use_cfg: bool = False,
    cfg_scale: float = 3.0,
    check_velocity_ref: bool = False,
):
    """
    Debug the full denoising loop to check if it converges.

    This runs the complete denoising process and tracks statistics at each step.

    Args:
        use_cfg: Enable Classifier-Free Guidance (run cond + uncond, combine)
        cfg_scale: CFG scale factor (default 3.0)
        check_velocity_ref: Check velocity statistics against expected ranges
    """
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, create_position_grid
    from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier, LTX2Scheduler
    from LTX_2_MLX.types import VideoLatentShape
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.model.video_vae.simple_decoder import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent
    from LTX_2_MLX.core_utils import to_velocity, to_denoised

    print("\n" + "=" * 60)
    print("DEBUGGING FULL DENOISING LOOP")
    print("=" * 60)

    # Small test dimensions
    height, width, num_frames = 256, 384, 9
    latent_height = height // 32
    latent_width = width // 32
    latent_frames = (num_frames - 1) // 8 + 1

    print(f"\nTest dimensions: {width}x{height}, {num_frames} frames")
    print(f"Latent shape: {latent_frames}x{latent_height}x{latent_width}")
    print(f"Denoising steps: {num_steps}")

    # Initialize
    mx.random.seed(42)
    latent = mx.random.normal(shape=(1, 128, latent_frames, latent_height, latent_width))

    # Text encoding
    if embedding_path:
        data = np.load(embedding_path)
        text_encoding = mx.array(data["embedding"])
        text_mask = mx.array(data["attention_mask"])
        print(f"Loaded embedding from {embedding_path}")
    elif use_gemma:
        print(f"\n--- Encoding with Gemma 3 ---")
        print(f"Prompt: '{prompt}'")
        text_encoding, text_mask = encode_with_gemma(
            prompt=prompt,
            gemma_path=gemma_path,
            ltx_weights_path=weights_path,
        )
        if text_encoding is None:
            print("ERROR: Failed to encode with Gemma, falling back to dummy")
            mx.random.seed(12345)
            text_encoding = mx.random.normal(shape=(1, 256, 3840)) * 0.1
            text_mask = mx.ones((1, 256))
        else:
            print(f"  Encoded! Shape: {text_encoding.shape}")
            check_array("Gemma text encoding", text_encoding)
    else:
        mx.random.seed(12345)
        text_encoding = mx.random.normal(shape=(1, 256, 3840)) * 0.1
        text_mask = mx.ones((1, 256))
        print("Using dummy text encoding")

    # Load model
    print("\nLoading transformer...")
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
    )
    load_transformer_weights(model, weights_path)

    # Load VAE decoder
    print("Loading VAE decoder...")
    vae_decoder = SimpleVideoDecoder()
    load_vae_decoder_weights(vae_decoder, weights_path)

    # Check VAE statistics
    print("\n--- VAE Statistics Check ---")
    check_array("mean_of_means", vae_decoder.mean_of_means)
    check_array("std_of_means", vae_decoder.std_of_means)

    # Setup
    patchifier = VideoLatentPatchifier(patch_size=1)

    # Use dynamic scheduler with resolution-dependent shifting
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=num_steps, latent=latent)
    print(f"\nUsing LTX2Scheduler (dynamic sigma schedule based on latent size)")
    print(f"  Token count: {latent.shape[2] * latent.shape[3] * latent.shape[4]}")
    # Note: DISTILLED_SIGMA_VALUES would be: mx.array(DISTILLED_SIGMA_VALUES[:num_steps + 1])

    output_shape = VideoLatentShape(
        batch=1,
        channels=128,
        frames=latent_frames,
        height=latent_height,
        width=latent_width,
    )

    # Create position grid
    grid = create_position_grid(1, latent_frames, latent_height, latent_width)
    grid_start = grid[..., None]
    grid_end = grid_start + 1
    positions = mx.concatenate([grid_start, grid_end], axis=-1)

    # Create null encoding for CFG if needed
    null_encoding, null_mask = None, None
    if use_cfg:
        null_encoding, null_mask = create_null_encoding(text_encoding.shape[:2])
        print(f"\nCFG enabled with scale={cfg_scale}")
        print(f"  Null encoding shape: {null_encoding.shape}")

    # Expected velocity ranges from PyTorch reference (approximate)
    # These are rough ranges - actual values depend on prompt and noise
    velocity_ref_ranges = {
        1: {"std": (0.3, 0.8), "abs_max": (2.0, 5.0)},  # Step 1: sigma 1.0->0.7
        2: {"std": (0.3, 0.9), "abs_max": (2.0, 6.0)},  # Step 2: sigma 0.7->0.4
        3: {"std": (0.3, 1.0), "abs_max": (2.0, 6.0)},  # Step 3: sigma 0.4->0.2
        4: {"std": (0.3, 1.2), "abs_max": (2.0, 7.0)},  # Step 4: sigma 0.2->0.1
        5: {"std": (0.3, 1.5), "abs_max": (2.0, 8.0)},  # Step 5: sigma 0.1->0.03
        6: {"std": (0.3, 2.0), "abs_max": (2.0, 10.0)}, # Step 6: sigma 0.03->0.003
        7: {"std": (0.3, 2.5), "abs_max": (2.0, 12.0)}, # Step 7: sigma 0.003->0
    }

    print("\n--- Denoising Loop ---")
    print(f"Initial latent: mean={float(latent.mean()):.4f}, std={float(latent.std()):.4f}")
    print(f"Sigmas: {[f'{float(s):.3f}' for s in sigmas]}")
    if use_cfg:
        print(f"Mode: CFG (scale={cfg_scale})")
    else:
        print(f"Mode: No CFG (conditional only)")
    print()

    for i in range(num_steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        # Patchify
        latent_patchified = patchifier.patchify(latent)

        if use_cfg:
            # CFG with X0 pattern (matching PyTorch reference):
            # 1. Get velocity from model for both cond and uncond
            # 2. Convert velocity → denoised (X0)
            # 3. Apply CFG on denoised samples
            # 4. Euler step with denoised

            # Conditional pass
            modality_cond = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )
            velocity_cond_patchified = model(modality_cond)
            mx.eval(velocity_cond_patchified)
            velocity_cond = patchifier.unpatchify(velocity_cond_patchified, output_shape=output_shape)

            # Unconditional pass
            modality_uncond = Modality(
                latent=latent_patchified,
                context=null_encoding,
                context_mask=null_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )
            velocity_uncond_patchified = model(modality_uncond)
            mx.eval(velocity_uncond_patchified)
            velocity_uncond = patchifier.unpatchify(velocity_uncond_patchified, output_shape=output_shape)

            # Convert velocity → denoised (X0) for both
            denoised_cond = to_denoised(latent, velocity_cond, sigma)
            denoised_uncond = to_denoised(latent, velocity_uncond, sigma)
            mx.eval(denoised_cond)
            mx.eval(denoised_uncond)

            # Apply CFG on denoised samples (NOT velocity!)
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
            mx.eval(denoised)

            # For logging, compute effective velocity
            velocity = to_velocity(latent, sigma, denoised)
            mx.eval(velocity)
        else:
            # No CFG: Single conditional pass with X0 pattern
            modality = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )
            velocity_patchified = model(modality)
            mx.eval(velocity_patchified)
            velocity = patchifier.unpatchify(velocity_patchified, output_shape=output_shape)
            mx.eval(velocity)

            # Convert velocity → denoised (X0)
            denoised = to_denoised(latent, velocity, sigma)
            mx.eval(denoised)

        # Euler step using X0 (denoised) prediction
        latent = euler_step_x0(latent, denoised, sigma, sigma_next)
        mx.eval(latent)

        # Check for issues
        has_nan = bool(mx.any(mx.isnan(latent)))
        has_inf = bool(mx.any(mx.isinf(latent)))
        status = "OK"
        if has_nan:
            status = "NaN DETECTED!"
        elif has_inf:
            status = "Inf DETECTED!"

        # Velocity reference check
        vel_status = ""
        if check_velocity_ref and (i + 1) in velocity_ref_ranges:
            ref = velocity_ref_ranges[i + 1]
            vel_std = float(velocity.std())
            vel_abs_max = max(abs(float(velocity.min())), abs(float(velocity.max())))

            std_ok = ref["std"][0] <= vel_std <= ref["std"][1]
            abs_ok = ref["abs_max"][0] <= vel_abs_max <= ref["abs_max"][1]

            if std_ok and abs_ok:
                vel_status = " [REF: OK]"
            else:
                issues = []
                if not std_ok:
                    issues.append(f"std {vel_std:.2f} not in {ref['std']}")
                if not abs_ok:
                    issues.append(f"abs_max {vel_abs_max:.2f} not in {ref['abs_max']}")
                vel_status = f" [REF: MISMATCH - {', '.join(issues)}]"

        print(f"Step {i+1}/{num_steps}: sigma={sigma:.3f}→{sigma_next:.3f}")
        print(f"  Velocity: mean={float(velocity.mean()):.4f}, std={float(velocity.std()):.4f}, range=[{float(velocity.min()):.4f}, {float(velocity.max()):.4f}]{vel_status}")
        print(f"  Latent:   mean={float(latent.mean()):.4f}, std={float(latent.std()):.4f}, range=[{float(latent.min()):.4f}, {float(latent.max()):.4f}] [{status}]")

        if has_nan or has_inf:
            print(f"\n  ERROR: {status} - stopping early")
            break

    # Final latent stats
    print("\n--- Final Latent ---")
    check_array("Denoised latent", latent)

    # Decode with VAE
    print("\n--- VAE Decode ---")
    try:
        video = decode_latent(latent, vae_decoder, timestep=0.05)
        mx.eval(video)

        # Check video stats
        print(f"Video output:")
        print(f"  Shape: {video.shape}")
        print(f"  Dtype: {video.dtype}")
        print(f"  Range: [{int(video.min())}, {int(video.max())}]")
        print(f"  Mean: {float(video.astype(mx.float32).mean()):.1f}")

        # Check if video is all black or all white
        if int(video.max()) < 10:
            print("  WARNING: Video is nearly all black!")
        elif int(video.min()) > 245:
            print("  WARNING: Video is nearly all white!")
        elif int(video.max()) - int(video.min()) < 20:
            print("  WARNING: Very low dynamic range!")
        else:
            print("  Status: Video has good dynamic range")

    except Exception as e:
        print(f"ERROR in VAE decode: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("DENOISING LOOP DEBUG COMPLETE")
    print("=" * 60)


def debug_inference_pipeline(weights_path: str, embedding_path: str = None):
    """Debug the full inference pipeline step by step."""
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, create_position_grid
    from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier
    from LTX_2_MLX.types import VideoLatentShape
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.model.video_vae.simple_decoder import SimpleVideoDecoder, load_vae_decoder_weights, decode_latent

    print("\n" + "=" * 60)
    print("DEBUGGING INFERENCE PIPELINE")
    print("=" * 60)

    # Small test dimensions
    height, width, num_frames = 128, 128, 9
    latent_height = height // 32
    latent_width = width // 32
    latent_frames = (num_frames - 1) // 8 + 1

    print(f"\nTest dimensions: {width}x{height}, {num_frames} frames")
    print(f"Latent shape: {latent_frames}x{latent_height}x{latent_width}")

    # 1. Check initial noise
    print("\n--- Step 1: Initial Noise ---")
    mx.random.seed(42)
    latent = mx.random.normal(shape=(1, 128, latent_frames, latent_height, latent_width))
    check_array("Initial latent noise", latent)

    # 2. Check text encoding
    print("\n--- Step 2: Text Encoding ---")
    if embedding_path:
        data = np.load(embedding_path)
        text_encoding = mx.array(data["embedding"])
        text_mask = mx.array(data["attention_mask"])
        print(f"Loaded embedding from {embedding_path}")
    else:
        mx.random.seed(12345)
        text_encoding = mx.random.normal(shape=(1, 256, 3840)) * 0.1
        text_mask = mx.ones((1, 256))
        print("Using dummy text encoding")

    check_array("Text encoding", text_encoding)
    check_array("Text mask", text_mask)

    # 3. Load and check transformer
    print("\n--- Step 3: Loading Transformer ---")
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
    )

    # Check weights before loading
    print("\nBefore weight loading:")
    check_array("patchify_proj.weight (before)", model.patchify_proj.weight)
    check_array("scale_shift_table (before)", model.scale_shift_table)

    # Load weights
    load_transformer_weights(model, weights_path)

    # Check weights after loading
    print("\nAfter weight loading:")
    check_array("patchify_proj.weight (after)", model.patchify_proj.weight)
    check_array("scale_shift_table (after)", model.scale_shift_table)
    check_array("proj_out.weight (after)", model.proj_out.weight)
    check_array("adaln_single.linear.weight", model.adaln_single.linear.weight)
    check_array("caption_projection.linear_1.weight", model.caption_projection.linear_1.weight)
    check_array("transformer_blocks[0].scale_shift_table", model.transformer_blocks[0].scale_shift_table)

    # 4. Test transformer forward pass
    print("\n--- Step 4: Transformer Forward Pass ---")
    patchifier = VideoLatentPatchifier(patch_size=1)

    # Patchify
    latent_patchified = patchifier.patchify(latent)
    check_array("Patchified latent", latent_patchified)

    # Create position grid
    grid = create_position_grid(1, latent_frames, latent_height, latent_width)
    grid_start = grid[..., None]
    grid_end = grid_start + 1
    positions = mx.concatenate([grid_start, grid_end], axis=-1)
    check_array("Position grid", positions)

    # Create modality input
    sigma = float(DISTILLED_SIGMA_VALUES[0])  # First sigma
    modality = Modality(
        latent=latent_patchified,
        context=text_encoding,
        context_mask=text_mask,
        timesteps=mx.array([sigma]),
        positions=positions,
        enabled=True,
    )

    # Run transformer
    print(f"\nRunning transformer forward pass (sigma={sigma})...")
    try:
        velocity = model(modality)
        mx.eval(velocity)
        check_array("Transformer output (velocity, patchified)", velocity)

        # Verify unpatchify
        print("\n--- Unpatchify Verification ---")
        output_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=latent_frames,
            height=latent_height,
            width=latent_width,
        )
        velocity_spatial = patchifier.unpatchify(velocity, output_shape=output_shape)
        mx.eval(velocity_spatial)
        check_array("Velocity (unpatchified, spatial)", velocity_spatial)

        expected_shape = (1, 128, latent_frames, latent_height, latent_width)
        if velocity_spatial.shape != expected_shape:
            print(f"  WARNING: Shape mismatch! Expected {expected_shape}, got {velocity_spatial.shape}")
        else:
            print(f"  Shape OK: {velocity_spatial.shape}")

    except Exception as e:
        print(f"ERROR in transformer forward: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Test VAE decoder
    print("\n--- Step 5: VAE Decoder ---")
    vae_decoder = SimpleVideoDecoder()

    # Check weights before loading
    print("\nBefore VAE weight loading:")
    check_array("mean_of_means (before)", vae_decoder.mean_of_means)
    check_array("std_of_means (before)", vae_decoder.std_of_means)
    check_array("conv_in.weight (before)", vae_decoder.conv_in.weight)

    # Load weights
    load_vae_decoder_weights(vae_decoder, weights_path)

    # Check weights after loading
    print("\nAfter VAE weight loading:")
    check_array("mean_of_means (after)", vae_decoder.mean_of_means)
    check_array("std_of_means (after)", vae_decoder.std_of_means)
    check_array("conv_in.weight (after)", vae_decoder.conv_in.weight)
    check_array("conv_out.weight (after)", vae_decoder.conv_out.weight)

    # Test VAE decode - raw decoder output
    print("\nTesting raw VAE decoder...")
    try:
        # Use the initial latent for VAE test
        video_raw = vae_decoder(latent, show_progress=False)
        mx.eval(video_raw)
        check_array("VAE output (raw)", video_raw)

        # Check final output processing
        video_processed = mx.clip((video_raw + 1) / 2, 0, 1) * 255
        check_array("VAE output (processed 0-255)", video_processed)

    except Exception as e:
        print(f"ERROR in raw VAE decode: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test decode_latent() function with bias correction
    print("\n--- Step 6: decode_latent() Function ---")
    try:
        video_final = decode_latent(latent, vae_decoder, timestep=0.05)
        mx.eval(video_final)

        print(f"decode_latent output:")
        print(f"  Shape: {video_final.shape}")
        print(f"  Dtype: {video_final.dtype}")
        print(f"  Range: [{int(video_final.min())}, {int(video_final.max())}]")
        print(f"  Mean: {float(video_final.astype(mx.float32).mean()):.1f}")

        # Check expected output format
        # Should be (T*8, H*32, W*32, 3) = (9*1, 4*32, 4*32, 3) = (9, 128, 128, 3)
        expected_shape = (num_frames, height, width, 3)
        if video_final.shape[-1] == 3:
            print(f"  Channels OK: 3 (RGB)")
        else:
            print(f"  WARNING: Expected 3 channels, got {video_final.shape[-1]}")

        if video_final.dtype == mx.uint8:
            print(f"  Dtype OK: uint8")
        else:
            print(f"  WARNING: Expected uint8, got {video_final.dtype}")

        # Quality checks
        pixel_range = int(video_final.max()) - int(video_final.min())
        if pixel_range < 20:
            print(f"  WARNING: Very low dynamic range ({pixel_range})")
        elif pixel_range < 50:
            print(f"  NOTE: Low dynamic range ({pixel_range})")
        else:
            print(f"  Dynamic range OK: {pixel_range}")

    except Exception as e:
        print(f"ERROR in decode_latent: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Debug NaN values in LTX-2 MLX pipeline")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to weights file"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Path to pre-computed text embedding"
    )
    parser.add_argument(
        "--check-weights-only",
        action="store_true",
        help="Only check weights, don't run inference"
    )
    parser.add_argument(
        "--full-denoise",
        action="store_true",
        help="Run full denoising loop (7 steps) to check convergence"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=7,
        help="Number of denoising steps (default: 7)"
    )
    parser.add_argument(
        "--use-gemma",
        action="store_true",
        help="Use real Gemma 3 text encoding instead of dummy embeddings"
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
        help="Path to Gemma 3 weights directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walking in a garden",
        help="Text prompt to encode (when using --use-gemma)"
    )
    parser.add_argument(
        "--use-cfg",
        action="store_true",
        help="Enable Classifier-Free Guidance (runs both conditional and unconditional passes)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="CFG scale factor (default: 3.0)"
    )
    parser.add_argument(
        "--check-velocity-ref",
        action="store_true",
        help="Check velocity statistics against expected reference ranges"
    )

    args = parser.parse_args()

    if args.check_weights_only:
        debug_vae_weights(args.weights)
        debug_transformer_weights(args.weights)
    elif args.full_denoise:
        # Full denoising loop test
        debug_denoising_loop(
            weights_path=args.weights,
            embedding_path=args.embedding,
            num_steps=args.steps,
            use_gemma=args.use_gemma,
            gemma_path=args.gemma_path,
            prompt=args.prompt,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            check_velocity_ref=args.check_velocity_ref,
        )
    else:
        # Standard single-step pipeline test
        debug_vae_weights(args.weights)
        debug_transformer_weights(args.weights)
        debug_inference_pipeline(args.weights, args.embedding)


if __name__ == "__main__":
    main()
