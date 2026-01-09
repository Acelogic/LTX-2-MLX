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


def debug_inference_pipeline(weights_path: str, embedding_path: str = None):
    """Debug the full inference pipeline step by step."""
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, create_position_grid
    from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier
    from LTX_2_MLX.types import VideoLatentShape
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.model.video_vae.simple_decoder import SimpleVideoDecoder, load_vae_decoder_weights

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
        check_array("Transformer output (velocity)", velocity)
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

    # Test VAE decode
    print("\nTesting VAE decode...")
    try:
        # Use the initial latent for VAE test
        video = vae_decoder(latent, show_progress=False)
        mx.eval(video)
        check_array("VAE output (raw)", video)

        # Check final output processing
        video_processed = mx.clip((video + 1) / 2, 0, 1) * 255
        check_array("VAE output (processed 0-255)", video_processed)

    except Exception as e:
        print(f"ERROR in VAE decode: {e}")
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

    args = parser.parse_args()

    if args.check_weights_only:
        debug_vae_weights(args.weights)
        debug_transformer_weights(args.weights)
    else:
        debug_vae_weights(args.weights)
        debug_transformer_weights(args.weights)
        debug_inference_pipeline(args.weights, args.embedding)


if __name__ == "__main__":
    main()
