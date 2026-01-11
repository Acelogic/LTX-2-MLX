"""Debug AdaLN scale_shift_table values to ensure they're loaded correctly."""

import argparse
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open


def check_scale_shift_tables(weights_path: str):
    """Check scale_shift_table values in the model weights."""

    print(f"Checking AdaLN tables in: {weights_path}\n")

    # Collect all scale_shift_table keys
    scale_shift_keys = []

    with safe_open(weights_path, framework="pt") as f:
        all_keys = list(f.keys())

        for key in all_keys:
            if "scale_shift_table" in key:
                scale_shift_keys.append(key)
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                np_arr = tensor.numpy()

                print(f"{key}")
                print(f"  Shape: {np_arr.shape}")
                print(f"  Dtype: {np_arr.dtype}")
                print(f"  Range: [{np_arr.min():.4f}, {np_arr.max():.4f}]")
                print(f"  Mean: {np_arr.mean():.4f}, Std: {np_arr.std():.4f}")

                # Check if all zeros
                if np_arr.max() == 0 and np_arr.min() == 0:
                    print(f"  *** ALL ZEROS! ***")
                print()

    print(f"Found {len(scale_shift_keys)} scale_shift_table entries")

    # Also check for video timestep embedding layers
    print("\n" + "="*60)
    print("Timestep embedding related weights:")
    print("="*60 + "\n")

    with safe_open(weights_path, framework="pt") as f:
        for key in all_keys:
            if "timestep" in key.lower() or "time_embed" in key.lower() or "t_embedder" in key.lower():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                np_arr = tensor.numpy()
                print(f"{key}")
                print(f"  Shape: {np_arr.shape}")
                print(f"  Range: [{np_arr.min():.4f}, {np_arr.max():.4f}]")
                print()


def check_loaded_model(weights_path: str):
    """Load model and check scale_shift_table values after loading."""

    print("\n" + "="*60)
    print("Loading model and checking scale_shift_table in loaded model:")
    print("="*60 + "\n")

    from LTX_2_MLX.model.transformer import LTXAVModel
    from LTX_2_MLX.loader.weight_converter import load_av_transformer_weights

    # Create model with default parameters
    model = LTXAVModel()

    # Load weights
    load_av_transformer_weights(model, weights_path, use_fp8=True)
    mx.eval(model.parameters())

    # Check first few transformer blocks
    for i in range(min(3, len(model.transformer_blocks))):
        block = model.transformer_blocks[i]
        table = block.scale_shift_table

        print(f"Block {i} scale_shift_table:")
        print(f"  Shape: {table.shape}")
        print(f"  Dtype: {table.dtype}")
        print(f"  Range: [{float(mx.min(table)):.4f}, {float(mx.max(table)):.4f}]")
        print(f"  Mean: {float(mx.mean(table)):.4f}, Std: {float(mx.std(table)):.4f}")

        if float(mx.max(table)) == 0 and float(mx.min(table)) == 0:
            print(f"  *** ALL ZEROS - NOT LOADED! ***")
        print()

    # Check audio scale_shift_table
    print("Audio scale_shift_table (Block 0):")
    audio_table = model.transformer_blocks[0].audio_scale_shift_table
    print(f"  Shape: {audio_table.shape}")
    print(f"  Range: [{float(mx.min(audio_table)):.4f}, {float(mx.max(audio_table)):.4f}]")
    print()


def check_timestep_embedding(weights_path: str):
    """Check the timestep embedding computation."""

    print("\n" + "="*60)
    print("Testing timestep embedding computation:")
    print("="*60 + "\n")

    from LTX_2_MLX.model.transformer import LTXAVModel
    from LTX_2_MLX.model.transformer.timestep_embedding import (
        get_timestep_embedding
    )
    from LTX_2_MLX.loader.weight_converter import load_av_transformer_weights

    # Create model with default parameters
    model = LTXAVModel()

    # Load weights
    load_av_transformer_weights(model, weights_path, use_fp8=True)
    mx.eval(model.parameters())

    # Test timestep embedding
    timestep = mx.array([[1.0]])  # sigma = 1.0

    # Get the raw sinusoidal embedding
    raw_embed = get_timestep_embedding(timestep * 1000, 256)  # Scale by 1000
    print(f"Raw timestep embedding (sigma=1.0):")
    print(f"  Shape: {raw_embed.shape}")
    print(f"  Range: [{float(mx.min(raw_embed)):.4f}, {float(mx.max(raw_embed)):.4f}]")
    print(f"  Mean: {float(mx.mean(raw_embed)):.4f}, Std: {float(mx.std(raw_embed)):.4f}")
    print()

    # Check the full timestep embedder output
    print("Checking video_timestep_embedder output:")
    video_timestep = model.video_timestep_embedder(timestep)
    print(f"  Shape: {video_timestep.shape}")
    print(f"  Range: [{float(mx.min(video_timestep)):.4f}, {float(mx.max(video_timestep)):.4f}]")
    print(f"  Mean: {float(mx.mean(video_timestep)):.4f}, Std: {float(mx.std(video_timestep)):.4f}")
    print()

    # Test at sigma = 0.5
    timestep_05 = mx.array([[0.5]])
    video_timestep_05 = model.video_timestep_embedder(timestep_05)
    print("Checking video_timestep_embedder output (sigma=0.5):")
    print(f"  Shape: {video_timestep_05.shape}")
    print(f"  Range: [{float(mx.min(video_timestep_05)):.4f}, {float(mx.max(video_timestep_05)):.4f}]")
    print()

    # Check difference between sigma=1.0 and sigma=0.5
    diff = video_timestep - video_timestep_05
    print("Difference between sigma=1.0 and sigma=0.5:")
    print(f"  Max abs diff: {float(mx.max(mx.abs(diff))):.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to model weights",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only check raw weights file without loading model",
    )
    args = parser.parse_args()

    # Check raw weights
    check_scale_shift_tables(args.weights)

    if not args.raw_only:
        # Load model and check
        check_loaded_model(args.weights)

        # Check timestep embedding
        check_timestep_embedding(args.weights)


if __name__ == "__main__":
    main()
