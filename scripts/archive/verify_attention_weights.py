#!/usr/bin/env python3
"""Verify attention weight shapes and values match between safetensors and model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
from safetensors import safe_open

def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 60)
    print("Checking attention weight shapes in safetensors")
    print("=" * 60)

    # LTX-2 expected dimensions
    # dim = 32 heads * 128 head_dim = 4096
    # So Q, K, V each should project from 4096 -> 4096
    # Combined QKV would be 4096 -> 12288 (3 * 4096)

    expected_dim = 4096
    expected_head_dim = 128
    expected_heads = 32

    # Load safetensors and check block 0
    with safe_open(weights_path, framework="pt") as f:
        print("\n--- Block 0 Self-Attention (attn1) ---")

        # Check for different naming conventions
        prefixes_to_check = [
            "model.diffusion_model.transformer_blocks.0.attn1.",
            "model.diffusion_model.transformer_blocks.0.video_attn.",
        ]

        for prefix in prefixes_to_check:
            found_keys = [k for k in f.keys() if k.startswith(prefix)]
            if found_keys:
                print(f"\nFound keys with prefix: {prefix}")
                for key in sorted(found_keys)[:15]:
                    tensor = f.get_tensor(key)
                    print(f"  {key.replace(prefix, '')}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

        # Check for QKV patterns
        print("\n--- Looking for QKV weight patterns ---")
        qkv_patterns = ["to_q", "to_k", "to_v", "to_qkv", "qkv", "to_out"]
        for key in sorted(f.keys()):
            if "transformer_blocks.0." in key:
                for pattern in qkv_patterns:
                    if pattern in key and "weight" in key:
                        tensor = f.get_tensor(key)
                        print(f"  {key}: shape={list(tensor.shape)}")
                        break

        # Check first 5 blocks
        print("\n--- Block weight patterns for blocks 0-4 ---")
        for block_idx in range(5):
            block_keys = [k for k in f.keys() if f"transformer_blocks.{block_idx}." in k]
            attn1_keys = [k for k in block_keys if "attn1" in k or "video_attn" in k]
            attn2_keys = [k for k in block_keys if "attn2" in k]

            print(f"\nBlock {block_idx}:")
            print(f"  Total keys: {len(block_keys)}")
            print(f"  Attn1 keys: {len(attn1_keys)}")
            print(f"  Attn2 keys: {len(attn2_keys)}")

            if attn1_keys:
                for key in sorted(attn1_keys)[:8]:
                    tensor = f.get_tensor(key)
                    short_key = key.split(f"transformer_blocks.{block_idx}.")[-1]
                    print(f"    {short_key}: {list(tensor.shape)}")

    # Now check how our model loads these
    print("\n" + "=" * 60)
    print("Checking our model structure")
    print("=" * 60)

    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType
    from LTX_2_MLX.loader import load_transformer_weights

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

    # Check model structure before loading
    print("\n--- Model attention structure (before loading) ---")
    block0 = model.transformer_blocks[0]

    print(f"Block 0 type: {type(block0).__name__}")

    # Check attn1 (self-attention)
    if hasattr(block0, 'attn1'):
        attn1 = block0.attn1
        print(f"\nattn1 type: {type(attn1).__name__}")
        if hasattr(attn1, 'to_q'):
            print(f"  to_q: {attn1.to_q.weight.shape if hasattr(attn1.to_q, 'weight') else 'no weight attr'}")
        if hasattr(attn1, 'to_k'):
            print(f"  to_k: {attn1.to_k.weight.shape if hasattr(attn1.to_k, 'weight') else 'no weight attr'}")
        if hasattr(attn1, 'to_v'):
            print(f"  to_v: {attn1.to_v.weight.shape if hasattr(attn1.to_v, 'weight') else 'no weight attr'}")
        if hasattr(attn1, 'to_out'):
            print(f"  to_out: {type(attn1.to_out)}")
    else:
        print("No attn1 attribute found!")

    # Check attn2 (cross-attention)
    if hasattr(block0, 'attn2'):
        attn2 = block0.attn2
        print(f"\nattn2 type: {type(attn2).__name__}")
        if hasattr(attn2, 'to_q'):
            print(f"  to_q: {attn2.to_q.weight.shape if hasattr(attn2.to_q, 'weight') else 'no weight attr'}")
        if hasattr(attn2, 'to_k'):
            print(f"  to_k: {attn2.to_k.weight.shape if hasattr(attn2.to_k, 'weight') else 'no weight attr'}")
        if hasattr(attn2, 'to_v'):
            print(f"  to_v: {attn2.to_v.weight.shape if hasattr(attn2.to_v, 'weight') else 'no weight attr'}")

    # Load weights and check again
    print("\n--- Loading weights ---")
    load_transformer_weights(model, weights_path)

    print("\n--- Model attention structure (after loading) ---")
    block0 = model.transformer_blocks[0]

    if hasattr(block0, 'attn1'):
        attn1 = block0.attn1
        if hasattr(attn1, 'to_q'):
            w = attn1.to_q.weight
            mx.eval(w)
            print(f"attn1.to_q.weight: shape={w.shape}, mean={float(mx.mean(w)):.6f}, std={float(mx.std(w)):.6f}")
        if hasattr(attn1, 'to_k'):
            w = attn1.to_k.weight
            mx.eval(w)
            print(f"attn1.to_k.weight: shape={w.shape}, mean={float(mx.mean(w)):.6f}, std={float(mx.std(w)):.6f}")
        if hasattr(attn1, 'to_v'):
            w = attn1.to_v.weight
            mx.eval(w)
            print(f"attn1.to_v.weight: shape={w.shape}, mean={float(mx.mean(w)):.6f}, std={float(mx.std(w)):.6f}")
        if hasattr(attn1, 'to_out'):
            if hasattr(attn1.to_out, 'weight'):
                w = attn1.to_out.weight
                mx.eval(w)
                print(f"attn1.to_out.weight: shape={w.shape}, mean={float(mx.mean(w)):.6f}, std={float(mx.std(w)):.6f}")

    # Verify by loading same weights directly and comparing
    print("\n--- Direct comparison with safetensors ---")
    with safe_open(weights_path, framework="np") as f:
        # Get to_q weight from safetensors
        st_key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
        if st_key in f.keys():
            st_weight = f.get_tensor(st_key)
            print(f"\nSafetensors {st_key}:")
            print(f"  Shape: {st_weight.shape}")
            print(f"  Mean: {st_weight.mean():.6f}")
            print(f"  Std: {st_weight.std():.6f}")
            print(f"  [0,0]: {st_weight[0,0]:.6f}")

            # Compare with model
            model_weight = np.array(model.transformer_blocks[0].attn1.to_q.weight)
            print(f"\nModel attn1.to_q.weight:")
            print(f"  Shape: {model_weight.shape}")
            print(f"  Mean: {model_weight.mean():.6f}")
            print(f"  Std: {model_weight.std():.6f}")
            print(f"  [0,0]: {model_weight[0,0]:.6f}")

            # Check if transposed
            if st_weight.shape == model_weight.shape:
                diff = np.abs(st_weight - model_weight).max()
                print(f"\n  Same shape, max diff: {diff:.6f}")
            elif st_weight.shape == model_weight.T.shape:
                diff = np.abs(st_weight - model_weight.T).max()
                print(f"\n  Shapes match after transpose, max diff: {diff:.6f}")
            else:
                print(f"\n  Shapes don't match: {st_weight.shape} vs {model_weight.shape}")

if __name__ == "__main__":
    main()
