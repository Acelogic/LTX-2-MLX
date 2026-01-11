"""Analyze text embedding differentiation in detail."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def analyze_dimension_differences(emb1: mx.array, emb2: mx.array, name: str = ""):
    """Analyze per-dimension differences between embeddings."""
    print(f"\n{'=' * 60}")
    print(f"Dimension-wise Analysis: {name}")
    print(f"{'=' * 60}")

    # Flatten to [seq_len, hidden_dim]
    e1 = emb1[0]  # [T, D]
    e2 = emb2[0]  # [T, D]

    # Per-dimension absolute difference
    abs_diff = mx.abs(e1 - e2)  # [T, D]

    # Average across sequence
    avg_diff_per_dim = mx.mean(abs_diff, axis=0)  # [D]

    print(f"\nPer-dimension average absolute difference:")
    print(f"  Mean: {float(mx.mean(avg_diff_per_dim)):.6f}")
    print(f"  Std: {float(mx.std(avg_diff_per_dim)):.6f}")
    print(f"  Max: {float(mx.max(avg_diff_per_dim)):.6f}")
    print(f"  Min: {float(mx.min(avg_diff_per_dim)):.6f}")

    # Find dimensions with largest differences
    sorted_indices = mx.argsort(-avg_diff_per_dim)
    top_k = 20
    print(f"\n  Top {top_k} dimensions with largest difference:")
    for i in range(top_k):
        idx = int(sorted_indices[i])
        diff = float(avg_diff_per_dim[idx])
        print(f"    dim {idx}: {diff:.6f}")

    # Percentage of dimensions with meaningful difference
    threshold = 0.01
    meaningful_dims = int(mx.sum(avg_diff_per_dim > threshold))
    total_dims = avg_diff_per_dim.shape[0]
    print(f"\n  Dimensions with diff > {threshold}: {meaningful_dims}/{total_dims} ({100*meaningful_dims/total_dims:.1f}%)")

    threshold = 0.001
    meaningful_dims = int(mx.sum(avg_diff_per_dim > threshold))
    print(f"  Dimensions with diff > {threshold}: {meaningful_dims}/{total_dims} ({100*meaningful_dims/total_dims:.1f}%)")

    # Per-token analysis
    print(f"\nPer-token average absolute difference:")
    avg_diff_per_token = mx.mean(abs_diff, axis=1)  # [T]
    print(f"  Mean: {float(mx.mean(avg_diff_per_token)):.6f}")
    print(f"  Max: {float(mx.max(avg_diff_per_token)):.6f}")
    print(f"  Non-zero tokens: {int(mx.sum(avg_diff_per_token > 0.001))}")

    # Euclidean distance
    euclidean = float(mx.sqrt(mx.sum((e1 - e2) ** 2)))
    print(f"\n  Total Euclidean distance: {euclidean:.4f}")

    # Cosine similarity
    dot = float(mx.sum(e1.flatten() * e2.flatten()))
    norm1 = float(mx.sqrt(mx.sum(e1 ** 2)))
    norm2 = float(mx.sqrt(mx.sum(e2 ** 2)))
    cosine = dot / (norm1 * norm2 + 1e-8)
    print(f"  Cosine similarity: {cosine:.6f}")

    return avg_diff_per_dim


def test_cross_attention_response(emb1: mx.array, emb2: mx.array):
    """Test if cross-attention produces different outputs for different embeddings."""
    print(f"\n{'=' * 60}")
    print("Cross-Attention Response Test")
    print(f"{'=' * 60}")

    from LTX_2_MLX.model.transformer.attention import Attention

    # Create a cross-attention layer similar to the diffusion transformer
    cross_attn = Attention(
        query_dim=4096,
        context_dim=4096,  # After caption projection
        heads=32,
        dim_head=128,
        norm_eps=1e-6,
    )

    # Initialize with random weights (just to test the mechanism)
    mx.eval(cross_attn.parameters())

    # Create a random query (simulating transformer hidden states)
    np.random.seed(42)
    query = mx.array(np.random.randn(1, 16, 4096).astype(np.float32)) * 0.1

    # Project embeddings to cross-attention dimension (3840 -> 4096)
    # For testing, we'll pad with zeros
    def pad_to_4096(x):
        # x is [B, T, 3840], pad to [B, T, 4096]
        padding = mx.zeros((x.shape[0], x.shape[1], 4096 - x.shape[2]))
        return mx.concatenate([x, padding], axis=-1)

    context1 = pad_to_4096(emb1)
    context2 = pad_to_4096(emb2)

    # Run cross-attention
    out1 = cross_attn(query, context=context1)
    out2 = cross_attn(query, context=context2)
    mx.eval(out1, out2)

    # Analyze difference in outputs
    diff = mx.abs(out1 - out2)
    print(f"\nCross-attention output difference:")
    print(f"  Mean abs diff: {float(mx.mean(diff)):.6f}")
    print(f"  Max abs diff: {float(mx.max(diff)):.6f}")

    # Correlation of outputs
    flat1 = out1.flatten()
    flat2 = out2.flatten()
    corr = float(mx.sum(flat1 * flat2)) / (
        float(mx.sqrt(mx.sum(flat1 ** 2))) * float(mx.sqrt(mx.sum(flat2 ** 2))) + 1e-8
    )
    print(f"  Output correlation: {corr:.6f}")

    # Check attention patterns
    print("\n  Testing with learned weights from real model...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Text Embedding Differentiation Analysis")
    print("=" * 60)

    # Test prompts
    prompts = [
        ("A blue ball on grass", "A red ball on grass"),
        ("A cat sleeping", "A dog running"),
        ("Fireworks exploding in night sky", "Ocean waves crashing on beach"),
    ]

    import os
    if not os.path.exists(args.gemma_path):
        print(f"ERROR: Gemma weights not found at {args.gemma_path}")
        print("Using random embeddings for testing...")

        # Create random embeddings with known difference
        np.random.seed(42)
        base = np.random.randn(1, 256, 3840).astype(np.float32) * 0.1

        # Add small perturbation
        diff = np.random.randn(1, 256, 3840).astype(np.float32) * 0.01

        emb1 = mx.array(base)
        emb2 = mx.array(base + diff)

        analyze_dimension_differences(emb1, emb2, "Random with 1% perturbation")
        test_cross_attention_response(emb1, emb2)
        return

    # Load encoder
    from scripts.generate import encode_with_gemma

    for prompt1, prompt2 in prompts:
        print(f"\n\n{'#' * 60}")
        print(f"Comparing:")
        print(f"  '{prompt1}'")
        print(f"  '{prompt2}'")
        print(f"{'#' * 60}")

        # Encode both prompts
        emb1, mask1 = encode_with_gemma(
            prompt=prompt1,
            gemma_path=args.gemma_path,
            ltx_weights_path=args.weights,
            max_length=256,
        )

        emb2, mask2 = encode_with_gemma(
            prompt=prompt2,
            gemma_path=args.gemma_path,
            ltx_weights_path=args.weights,
            max_length=256,
        )

        if emb1 is None or emb2 is None:
            print("  Failed to encode prompts")
            continue

        mx.eval(emb1, emb2)

        # Analyze
        analyze_dimension_differences(emb1, emb2, "Full embeddings")

        # Test with only non-zero positions
        valid1 = int(mx.sum(mask1))
        valid2 = int(mx.sum(mask2))
        print(f"\n  Valid tokens: {valid1}, {valid2}")

        # Test cross-attention response
        test_cross_attention_response(emb1, emb2)


if __name__ == "__main__":
    main()
