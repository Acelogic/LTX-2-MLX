#!/usr/bin/env python3
"""
Validate MLX text encoder outputs against PyTorch reference.

This script compares the text encoding outputs between:
1. MLX native Gemma 3 + LTX-2 text encoder
2. PyTorch reference implementation

Use this to verify the MLX port produces correct embeddings.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def load_tokenizer(model_path: str):
    """Load the Gemma tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer
    except ImportError:
        print("Error: transformers library required")
        print("Install with: pip install transformers")
        return None


def encode_with_mlx(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    tokenizer,
    max_length: int = 256,
) -> tuple:
    """Encode prompt using MLX implementation."""
    from LTX_2_MLX.model.text_encoder.gemma3 import (
        Gemma3Config,
        Gemma3Model,
        load_gemma3_weights,
    )
    from LTX_2_MLX.model.text_encoder.encoder import (
        create_text_encoder,
        load_text_encoder_weights,
    )

    print("\n[MLX] Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, gemma_path)

    print("[MLX] Loading text encoder projection...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, ltx_weights_path)

    # CRITICAL: Use right padding to avoid NaN in attention
    tokenizer.padding_side = "right"

    # Create chat prompt
    T2V_SYSTEM_PROMPT = "Describe the video in extreme detail, focusing on the visual content, without any introductory phrases."
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

    print(f"[MLX] Token count: {int(attention_mask.sum())}/{max_length}")

    # Run Gemma
    print("[MLX] Running Gemma 3 forward pass...")
    last_hidden, all_hidden_states = gemma(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mx.eval(last_hidden)

    # Run text encoder (without caption projection - transformer does it)
    print("[MLX] Processing through text encoder pipeline...")
    encoded = text_encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=attention_mask,
        padding_side="right",
    )

    large_value = 1e9
    connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
    connector_mask = connector_mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])

    encoded, output_mask = text_encoder.embeddings_connector(encoded, connector_mask)
    binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)
    encoded = encoded * binary_mask[:, :, None]

    mx.eval(encoded)

    print(f"[MLX] Output shape: {encoded.shape}")

    return np.array(encoded), np.array(binary_mask)


def encode_with_pytorch(
    prompt: str,
    ltx_weights_path: str,
    tokenizer,
    max_length: int = 256,
) -> tuple:
    """
    Encode prompt using PyTorch reference implementation.

    This requires the original LTX-2 repository to be installed.
    """
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        print("Error: torch and safetensors required for PyTorch validation")
        return None, None

    print("\n[PyTorch] Note: PyTorch reference encoding not implemented yet")
    print("[PyTorch] To compare, save embeddings from the original LTX-2 repo")
    print("[PyTorch] and load them here for comparison.")

    return None, None


def compare_embeddings(
    mlx_embedding: np.ndarray,
    pytorch_embedding: np.ndarray,
    name: str = "embedding",
) -> dict:
    """Compare two embeddings and return statistics."""
    if pytorch_embedding is None:
        return {"error": "PyTorch embedding not available"}

    # Flatten for comparison
    mlx_flat = mlx_embedding.flatten()
    pt_flat = pytorch_embedding.flatten()

    # Basic statistics
    stats = {
        "mlx_mean": float(mlx_flat.mean()),
        "mlx_std": float(mlx_flat.std()),
        "mlx_min": float(mlx_flat.min()),
        "mlx_max": float(mlx_flat.max()),
        "pytorch_mean": float(pt_flat.mean()),
        "pytorch_std": float(pt_flat.std()),
        "pytorch_min": float(pt_flat.min()),
        "pytorch_max": float(pt_flat.max()),
    }

    # Comparison metrics
    diff = mlx_flat - pt_flat
    stats["l2_diff"] = float(np.linalg.norm(diff))
    stats["max_abs_diff"] = float(np.abs(diff).max())
    stats["mean_abs_diff"] = float(np.abs(diff).mean())

    # Cosine similarity
    mlx_norm = np.linalg.norm(mlx_flat)
    pt_norm = np.linalg.norm(pt_flat)
    if mlx_norm > 0 and pt_norm > 0:
        cosine_sim = np.dot(mlx_flat, pt_flat) / (mlx_norm * pt_norm)
        stats["cosine_similarity"] = float(cosine_sim)
    else:
        stats["cosine_similarity"] = 0.0

    # Correlation
    if mlx_flat.std() > 0 and pt_flat.std() > 0:
        correlation = np.corrcoef(mlx_flat, pt_flat)[0, 1]
        stats["correlation"] = float(correlation)
    else:
        stats["correlation"] = 0.0

    return stats


def print_stats(stats: dict, name: str):
    """Pretty print comparison statistics."""
    print(f"\n{'='*50}")
    print(f"Comparison: {name}")
    print(f"{'='*50}")

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    print(f"\nMLX Statistics:")
    print(f"  Mean: {stats['mlx_mean']:.6f}")
    print(f"  Std:  {stats['mlx_std']:.6f}")
    print(f"  Min:  {stats['mlx_min']:.6f}")
    print(f"  Max:  {stats['mlx_max']:.6f}")

    if "pytorch_mean" in stats:
        print(f"\nPyTorch Statistics:")
        print(f"  Mean: {stats['pytorch_mean']:.6f}")
        print(f"  Std:  {stats['pytorch_std']:.6f}")
        print(f"  Min:  {stats['pytorch_min']:.6f}")
        print(f"  Max:  {stats['pytorch_max']:.6f}")

        print(f"\nComparison Metrics:")
        print(f"  Cosine Similarity: {stats['cosine_similarity']:.6f}")
        print(f"  Correlation:       {stats['correlation']:.6f}")
        print(f"  L2 Difference:     {stats['l2_diff']:.6f}")
        print(f"  Max Abs Diff:      {stats['max_abs_diff']:.6f}")
        print(f"  Mean Abs Diff:     {stats['mean_abs_diff']:.6f}")

        # Quality assessment
        print(f"\nQuality Assessment:")
        if stats["cosine_similarity"] > 0.99:
            print("  [EXCELLENT] Cosine similarity > 0.99")
        elif stats["cosine_similarity"] > 0.95:
            print("  [GOOD] Cosine similarity > 0.95")
        elif stats["cosine_similarity"] > 0.90:
            print("  [ACCEPTABLE] Cosine similarity > 0.90")
        else:
            print("  [WARNING] Cosine similarity < 0.90 - may indicate issues")


def validate_mlx_only(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    max_length: int = 256,
):
    """Validate MLX encoding produces reasonable outputs."""
    print("=" * 60)
    print("MLX Text Encoder Validation")
    print("=" * 60)
    print(f"Prompt: {prompt}")

    tokenizer = load_tokenizer(gemma_path)
    if tokenizer is None:
        return

    mlx_emb, mlx_mask = encode_with_mlx(
        prompt, gemma_path, ltx_weights_path, tokenizer, max_length
    )

    # Print MLX statistics
    print(f"\n{'='*50}")
    print("MLX Embedding Statistics")
    print(f"{'='*50}")
    print(f"Shape: {mlx_emb.shape}")
    print(f"Dtype: {mlx_emb.dtype}")
    print(f"Mean:  {mlx_emb.mean():.6f}")
    print(f"Std:   {mlx_emb.std():.6f}")
    print(f"Min:   {mlx_emb.min():.6f}")
    print(f"Max:   {mlx_emb.max():.6f}")

    # Check for issues
    print(f"\nHealth Checks:")

    # Check for NaN/Inf
    if np.isnan(mlx_emb).any():
        print("  [FAIL] Contains NaN values!")
    else:
        print("  [PASS] No NaN values")

    if np.isinf(mlx_emb).any():
        print("  [FAIL] Contains Inf values!")
    else:
        print("  [PASS] No Inf values")

    # Check for reasonable range
    if mlx_emb.std() < 0.01:
        print("  [WARN] Very low std - embeddings may be collapsed")
    elif mlx_emb.std() > 10:
        print("  [WARN] Very high std - embeddings may be exploding")
    else:
        print(f"  [PASS] Std in reasonable range ({mlx_emb.std():.4f})")

    # Check mask
    active_tokens = mlx_mask.sum()
    print(f"\nAttention Mask:")
    print(f"  Active tokens: {int(active_tokens)}/{mlx_mask.shape[1]}")

    return mlx_emb, mlx_mask


def compare_with_saved(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    reference_path: str,
    max_length: int = 256,
):
    """Compare MLX output with saved PyTorch reference."""
    print("=" * 60)
    print("MLX vs PyTorch Reference Comparison")
    print("=" * 60)

    tokenizer = load_tokenizer(gemma_path)
    if tokenizer is None:
        return

    # Load reference
    print(f"\nLoading reference from {reference_path}...")
    ref_data = np.load(reference_path)
    ref_emb = ref_data["embedding"]
    ref_mask = ref_data.get("attention_mask", None)

    print(f"Reference shape: {ref_emb.shape}")

    # Encode with MLX
    mlx_emb, mlx_mask = encode_with_mlx(
        prompt, gemma_path, ltx_weights_path, tokenizer, max_length
    )

    # Compare
    stats = compare_embeddings(mlx_emb, ref_emb, "Text Embedding")
    print_stats(stats, "MLX vs PyTorch Reference")


def main():
    parser = argparse.ArgumentParser(
        description="Validate MLX text encoder against PyTorch reference"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt to encode",
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
        help="Path to Gemma 3 weights directory",
    )
    parser.add_argument(
        "--ltx-weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to LTX-2 weights",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to saved PyTorch reference embedding (.npz)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save MLX embedding to this path (.npz)",
    )

    args = parser.parse_args()

    # Check paths
    if not os.path.exists(args.gemma_path):
        print(f"Error: Gemma weights not found at {args.gemma_path}")
        print("Run: python scripts/download_gemma.py")
        sys.exit(1)

    if not os.path.exists(args.ltx_weights):
        print(f"Error: LTX-2 weights not found at {args.ltx_weights}")
        sys.exit(1)

    if args.reference:
        # Compare with saved reference
        compare_with_saved(
            args.prompt,
            args.gemma_path,
            args.ltx_weights,
            args.reference,
            args.max_length,
        )
    else:
        # Just validate MLX output
        mlx_emb, mlx_mask = validate_mlx_only(
            args.prompt,
            args.gemma_path,
            args.ltx_weights,
            args.max_length,
        )

        # Save if requested
        if args.save and mlx_emb is not None:
            print(f"\nSaving embedding to {args.save}...")
            np.savez(
                args.save,
                embedding=mlx_emb,
                attention_mask=mlx_mask,
                prompt=args.prompt,
            )
            print("Done!")


if __name__ == "__main__":
    main()
