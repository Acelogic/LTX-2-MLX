#!/usr/bin/env python3
"""
Encode text prompts using native MLX Gemma 3 + LTX-2 text encoder.

This script uses the full MLX pipeline:
1. Gemma 3 12B (MLX) - extracts hidden states from all 48 layers
2. Feature Extractor - aggregates multi-layer hidden states
3. Embeddings Connector - 1D transformer refinement
4. Caption Projection - projects to cross-attention dimension (4096)

Usage:
    python scripts/encode_text_mlx.py "A cat walking through a garden" \
        --gemma-path weights/gemma-3-12b \
        --ltx-weights weights/ltx-2/ltx-2-19b-distilled.safetensors \
        --output prompt_embedding.npz

    python scripts/generate.py --embedding prompt_embedding.npz
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.text_encoder.gemma3 import create_gemma3_model, Gemma3Config
from LTX_2_MLX.model.text_encoder.encoder import (
    VideoGemmaTextEncoderModel,
    create_text_encoder,
    load_text_encoder_weights,
)


def load_tokenizer(gemma_path: str):
    """Load Gemma tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers is required for tokenization.")
        print("Install with: pip install transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    return tokenizer


def encode_prompt(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    max_tokens: int = 256,
) -> tuple:
    """
    Encode text prompt using MLX Gemma 3 + LTX-2 text encoder.

    Args:
        prompt: Text prompt to encode.
        gemma_path: Path to Gemma 3 weights directory.
        ltx_weights_path: Path to LTX-2 weights file.
        max_tokens: Maximum number of tokens.

    Returns:
        Tuple of (embedding [1, seq, 4096], attention_mask [1, seq]).
    """
    print(f"Loading tokenizer from {gemma_path}...")
    tokenizer = load_tokenizer(gemma_path)

    print(f"Tokenizing prompt: {prompt}")
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"  Token count: {len(tokens)}")

    # Truncate or pad to max_tokens
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        attention_mask = [1] * max_tokens
    else:
        attention_mask = [1] * len(tokens) + [0] * (max_tokens - len(tokens))
        tokens = tokens + [tokenizer.pad_token_id or 0] * (max_tokens - len(tokens))

    input_ids = mx.array([tokens])
    attention_mask = mx.array([attention_mask])

    print(f"  Padded to {max_tokens} tokens")

    # Load Gemma 3 model
    print(f"\nLoading Gemma 3 model from {gemma_path}...")
    gemma = create_gemma3_model(gemma_path)
    mx.eval(gemma.parameters())
    print("  Gemma 3 loaded!")

    # Run Gemma to get hidden states
    print("\nRunning Gemma 3 forward pass...")
    _, hidden_states = gemma(input_ids, attention_mask, output_hidden_states=True)
    mx.eval(hidden_states)
    print(f"  Got {len(hidden_states)} hidden states (embedding + 48 layers)")
    print(f"  Hidden state shape: {hidden_states[0].shape}")

    # Load LTX-2 text encoder components
    print(f"\nLoading LTX-2 text encoder from {ltx_weights_path}...")
    text_encoder = create_text_encoder(
        hidden_dim=3840,
        num_gemma_layers=49,  # embedding + 48 layers
    )
    load_text_encoder_weights(text_encoder, ltx_weights_path)
    mx.eval(text_encoder.parameters())

    # Encode through LTX-2 pipeline
    # Note: We output 3840-dim embeddings (pre-projection) because the
    # transformer has its own caption_projection that expects 3840 input.
    print("\nProcessing through LTX-2 text encoder pipeline...")

    # Step 1: Feature extraction from hidden states
    stacked = mx.stack(hidden_states, axis=-1)  # [B, T, D, L]
    from LTX_2_MLX.model.text_encoder.feature_extractor import norm_and_concat_padded_batch
    sequence_lengths = attention_mask.sum(axis=-1).astype(mx.int32)
    normed = norm_and_concat_padded_batch(stacked, sequence_lengths, padding_side="right")
    projected = text_encoder.feature_extractor.aggregate_embed(normed)
    mx.eval(projected)
    print(f"  After feature extraction: {projected.shape}")

    # Step 2: Embeddings connector
    connector_mask = text_encoder._convert_to_additive_mask(attention_mask, projected.dtype)
    encoded, output_mask = text_encoder.embeddings_connector(projected, connector_mask)
    mx.eval(encoded)
    print(f"  After connector: {encoded.shape}")

    # Note: We skip caption_projection here - the transformer does its own projection
    # The output should be 3840-dimensional for the transformer's caption_projection

    # Convert mask back to binary
    binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)
    encoded = encoded * binary_mask[:, :, None]

    embedding = encoded
    mask = binary_mask

    print(f"  Output embedding shape: {embedding.shape}")
    print(f"  Output mask shape: {mask.shape}")

    return embedding, mask


def main():
    parser = argparse.ArgumentParser(
        description="Encode text prompts using MLX Gemma 3 + LTX-2 pipeline"
    )
    parser.add_argument("prompt", type=str, help="Text prompt to encode")
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
        help="Path to LTX-2 weights file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prompt_embedding.npz",
        help="Output path for embedding",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens",
    )

    args = parser.parse_args()

    embedding, mask = encode_prompt(
        args.prompt,
        args.gemma_path,
        args.ltx_weights,
        args.max_tokens,
    )

    print(f"\nSaving to {args.output}...")
    np.savez(
        args.output,
        embedding=np.array(embedding),
        attention_mask=np.array(mask),
        prompt=args.prompt,
    )
    print("Done!")
    print(f"\nUse with: python scripts/generate.py --embedding {args.output}")


if __name__ == "__main__":
    main()
