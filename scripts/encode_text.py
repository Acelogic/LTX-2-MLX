#!/usr/bin/env python3
"""
Encode text prompts to embeddings using Gemma.

This script encodes text prompts and saves the embeddings for use with generate.py.
Requires Gemma model access via mlx-lm.

Usage:
    python scripts/encode_text.py "Your prompt here" --output prompt_embedding.npz
    python scripts/generate.py --embedding prompt_embedding.npz --height 128 --width 128
"""

import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def encode_with_gemma(
    prompt: str,
    model_path: str = "google/gemma-2b",
    max_tokens: int = 256,
) -> tuple:
    """
    Encode text prompt using Gemma model.

    Args:
        prompt: Text prompt to encode.
        model_path: Path to Gemma model (HuggingFace ID or local path).
        max_tokens: Maximum number of tokens.

    Returns:
        Tuple of (hidden_states list, attention_mask).
    """
    try:
        from mlx_lm import load
        from mlx_lm.tokenizer_utils import TokenizerWrapper
    except ImportError:
        print("Error: mlx-lm is required. Install with: pip install mlx-lm")
        sys.exit(1)

    print(f"Loading Gemma model from {model_path}...")

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTo use Gemma, you need to:")
        print("1. Accept the license at https://huggingface.co/google/gemma-2b")
        print("2. Login with: huggingface-cli login")
        print("\nAlternatively, convert a local model with mlx_lm.convert")
        sys.exit(1)

    print("Tokenizing prompt...")

    # Tokenize
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    # Pad to max_tokens
    attention_mask = [1] * len(tokens) + [0] * (max_tokens - len(tokens))
    tokens = tokens + [tokenizer.pad_token_id or 0] * (max_tokens - len(tokens))

    input_ids = mx.array([tokens])
    attention_mask = mx.array([attention_mask])

    print(f"Running Gemma forward pass ({len(tokens)} tokens)...")

    # Get hidden states from all layers
    # Note: This requires modifying the model to output hidden states
    # For now, we'll use the last hidden state as a placeholder

    # Run model
    with mx.stream(mx.cpu):  # Use CPU for stability
        outputs = model(input_ids)

    # Get last hidden state
    # Shape: [batch, seq_len, hidden_dim]
    last_hidden = outputs

    print(f"Hidden state shape: {last_hidden.shape}")

    return last_hidden, attention_mask


def create_dummy_embedding(
    prompt: str,
    max_tokens: int = 256,
    hidden_dim: int = 3840,
    cross_attention_dim: int = 4096,
) -> tuple:
    """
    Create a dummy embedding based on prompt hash.

    This is used when Gemma is not available.
    The embedding is deterministic based on the prompt.

    Args:
        prompt: Text prompt.
        max_tokens: Number of tokens.
        hidden_dim: Gemma hidden dimension (not used in output).
        cross_attention_dim: Output dimension for cross-attention.

    Returns:
        Tuple of (embedding, attention_mask).
    """
    # Use prompt hash for deterministic random
    mx.random.seed(hash(prompt) % (2**31))

    # Create embedding in cross-attention dimension
    embedding = mx.random.normal(shape=(1, max_tokens, cross_attention_dim)) * 0.1
    attention_mask = mx.ones((1, max_tokens))

    return embedding, attention_mask


def main():
    parser = argparse.ArgumentParser(
        description="Encode text prompts for LTX-2 video generation"
    )
    parser.add_argument("prompt", type=str, help="Text prompt to encode")
    parser.add_argument(
        "--output", type=str, default="prompt_embedding.npz",
        help="Output path for embedding"
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-2b",
        help="Gemma model path"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum number of tokens"
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="Create dummy embedding (for testing without Gemma)"
    )

    args = parser.parse_args()

    if args.dummy:
        print(f"Creating dummy embedding for: {args.prompt}")
        embedding, mask = create_dummy_embedding(
            args.prompt,
            max_tokens=args.max_tokens,
        )
        print(f"  Shape: {embedding.shape}")
    else:
        print(f"Encoding: {args.prompt}")
        embedding, mask = encode_with_gemma(
            args.prompt,
            model_path=args.model,
            max_tokens=args.max_tokens,
        )

    # Save embedding
    print(f"Saving to {args.output}...")
    np.savez(
        args.output,
        embedding=np.array(embedding),
        attention_mask=np.array(mask),
        prompt=args.prompt,
    )
    print("Done!")


if __name__ == "__main__":
    main()
