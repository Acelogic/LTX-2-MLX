#!/usr/bin/env python3
"""
Encode text prompts using native MLX Gemma 3 for LTX-2.

This script encodes text prompts through the full LTX-2 text encoding pipeline:
1. Gemma 3 (native MLX) - Extract hidden states from all 48 layers
2. Feature extractor - Project multi-layer hidden states to 3840-dim
3. Embeddings connector - Refine with 2-layer transformer + registers
4. Caption projection - Project to 4096-dim for cross-attention

The output embedding can be used directly with generate.py --embedding flag.
"""

import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.text_encoder.gemma3 import (
    Gemma3Config,
    Gemma3Model,
    load_gemma3_weights,
)
from LTX_2_MLX.model.text_encoder.encoder import (
    create_text_encoder,
    load_text_encoder_weights,
)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm as _tqdm  # noqa: F401
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# LTX-2 system prompt for video generation
T2V_SYSTEM_PROMPT = """<bos>Describe the video in extreme detail, focusing on the visual content, without any introductory phrases."""


def load_tokenizer(model_path: str):
    """
    Load the Gemma tokenizer.

    Args:
        model_path: Path to the Gemma model directory containing tokenizer.json.

    Returns:
        Tokenizer instance.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer
    except ImportError:
        print("Error: transformers library required for tokenizer")
        print("Install with: pip install transformers")
        sys.exit(1)


def create_chat_prompt(user_prompt: str) -> str:
    """
    Create a chat-format prompt for Gemma 3.

    Args:
        user_prompt: The user's video description prompt.

    Returns:
        Formatted chat prompt string.
    """
    # Gemma 3 instruction-tuned format
    # <bos><start_of_turn>user\n{system}\n{user}<end_of_turn>\n<start_of_turn>model\n
    chat = f"<start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    return chat


def encode_prompt(
    prompt: str,
    gemma_model: Gemma3Model,
    text_encoder,
    tokenizer,
    max_length: int = 256,
) -> tuple:
    """
    Encode a text prompt through the full LTX-2 pipeline.

    Args:
        prompt: Text prompt to encode.
        gemma_model: Loaded Gemma3Model instance.
        text_encoder: Loaded VideoGemmaTextEncoderModel instance.
        tokenizer: Gemma tokenizer.
        max_length: Maximum token length (default 256).

    Returns:
        Tuple of (embedding, attention_mask) as numpy arrays.
    """
    print(f"Encoding prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Create chat format prompt
    chat_prompt = create_chat_prompt(prompt)

    # Tokenize
    print("  Tokenizing...")
    encoding = tokenizer(
        chat_prompt,
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
    print("  Running Gemma 3 forward pass...")
    last_hidden, all_hidden_states = gemma_model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mx.eval(last_hidden)

    assert all_hidden_states is not None, "Gemma model did not return hidden states"

    print(f"  Got {len(all_hidden_states)} hidden states (including embedding layer)")

    # Run through text encoder pipeline
    print("  Processing through text encoder pipeline...")
    output = text_encoder(
        hidden_states=all_hidden_states,
        attention_mask=attention_mask,
        padding_side="right",  # Gemma uses right padding
    )

    mx.eval(output.video_encoding)
    mx.eval(output.attention_mask)

    print(f"  Output embedding shape: {output.video_encoding.shape}")
    print(f"  Output embedding dtype: {output.video_encoding.dtype}")

    return (
        np.array(output.video_encoding),
        np.array(output.attention_mask),
    )


def main():
    parser = argparse.ArgumentParser(description="Encode text prompts for LTX-2")
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt to encode",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="prompt_embedding.npz",
        help="Output path for embedding (default: prompt_embedding.npz)",
    )
    parser.add_argument(
        "--gemma-weights",
        type=str,
        default="weights/gemma/gemma-3-12b-it",
        help="Path to Gemma 3 weights directory",
    )
    parser.add_argument(
        "--ltx-weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to LTX-2 weights (for text encoder projection)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length (default: 256)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LTX-2 Text Encoder (Native MLX)")
    print("=" * 60)

    # Check for weights
    if not os.path.exists(args.gemma_weights):
        print(f"\nError: Gemma weights not found at {args.gemma_weights}")
        print("\nTo download Gemma 3 12B-IT:")
        print("  python scripts/download_gemma.py")
        print("\nOr use HuggingFace CLI:")
        print("  huggingface-cli download google/gemma-3-12b-it --local-dir weights/gemma/gemma-3-12b-it")
        sys.exit(1)

    if not os.path.exists(args.ltx_weights):
        print(f"\nError: LTX-2 weights not found at {args.ltx_weights}")
        sys.exit(1)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = load_tokenizer(args.gemma_weights)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Load Gemma 3 model
    print("\n[2/4] Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, args.gemma_weights)

    # Load text encoder projection layers
    print("\n[3/4] Loading text encoder...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, args.ltx_weights)

    # Encode prompt
    print("\n[4/4] Encoding prompt...")
    embedding, mask = encode_prompt(
        args.prompt,
        gemma,
        text_encoder,
        tokenizer,
        max_length=args.max_length,
    )

    # Save embedding
    print(f"\nSaving embedding to {args.output}...")
    np.savez(
        args.output,
        embedding=embedding,
        attention_mask=mask,
        prompt=args.prompt,
    )

    print(f"\nDone! Embedding saved to {args.output}")
    print(f"  Shape: {embedding.shape}")
    print(f"  Use with: python scripts/generate.py \"your prompt\" --embedding {args.output}")


if __name__ == "__main__":
    main()
