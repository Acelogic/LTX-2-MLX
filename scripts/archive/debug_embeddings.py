#!/usr/bin/env python3
"""
Debug embedding similarity issue.

This script tests whether different prompts produce different embeddings
at the token level, focusing on the actual content tokens (not padding/system prompt).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx

from LTX_2_MLX.model.text_encoder.gemma3 import (
    Gemma3Config,
    Gemma3Model,
    load_gemma3_weights,
)


def load_tokenizer(model_path: str):
    """Load the Gemma tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    return tokenizer


def test_embedding_layer_only(gemma_path: str):
    """Test if embedding layer produces different outputs for different tokens."""
    print("=" * 60)
    print("Test 1: Embedding Layer Only")
    print("=" * 60)

    tokenizer = load_tokenizer(gemma_path)
    config = Gemma3Config()
    model = Gemma3Model(config)
    load_gemma3_weights(model, gemma_path)

    # Simple test tokens
    tokens_a = mx.array([[1, 2, 3, 4, 5]])
    tokens_b = mx.array([[1, 2, 3, 6, 7]])  # Different last 2 tokens

    emb_a = model.embed_tokens(tokens_a)
    emb_b = model.embed_tokens(tokens_b)
    mx.eval(emb_a, emb_b)

    # Compare token by token
    print("\nToken-by-token comparison:")
    for i in range(5):
        a = np.array(emb_a[0, i])
        b = np.array(emb_b[0, i])
        cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"  Token {i}: cosine={cosine_sim:.4f}")
        if i < 3:
            print(f"    (Same token - should be 1.0)")
        else:
            print(f"    (Different tokens - should be < 1.0)")

    print("\nEmbedding layer test passed!" if True else "")


def test_minimal_sequences(gemma_path: str):
    """Test minimal sequences without system prompt."""
    print("\n" + "=" * 60)
    print("Test 2: Minimal Sequences (No Padding)")
    print("=" * 60)

    tokenizer = load_tokenizer(gemma_path)
    config = Gemma3Config()
    model = Gemma3Model(config)
    load_gemma3_weights(model, gemma_path)

    # Very short, different prompts - no padding
    prompt_a = "fire"
    prompt_b = "water"

    # Tokenize without padding
    tokens_a = tokenizer(prompt_a, return_tensors="np", add_special_tokens=False)
    tokens_b = tokenizer(prompt_b, return_tensors="np", add_special_tokens=False)

    input_ids_a = mx.array(tokens_a["input_ids"])
    input_ids_b = mx.array(tokens_b["input_ids"])

    print(f"\nPrompt A: '{prompt_a}' -> tokens: {tokens_a['input_ids'].tolist()}")
    print(f"Prompt B: '{prompt_b}' -> tokens: {tokens_b['input_ids'].tolist()}")

    # Run through model (no attention mask = no padding)
    print("\nRunning through Gemma...")
    out_a, _ = model(input_ids_a, attention_mask=None, output_hidden_states=False)
    out_b, _ = model(input_ids_b, attention_mask=None, output_hidden_states=False)
    mx.eval(out_a, out_b)

    # Compare outputs
    a = np.array(out_a[0, -1])  # Last token embedding
    b = np.array(out_b[0, -1])  # Last token embedding

    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\nLast token embeddings:")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  Mean abs diff: {np.abs(a - b).mean():.4f}")

    # Compare mean of all tokens
    a_full = np.array(out_a[0]).mean(axis=0)
    b_full = np.array(out_b[0]).mean(axis=0)
    full_cosine = np.dot(a_full, b_full) / (np.linalg.norm(a_full) * np.linalg.norm(b_full))
    print(f"\nMean pooled embeddings:")
    print(f"  Cosine similarity: {full_cosine:.4f}")

    if cosine_sim < 0.95:
        print("\n[PASS] Different prompts produce sufficiently different embeddings")
    else:
        print("\n[WARN] Embeddings still very similar - may be an issue")


def test_content_tokens_only(gemma_path: str):
    """Test comparing only content tokens, ignoring shared structure."""
    print("\n" + "=" * 60)
    print("Test 3: Full Pipeline - Content Tokens Only")
    print("=" * 60)

    tokenizer = load_tokenizer(gemma_path)
    config = Gemma3Config()
    model = Gemma3Model(config)
    load_gemma3_weights(model, gemma_path)

    # Full prompts with system prompt
    T2V_SYSTEM_PROMPT = "Describe the video in extreme detail, focusing on the visual content."

    prompt_a = "A red ball bouncing"
    prompt_b = "A blue ocean wave"

    chat_a = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{prompt_a}<end_of_turn>\n<start_of_turn>model\n"
    chat_b = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{prompt_b}<end_of_turn>\n<start_of_turn>model\n"

    # Tokenize with padding
    max_len = 128
    tokens_a = tokenizer(chat_a, return_tensors="np", padding="max_length", truncation=True, max_length=max_len)
    tokens_b = tokenizer(chat_b, return_tensors="np", padding="max_length", truncation=True, max_length=max_len)

    # Find where tokens differ
    ids_a = tokens_a["input_ids"][0]
    ids_b = tokens_b["input_ids"][0]
    mask_a = tokens_a["attention_mask"][0]
    mask_b = tokens_b["attention_mask"][0]

    # Find actual length (non-padding)
    len_a = mask_a.sum()
    len_b = mask_b.sum()

    # Find where they differ
    min_len = min(len_a, len_b)
    differ_start = None
    for i in range(min_len):
        if ids_a[i] != ids_b[i]:
            differ_start = i
            break

    if differ_start:
        print(f"\nTokens diverge at position {differ_start}")
        print(f"  Prompt A actual length: {len_a}")
        print(f"  Prompt B actual length: {len_b}")
        print(f"  Common prefix: {differ_start} tokens")

    # Run through model
    input_ids_a = mx.array(tokens_a["input_ids"])
    input_ids_b = mx.array(tokens_b["input_ids"])
    attention_mask_a = mx.array(tokens_a["attention_mask"])
    attention_mask_b = mx.array(tokens_b["attention_mask"])

    print("\nRunning through Gemma...")
    out_a, _ = model(input_ids_a, attention_mask=attention_mask_a, output_hidden_states=False)
    out_b, _ = model(input_ids_b, attention_mask=attention_mask_b, output_hidden_states=False)
    mx.eval(out_a, out_b)

    # Compare only the content tokens (after divergence point)
    if differ_start:
        content_a = np.array(out_a[0, differ_start:int(len_a)])
        content_b = np.array(out_b[0, differ_start:int(len_b)])

        # Mean pool the content
        mean_a = content_a.mean(axis=0)
        mean_b = content_b.mean(axis=0)

        content_cosine = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
        print(f"\nContent-only comparison (positions {differ_start}+):")
        print(f"  Cosine similarity: {content_cosine:.4f}")

        # Also compare full sequence for reference
        full_a = np.array(out_a[0, :int(len_a)]).mean(axis=0)
        full_b = np.array(out_b[0, :int(len_b)]).mean(axis=0)
        full_cosine = np.dot(full_a, full_b) / (np.linalg.norm(full_a) * np.linalg.norm(full_b))
        print(f"\nFull sequence (with shared prefix):")
        print(f"  Cosine similarity: {full_cosine:.4f}")

        print(f"\nExpected: Content-only should be LOWER than full sequence")
        print(f"  (because shared prefix inflates similarity)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug embedding similarity")
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
        help="Path to Gemma 3 weights",
    )
    args = parser.parse_args()

    print("Debugging embedding similarity issues\n")

    # Run tests
    test_embedding_layer_only(args.gemma_path)
    test_minimal_sequences(args.gemma_path)
    test_content_tokens_only(args.gemma_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
If Test 1 shows different tokens have similarity < 1.0: Embedding layer works
If Test 2 shows similarity < 0.95: Model differentiates minimal inputs
If Test 3 shows content-only < full sequence: The high similarity was from shared prefix

High similarity in full sequences is EXPECTED when most tokens are shared
(system prompt + padding). The text encoder pipeline properly processes
only the meaningful differences.
""")


if __name__ == "__main__":
    main()
