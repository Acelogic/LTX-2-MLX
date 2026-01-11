#!/usr/bin/env python3
"""Debug text encoder to find where token diversity collapses."""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer


def analyze_tokens(name: str, tokens: mx.array):
    """Analyze token diversity."""
    # Force evaluation
    mx.eval(tokens)
    tokens_np = np.array(tokens)

    print(f"\n=== {name} ===")
    print(f"  Shape: {tokens.shape}")
    print(f"  dtype: {tokens.dtype}")

    if tokens.ndim == 3:
        # [B, T, D] format
        t = tokens_np[0]  # First batch item
    else:
        t = tokens_np

    # Basic stats
    print(f"  Mean: {np.mean(t):.6f}, Std: {np.std(t):.6f}")
    print(f"  Min: {np.min(t):.4f}, Max: {np.max(t):.4f}")

    # Token-level analysis
    num_tokens = t.shape[0]
    print(f"  Num tokens: {num_tokens}")

    # Cosine similarities
    norms = np.linalg.norm(t, axis=-1, keepdims=True)
    t_normed = t / (norms + 1e-8)

    # Sample similarities
    if num_tokens >= 2:
        cos_01 = float(np.sum(t_normed[0] * t_normed[1]))
        cos_0_mid = float(np.sum(t_normed[0] * t_normed[num_tokens//2]))
        cos_0_last = float(np.sum(t_normed[0] * t_normed[-1]))
        print(f"  Cosine similarities:")
        print(f"    cos(0, 1): {cos_01:.4f}")
        print(f"    cos(0, mid): {cos_0_mid:.4f}")
        print(f"    cos(0, last): {cos_0_last:.4f}")

    # PCA to check dimensionality
    t_centered = t - np.mean(t, axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(t_centered, full_matrices=False)
        var_explained = (S ** 2) / np.sum(S ** 2)
        print(f"  PCA variance explained:")
        print(f"    PC1: {var_explained[0]*100:.1f}%")
        if len(var_explained) > 1:
            print(f"    PC1-2: {sum(var_explained[:2])*100:.1f}%")
        if len(var_explained) > 4:
            print(f"    PC1-5: {sum(var_explained[:5])*100:.1f}%")
    except Exception as e:
        print(f"  PCA failed: {e}")

    # Cross-token variance (should be high for diverse tokens)
    token_means = np.mean(t, axis=-1)  # Mean per token
    cross_token_var = np.var(token_means)
    print(f"  Cross-token variance of means: {cross_token_var:.6f}")

    # Per-token variance (should be similar across tokens)
    token_vars = np.var(t, axis=-1)
    print(f"  Per-token variance: mean={np.mean(token_vars):.4f}, std={np.std(token_vars):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Debug text encoder")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--gemma-path", type=str, default="weights/gemma-3-12b", help="Path to Gemma weights")
    parser.add_argument("--prompt", type=str, default="A cat walking in a garden", help="Text prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("TEXT ENCODER DEBUG")
    print("=" * 70)

    # Load Gemma tokenizer
    print("\n1. Loading Gemma tokenizer and model...")
    from LTX_2_MLX.model.text_encoder.gemma3 import (
        Gemma3Config,
        Gemma3Model,
        load_gemma3_weights,
    )

    gemma_path = args.gemma_path
    print(f"   Loading tokenizer from {gemma_path}...")
    tokenizer = AutoTokenizer.from_pretrained(gemma_path)

    print("   Creating Gemma model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)

    # Load weights (FP32 to avoid numerical issues)
    print(f"   Loading Gemma weights from {gemma_path}...")
    load_gemma3_weights(gemma, gemma_path, use_fp16=False)

    # Create text encoder
    print("\n2. Creating text encoder components...")
    from LTX_2_MLX.model.text_encoder.encoder import (
        create_text_encoder,
        load_text_encoder_weights,
    )

    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, args.weights)

    # Check learnable registers
    print("\n3. Checking learnable registers...")
    registers = text_encoder.embeddings_connector.learnable_registers
    mx.eval(registers)
    analyze_tokens("Learnable Registers (128 tokens)", registers[None, :, :])

    # Check register norms individually
    print("\n   First 10 register norms:")
    for i in range(10):
        norm = float(mx.sqrt(mx.sum(registers[i] ** 2)))
        print(f"     Register {i}: norm={norm:.4f}")

    # Check if registers are similar to each other
    print("\n   Register pairwise similarities:")
    reg_norms = mx.sqrt(mx.sum(registers ** 2, axis=-1, keepdims=True))
    reg_normed = registers / (reg_norms + 1e-8)
    for i in range(0, 10, 2):
        cos = float(mx.sum(reg_normed[i] * reg_normed[i+1]))
        print(f"     cos(reg_{i}, reg_{i+1}): {cos:.4f}")

    # Tokenize prompt with chat format
    print(f"\n4. Tokenizing prompt: '{args.prompt}'")

    T2V_SYSTEM_PROMPT = "You are an assistant that generates video captions. Describe the scene in detail."
    chat_prompt = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"

    tokens_list = tokenizer.encode(chat_prompt, add_special_tokens=False)
    tokens = mx.array([tokens_list], dtype=mx.int32)
    print(f"   Raw token count: {tokens.shape[-1]}")

    # Pad to 256 tokens (left padding)
    max_len = 256
    if tokens.shape[-1] < max_len:
        pad_len = max_len - tokens.shape[-1]
        tokens = mx.concatenate([
            mx.zeros((1, pad_len), dtype=mx.int32),  # Left padding
            tokens
        ], axis=1)

    # Create attention mask
    attention_mask = (tokens != 0).astype(mx.int32)
    num_valid = int(attention_mask.sum())
    print(f"   Padded shape: {tokens.shape}, valid tokens: {num_valid}")

    # Run Gemma and collect hidden states
    print("\n5. Running Gemma forward pass...")

    _, hidden_states_list = gemma(tokens, output_hidden_states=True)
    print(f"   Collected {len(hidden_states_list)} hidden states")
    mx.eval(*hidden_states_list)

    analyze_tokens("Gemma Hidden State (layer 0)", hidden_states_list[0])
    analyze_tokens("Gemma Hidden State (layer 24)", hidden_states_list[24])
    analyze_tokens("Gemma Hidden State (layer 48)", hidden_states_list[48])

    # Check only valid tokens in the last hidden state
    analyze_tokens("Gemma layer 48 (valid tokens only)",
                   hidden_states_list[48][:, -num_valid:, :])

    # Step 1: Feature Extractor
    print("\n6. Feature Extractor...")

    # Stack hidden states
    stacked = mx.stack(hidden_states_list, axis=-1)
    mx.eval(stacked)
    print(f"   Stacked shape: {stacked.shape}")  # [B, T, 3840, 49]

    # Normalize and concatenate
    from LTX_2_MLX.model.text_encoder.feature_extractor import norm_and_concat_padded_batch

    sequence_lengths = attention_mask.sum(axis=-1).astype(mx.int32)
    normed = norm_and_concat_padded_batch(stacked, sequence_lengths, padding_side="left")
    mx.eval(normed)
    print(f"   After norm_and_concat: {normed.shape}")
    analyze_tokens("After norm_and_concat (valid tokens only)",
                   normed[0, -num_valid:, :][None, :, :])
    analyze_tokens("After norm_and_concat (ALL 256 tokens)", normed)

    # Apply aggregate_embed projection
    projected = text_encoder.feature_extractor.aggregate_embed(normed)
    mx.eval(projected)
    print(f"   After aggregate_embed: {projected.shape}")
    analyze_tokens("After aggregate_embed (valid tokens)",
                   projected[0, -num_valid:, :][None, :, :])
    analyze_tokens("After aggregate_embed (ALL 256 tokens)", projected)

    # Step 2: Embeddings Connector
    print("\n7. Embeddings Connector...")

    # Convert attention mask to additive format
    additive_mask = text_encoder._convert_to_additive_mask(attention_mask, projected.dtype)
    print(f"   Additive mask shape: {additive_mask.shape}")

    # Apply register replacement
    hidden_with_registers, new_mask = text_encoder.embeddings_connector._replace_padded_with_learnable_registers(
        projected, additive_mask
    )
    mx.eval(hidden_with_registers)
    print(f"   After register replacement: {hidden_with_registers.shape}")
    analyze_tokens("After register replacement", hidden_with_registers)

    # Check what the first few and last few tokens look like
    print("\n   Checking token sources (first 5, last 5):")
    for i in list(range(5)) + list(range(251, 256)):
        token = hidden_with_registers[0, i]
        token_norm = float(mx.sqrt(mx.sum(token ** 2)))

        # Check if matches a register
        matched_reg = None
        for r in range(128):
            reg = text_encoder.embeddings_connector.learnable_registers[r]
            diff = float(mx.sqrt(mx.sum((token - reg) ** 2)))
            if diff < 0.01:
                matched_reg = r
                break

        # Check if matches projected token
        proj_token = projected[0, i]
        proj_diff = float(mx.sqrt(mx.sum((token - proj_token) ** 2)))

        if matched_reg is not None:
            print(f"     Token {i}: register {matched_reg} (norm={token_norm:.4f})")
        elif proj_diff < 0.01:
            print(f"     Token {i}: original projection (norm={token_norm:.4f})")
        else:
            print(f"     Token {i}: unknown source (norm={token_norm:.4f})")

    # Run through transformer blocks
    from LTX_2_MLX.model.transformer.rope import precompute_freqs_cis

    seq_len = hidden_with_registers.shape[1]
    indices_grid = mx.arange(seq_len, dtype=mx.float32)[None, None, :]

    freqs_cis = precompute_freqs_cis(
        indices_grid=indices_grid,
        dim=text_encoder.embeddings_connector.inner_dim,
        out_dtype=hidden_with_registers.dtype,
        theta=text_encoder.embeddings_connector.positional_embedding_theta,
        max_pos=text_encoder.embeddings_connector.positional_embedding_max_pos,
        num_attention_heads=text_encoder.embeddings_connector.num_attention_heads,
        rope_type=text_encoder.embeddings_connector.rope_type,
    )

    x = hidden_with_registers
    for i, block in enumerate(text_encoder.embeddings_connector.transformer_1d_blocks):
        x = block(x, attention_mask=new_mask, pe=freqs_cis)
        mx.eval(x)
        analyze_tokens(f"After transformer block {i}", x)

    # Final RMSNorm
    from LTX_2_MLX.model.transformer.attention import rms_norm
    x = rms_norm(x, eps=text_encoder.embeddings_connector.norm_eps)
    mx.eval(x)
    analyze_tokens("After final RMSNorm", x)

    # Step 3: Caption Projection
    print("\n8. Caption Projection...")

    caption_out = text_encoder.caption_projection(x)
    mx.eval(caption_out)
    analyze_tokens("Final text encoder output", caption_out)

    # Compare with null encoding
    print("\n9. Comparing with null (empty prompt) encoding...")
    null_tokens = mx.zeros((1, 256), dtype=mx.int32)
    null_mask = mx.zeros((1, 256), dtype=mx.int32)

    # Run full encoder on null
    _, null_hidden_states = gemma(null_tokens, output_hidden_states=True)
    mx.eval(*null_hidden_states)

    null_output = text_encoder.encode_from_hidden_states(
        null_hidden_states, null_mask, padding_side="left"
    )
    mx.eval(null_output.video_encoding)

    # Compare
    diff = float(mx.mean(mx.abs(caption_out - null_output.video_encoding)))
    print(f"   Mean absolute difference from null: {diff:.6f}")

    # Cosine similarity between corresponding tokens
    cap_norm = caption_out / (mx.sqrt(mx.sum(caption_out ** 2, axis=-1, keepdims=True)) + 1e-8)
    null_norm = null_output.video_encoding / (mx.sqrt(mx.sum(null_output.video_encoding ** 2, axis=-1, keepdims=True)) + 1e-8)
    cos_sim = mx.mean(mx.sum(cap_norm * null_norm, axis=-1))
    print(f"   Mean cosine similarity with null: {float(cos_sim):.6f}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # Get final cosine similarity between tokens
    final_t = np.array(caption_out[0])
    final_norms = np.linalg.norm(final_t, axis=-1, keepdims=True)
    final_normed = final_t / (final_norms + 1e-8)

    avg_cos = 0
    count = 0
    for i in range(10):
        for j in range(i+1, 10):
            avg_cos += float(np.sum(final_normed[i] * final_normed[j]))
            count += 1
    avg_cos /= count

    print(f"\n  Average cosine similarity (first 10 tokens): {avg_cos:.4f}")

    if avg_cos > 0.95:
        print("\n  PROBLEM: Token diversity is collapsed!")
        print("  Tokens are nearly identical, causing uniform cross-attention.")
        print("\n  Look at the output above to find where diversity drops:")
        print("    - If learnable registers are similar → weights not loaded")
        print("    - If transformer blocks increase similarity → attention problem")
        print("    - If aggregate_embed collapses → projection weights problem")
    else:
        print("\n  Token diversity looks OK.")


if __name__ == "__main__":
    main()
