"""Debug text embedding stages to find where differentiation is lost."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def compute_correlation(a, b):
    """Compute correlation between two flattened arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(mx.sum(a_flat * b_flat)) / (
        float(mx.sqrt(mx.sum(a_flat**2))) * float(mx.sqrt(mx.sum(b_flat**2))) + 1e-8
    )


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
    print("Embedding Stage-by-Stage Analysis")
    print("=" * 60)

    # Load models
    from scripts.generate import load_tokenizer, create_chat_prompt
    from LTX_2_MLX.model.text_encoder.gemma3 import Gemma3Model, Gemma3Config, load_gemma3_weights
    from LTX_2_MLX.model.text_encoder.encoder import create_text_encoder, load_text_encoder_weights

    print("\n[1] Loading models...")
    tokenizer = load_tokenizer(args.gemma_path)
    tokenizer.padding_side = "right"

    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, args.gemma_path, use_fp16=False)

    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, args.weights)
    mx.eval(text_encoder.parameters())

    # Test prompts
    prompt1 = "A blue ball on grass"
    prompt2 = "A red ball on grass"

    print(f"\nComparing:\n  '{prompt1}'\n  '{prompt2}'")

    results = {}
    for prompt in [prompt1, prompt2]:
        chat_prompt = create_chat_prompt(prompt)
        encoding = tokenizer(
            chat_prompt, return_tensors="np", padding="max_length",
            truncation=True, max_length=256,
        )
        input_ids = mx.array(encoding["input_ids"])
        attention_mask = mx.array(encoding["attention_mask"])

        # Get Gemma hidden states
        _, all_hidden = gemma(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        mx.eval(all_hidden[-1])

        # Store at each stage
        results[prompt] = {
            "num_tokens": int(attention_mask.sum()),
            "hidden_states": all_hidden,
            "attention_mask": attention_mask,
        }

    n1 = results[prompt1]["num_tokens"]
    n2 = results[prompt2]["num_tokens"]
    min_n = min(n1, n2)
    print(f"\nValid tokens: {n1}, {n2} (comparing first {min_n})")

    # Stage 1: Raw Gemma hidden states
    print("\n" + "=" * 60)
    print("[2] Stage 1: Raw Gemma Hidden States")
    print("=" * 60)

    # Compare each layer
    all_hidden1 = results[prompt1]["hidden_states"]
    all_hidden2 = results[prompt2]["hidden_states"]

    for layer_idx in [0, 12, 24, 36, 48]:
        h1 = all_hidden1[layer_idx][0, :min_n, :]  # Only valid tokens
        h2 = all_hidden2[layer_idx][0, :min_n, :]
        corr = compute_correlation(h1, h2)
        print(f"  Layer {layer_idx}: correlation = {corr:.6f}")

    # Stage 2: After stacking (before projection)
    print("\n" + "=" * 60)
    print("[3] Stage 2: After Stacking All Layers (before projection)")
    print("=" * 60)

    stacked1 = mx.stack(all_hidden1, axis=-1)  # [B, T, D, L]
    stacked2 = mx.stack(all_hidden2, axis=-1)

    s1 = stacked1[0, :min_n, :, :]  # [T, D, L]
    s2 = stacked2[0, :min_n, :, :]
    corr = compute_correlation(s1, s2)
    print(f"  Stacked (all 49 layers, {min_n} tokens): correlation = {corr:.6f}")

    # Reshape for projection
    b, t, d, num_layers = stacked1.shape
    concat1 = stacked1.reshape(b, t, d * num_layers)
    concat2 = stacked2.reshape(b, t, d * num_layers)

    c1 = concat1[0, :min_n, :]
    c2 = concat2[0, :min_n, :]
    corr = compute_correlation(c1, c2)
    print(f"  Concatenated (3840*49 = 188160 dims): correlation = {corr:.6f}")

    # Stage 3: After linear projection (aggregate_embed)
    print("\n" + "=" * 60)
    print("[4] Stage 3: After Linear Projection (aggregate_embed)")
    print("=" * 60)

    projected1 = text_encoder.feature_extractor.aggregate_embed(concat1)
    projected2 = text_encoder.feature_extractor.aggregate_embed(concat2)
    mx.eval(projected1, projected2)

    p1 = projected1[0, :min_n, :]
    p2 = projected2[0, :min_n, :]
    corr = compute_correlation(p1, p2)
    print(f"  After aggregate_embed (3840 dims): correlation = {corr:.6f}")

    # Check projection weight statistics
    print("\n  aggregate_embed weight statistics:")
    w = text_encoder.feature_extractor.aggregate_embed.weight
    print(f"    Shape: {w.shape}")
    print(f"    Range: [{float(mx.min(w)):.4f}, {float(mx.max(w)):.4f}]")
    print(f"    Mean: {float(mx.mean(w)):.6f}, Std: {float(mx.std(w)):.6f}")

    # Stage 4: After connector
    print("\n" + "=" * 60)
    print("[5] Stage 4: After Connector (with learnable registers)")
    print("=" * 60)

    # Full feature extraction
    encoded1 = text_encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden1,
        attention_mask=results[prompt1]["attention_mask"],
        padding_side="right",
    )
    encoded2 = text_encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden2,
        attention_mask=results[prompt2]["attention_mask"],
        padding_side="right",
    )

    # Connector
    large_value = 1e9
    mask1 = results[prompt1]["attention_mask"]
    mask2 = results[prompt2]["attention_mask"]
    conn_mask1 = (mask1.astype(encoded1.dtype) - 1) * large_value
    conn_mask1 = conn_mask1.reshape(1, 1, 1, 256)
    conn_mask2 = (mask2.astype(encoded2.dtype) - 1) * large_value
    conn_mask2 = conn_mask2.reshape(1, 1, 1, 256)

    connected1, _ = text_encoder.embeddings_connector(encoded1, conn_mask1)
    connected2, _ = text_encoder.embeddings_connector(encoded2, conn_mask2)
    mx.eval(connected1, connected2)

    co1 = connected1[0, :min_n, :]
    co2 = connected2[0, :min_n, :]
    corr = compute_correlation(co1, co2)
    print(f"  After connector (valid tokens only): correlation = {corr:.6f}")

    # All positions
    corr_all = compute_correlation(connected1[0], connected2[0])
    print(f"  After connector (all 256 positions): correlation = {corr_all:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The correlation should DECREASE at each stage if text")
    print("conditioning is working. If it stays high throughout,")
    print("the model cannot distinguish between different prompts.")


if __name__ == "__main__":
    main()
