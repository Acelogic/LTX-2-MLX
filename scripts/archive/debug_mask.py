"""Debug text encoder mask handling."""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


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
    print("Text Encoder Mask Debugging")
    print("=" * 60)

    # Check if registers are in the weights
    print("\n[1] Checking learnable registers in weights...")
    from safetensors import safe_open
    import torch

    with safe_open(args.weights, framework="pt") as f:
        reg_key = "model.diffusion_model.video_embeddings_connector.learnable_registers"
        if reg_key in f.keys():
            tensor = f.get_tensor(reg_key)
            print(f"  Found learnable_registers: shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"  Range: [{float(tensor.min()):.4f}, {float(tensor.max()):.4f}]")
            print(f"  Mean: {float(tensor.mean()):.4f}, Std: {float(tensor.std()):.4f}")
        else:
            print(f"  WARNING: {reg_key} not found in weights!")
            print("  Available connector keys:")
            for k in f.keys():
                if "embeddings_connector" in k:
                    print(f"    {k}")

    # Load text encoder
    print("\n[2] Loading text encoder...")
    from LTX_2_MLX.model.text_encoder.encoder import create_text_encoder, load_text_encoder_weights

    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, args.weights)
    mx.eval(text_encoder.parameters())

    # Check loaded registers
    print("\n[3] Checking loaded registers...")
    regs = text_encoder.embeddings_connector.learnable_registers
    print(f"  Shape: {regs.shape}")
    print(f"  Range: [{float(mx.min(regs)):.4f}, {float(mx.max(regs)):.4f}]")
    print(f"  Mean: {float(mx.mean(regs)):.4f}, Std: {float(mx.std(regs)):.4f}")

    # Test encoding with mask analysis
    import os
    if not os.path.exists(args.gemma_path):
        print(f"\nERROR: Gemma weights not found at {args.gemma_path}")
        return

    print("\n[4] Testing encoding with mask analysis...")

    from scripts.generate import load_tokenizer, create_chat_prompt
    from LTX_2_MLX.model.text_encoder.gemma3 import Gemma3Model, Gemma3Config, load_gemma3_weights

    tokenizer = load_tokenizer(args.gemma_path)
    tokenizer.padding_side = "right"

    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, args.gemma_path, use_fp16=False)

    # Encode two different prompts
    prompts = ["A blue ball", "A red ball"]

    for prompt in prompts:
        print(f"\n--- Encoding: '{prompt}' ---")

        chat_prompt = create_chat_prompt(prompt)
        encoding = tokenizer(
            chat_prompt,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        input_ids = mx.array(encoding["input_ids"])
        attention_mask = mx.array(encoding["attention_mask"])

        num_tokens = int(attention_mask.sum())
        print(f"  Token count: {num_tokens}/256")
        print(f"  Original mask sum: {float(mx.sum(attention_mask))}")

        # Get hidden states
        _, all_hidden = gemma(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        mx.eval(all_hidden[-1])

        # Feature extraction
        encoded = text_encoder.feature_extractor.extract_from_hidden_states(
            hidden_states=all_hidden,
            attention_mask=attention_mask,
            padding_side="right",
        )
        mx.eval(encoded)

        print(f"  After feature extraction:")
        print(f"    Shape: {encoded.shape}")
        print(f"    Range: [{float(mx.min(encoded)):.4f}, {float(mx.max(encoded)):.4f}]")

        # Convert mask for connector
        large_value = 1e9
        connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
        connector_mask = connector_mask.reshape(1, 1, 1, 256)

        # Process through connector
        encoded_out, output_mask = text_encoder.embeddings_connector(encoded, connector_mask)
        mx.eval(encoded_out)
        mx.eval(output_mask)

        print(f"  After connector:")
        print(f"    Shape: {encoded_out.shape}")
        print(f"    Output mask unique values: {np.unique(np.array(output_mask))}")

        # Binary mask from connector output
        binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)
        print(f"    Binary mask sum: {float(mx.sum(binary_mask))} (should be 256 if all ones)")

        # What we SHOULD use: original mask
        print(f"    Original mask sum: {num_tokens}")

    # Test correlation with different masks
    print("\n[5] Testing correlation with original vs connector mask...")

    embeddings = {}
    for prompt in prompts:
        chat_prompt = create_chat_prompt(prompt)
        encoding = tokenizer(
            chat_prompt, return_tensors="np", padding="max_length",
            truncation=True, max_length=256,
        )
        input_ids = mx.array(encoding["input_ids"])
        attention_mask = mx.array(encoding["attention_mask"])

        _, all_hidden = gemma(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded = text_encoder.feature_extractor.extract_from_hidden_states(
            hidden_states=all_hidden, attention_mask=attention_mask, padding_side="right",
        )

        connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
        connector_mask = connector_mask.reshape(1, 1, 1, 256)

        encoded_out, _ = text_encoder.embeddings_connector(encoded, connector_mask)
        mx.eval(encoded_out)

        # Store both original mask and full embedding
        embeddings[prompt] = {
            "embedding": encoded_out,
            "original_mask": attention_mask,
            "num_tokens": int(attention_mask.sum()),
        }

    e1 = embeddings[prompts[0]]
    e2 = embeddings[prompts[1]]

    # Correlation with full 256 positions (current behavior)
    flat1 = e1["embedding"].flatten()
    flat2 = e2["embedding"].flatten()
    corr_full = float(mx.sum(flat1 * flat2)) / (
        float(mx.sqrt(mx.sum(flat1**2))) * float(mx.sqrt(mx.sum(flat2**2))) + 1e-8
    )
    print(f"  Correlation (all 256 positions): {corr_full:.6f}")

    # Correlation with only valid tokens (using original mask)
    n1 = e1["num_tokens"]
    n2 = e2["num_tokens"]
    min_tokens = min(n1, n2)

    valid1 = e1["embedding"][0, :n1, :].flatten()
    valid2 = e2["embedding"][0, :n2, :].flatten()

    # Only compare first min_tokens positions
    v1 = e1["embedding"][0, :min_tokens, :].flatten()
    v2 = e2["embedding"][0, :min_tokens, :].flatten()
    corr_valid = float(mx.sum(v1 * v2)) / (
        float(mx.sqrt(mx.sum(v1**2))) * float(mx.sqrt(mx.sum(v2**2))) + 1e-8
    )
    print(f"  Correlation (first {min_tokens} valid positions): {corr_valid:.6f}")

    print("\n[6] Conclusion:")
    if corr_full > corr_valid + 0.01:
        print("  >> Registers are INCREASING correlation - masking them could help!")
    elif corr_full < corr_valid - 0.01:
        print("  >> Registers are DECREASING correlation - keep using them")
    else:
        print("  >> Registers have minimal effect on correlation")


if __name__ == "__main__":
    main()
