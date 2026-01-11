"""Compare text embeddings between MLX and PyTorch implementations.

This script helps verify that our MLX text encoding pipeline produces
the same embeddings as the official PyTorch implementation.

Usage:
    # Step 1: Generate MLX embeddings
    python scripts/compare_text_embeddings.py --mode mlx --prompt "A blue ball on grass"

    # Step 2: Generate PyTorch embeddings (requires official LTX-2 repo)
    python scripts/compare_text_embeddings.py --mode pytorch --prompt "A blue ball on grass"

    # Step 3: Compare the two
    python scripts/compare_text_embeddings.py --mode compare

    # Or compare two prompts within MLX to check differentiation
    python scripts/compare_text_embeddings.py --mode diff --prompt "A blue ball" --prompt2 "A red ball"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def compute_stats(arr: np.ndarray, name: str) -> dict:
    """Compute statistics for an array."""
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
    }


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute correlation between two arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    if len(a_flat) != len(b_flat):
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def compute_max_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Compute max absolute difference."""
    return float(np.abs(a - b).max())


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error."""
    return float(np.mean((a - b) ** 2))


def generate_mlx_embeddings(
    prompt: str,
    weights_path: str,
    gemma_path: str,
    output_path: str,
    use_early_layers: bool = False,
):
    """Generate text embeddings using MLX and save to file."""
    import mlx.core as mx
    from scripts.generate import load_tokenizer, create_chat_prompt
    from LTX_2_MLX.model.text_encoder.gemma3 import (
        Gemma3Model,
        Gemma3Config,
        load_gemma3_weights,
    )
    from LTX_2_MLX.model.text_encoder.encoder import (
        create_text_encoder,
        load_text_encoder_weights,
    )

    print("=" * 60)
    print("Generating MLX Text Embeddings")
    print("=" * 60)
    print(f"Prompt: {prompt}")

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = load_tokenizer(gemma_path)
    tokenizer.padding_side = "right"

    # Tokenize
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

    # Load Gemma
    print("\n[2/4] Loading Gemma 3...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, gemma_path, use_fp16=False)

    # Run Gemma forward pass
    print("\n[3/4] Running Gemma forward pass...")
    _, all_hidden_states = gemma(
        input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    mx.eval(all_hidden_states[-1])
    print(f"  Got {len(all_hidden_states)} hidden states")

    # Convert to numpy for saving
    hidden_states_np = [np.array(h) for h in all_hidden_states]

    # Load text encoder
    print("\n[4/4] Running text encoder pipeline...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, weights_path)
    mx.eval(text_encoder.parameters())

    # Feature extraction
    encoded = text_encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=attention_mask,
        padding_side="right",
    )
    mx.eval(encoded)
    feature_extractor_output = np.array(encoded)

    # Connector
    large_value = 1e9
    connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
    connector_mask = connector_mask.reshape(1, 1, 1, 256)

    final_embedding, output_mask = text_encoder.embeddings_connector(
        encoded, connector_mask
    )
    mx.eval(final_embedding)
    final_embedding_np = np.array(final_embedding)

    # Early layers mode (for comparison)
    if use_early_layers:
        # Use only layer 0
        early_embedding = all_hidden_states[0]
        mx.eval(early_embedding)
        early_embedding_np = np.array(early_embedding)
    else:
        early_embedding_np = None

    # Save all intermediate results
    print(f"\nSaving embeddings to {output_path}...")
    results = {
        "prompt": prompt,
        "chat_prompt": chat_prompt,
        "num_tokens": num_tokens,
        "input_ids": np.array(input_ids).tolist(),
        "attention_mask": np.array(attention_mask).tolist(),
    }

    np.savez(
        output_path,
        # Metadata
        prompt=prompt,
        num_tokens=num_tokens,
        # Gemma outputs (save key layers only to reduce size)
        gemma_layer_0=hidden_states_np[0],
        gemma_layer_12=hidden_states_np[12],
        gemma_layer_24=hidden_states_np[24],
        gemma_layer_36=hidden_states_np[36],
        gemma_layer_48=hidden_states_np[48],
        # Feature extractor output
        feature_extractor_output=feature_extractor_output,
        # Final embedding
        final_embedding=final_embedding_np,
        # Early layers (optional)
        early_embedding=early_embedding_np if early_embedding_np is not None else np.array([]),
    )

    # Print statistics
    print("\n" + "=" * 60)
    print("Embedding Statistics")
    print("=" * 60)

    stats = [
        compute_stats(hidden_states_np[0], "Gemma Layer 0"),
        compute_stats(hidden_states_np[48], "Gemma Layer 48"),
        compute_stats(feature_extractor_output, "Feature Extractor Output"),
        compute_stats(final_embedding_np, "Final Embedding"),
    ]

    for s in stats:
        print(f"\n{s['name']}:")
        print(f"  Shape: {s['shape']}")
        print(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"  Mean: {s['mean']:.4f}, Std: {s['std']:.4f}")
        if s["has_nan"] or s["has_inf"]:
            print(f"  WARNING: has_nan={s['has_nan']}, has_inf={s['has_inf']}")

    print(f"\nSaved to: {output_path}")
    return output_path


def generate_pytorch_embeddings(
    prompt: str,
    weights_path: str,
    gemma_path: str,
    output_path: str,
):
    """Generate text embeddings using PyTorch (requires official LTX-2 installation)."""
    print("=" * 60)
    print("Generating PyTorch Text Embeddings")
    print("=" * 60)
    print(f"Prompt: {prompt}")

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: torch and transformers required for PyTorch mode")
        return None

    # Check if official LTX-2 is available
    try:
        from ltx_core.models.text_encoder import TextEncoder
        has_ltx = True
    except ImportError:
        has_ltx = False
        print("\nWARNING: Official LTX-2 package (ltx_core) not installed.")
        print("To install: pip install git+https://github.com/Lightricks/LTX-2.git")
        print("\nGenerating reference embeddings manually instead...")

    if not has_ltx:
        # Generate reference using transformers directly
        print("\n[1/3] Loading Gemma 3 with transformers...")
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            gemma_path,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(gemma_path)
        tokenizer.padding_side = "right"

        # Create chat prompt (same format as MLX generate.py)
        from scripts.generate import create_chat_prompt
        chat_prompt = create_chat_prompt(prompt)

        print("\n[2/3] Tokenizing and running forward pass...")
        inputs = tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
        print(f"  Got {len(hidden_states)} hidden states")

        # Save key layers
        print(f"\n[3/3] Saving to {output_path}...")
        np.savez(
            output_path,
            prompt=prompt,
            num_tokens=int(inputs["attention_mask"].sum()),
            gemma_layer_0=hidden_states[0].numpy(),
            gemma_layer_12=hidden_states[12].numpy() if len(hidden_states) > 12 else np.array([]),
            gemma_layer_24=hidden_states[24].numpy() if len(hidden_states) > 24 else np.array([]),
            gemma_layer_36=hidden_states[36].numpy() if len(hidden_states) > 36 else np.array([]),
            gemma_layer_48=hidden_states[48].numpy() if len(hidden_states) > 48 else np.array([]),
            # Note: feature_extractor and final_embedding would need official LTX-2
            feature_extractor_output=np.array([]),
            final_embedding=np.array([]),
        )

        print(f"\nSaved Gemma hidden states to: {output_path}")
        print("NOTE: feature_extractor_output and final_embedding require official LTX-2 package")
        return output_path

    else:
        # Use official LTX-2 implementation
        print("\nUsing official LTX-2 text encoder...")
        # TODO: Implement when official package is available
        print("ERROR: Official LTX-2 integration not yet implemented")
        return None


def compare_embeddings(mlx_path: str, pytorch_path: str):
    """Compare MLX and PyTorch embeddings."""
    print("=" * 60)
    print("Comparing MLX vs PyTorch Embeddings")
    print("=" * 60)

    # Load embeddings
    mlx_data = np.load(mlx_path, allow_pickle=True)
    pytorch_data = np.load(pytorch_path, allow_pickle=True)

    print(f"\nMLX prompt: {mlx_data['prompt']}")
    print(f"PyTorch prompt: {pytorch_data['prompt']}")

    if str(mlx_data['prompt']) != str(pytorch_data['prompt']):
        print("WARNING: Prompts don't match!")

    # Compare at each stage
    stages = [
        ("gemma_layer_0", "Gemma Layer 0 (Input Embeddings)"),
        ("gemma_layer_12", "Gemma Layer 12"),
        ("gemma_layer_24", "Gemma Layer 24"),
        ("gemma_layer_36", "Gemma Layer 36"),
        ("gemma_layer_48", "Gemma Layer 48 (Final)"),
        ("feature_extractor_output", "Feature Extractor Output"),
        ("final_embedding", "Final Embedding"),
    ]

    print("\n" + "-" * 60)
    print(f"{'Stage':<35} {'Correlation':>12} {'Max Diff':>12} {'MSE':>12}")
    print("-" * 60)

    results = []
    for key, name in stages:
        mlx_arr = mlx_data[key]
        pt_arr = pytorch_data[key]

        if mlx_arr.size == 0 or pt_arr.size == 0:
            print(f"{name:<35} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
            continue

        if mlx_arr.shape != pt_arr.shape:
            print(f"{name:<35} Shape mismatch: {mlx_arr.shape} vs {pt_arr.shape}")
            continue

        corr = compute_correlation(mlx_arr, pt_arr)
        max_diff = compute_max_diff(mlx_arr, pt_arr)
        mse = compute_mse(mlx_arr, pt_arr)

        status = "OK" if corr > 0.999 else "WARN" if corr > 0.99 else "FAIL"
        print(f"{name:<35} {corr:>11.6f} {max_diff:>11.4f} {mse:>11.6f}  [{status}]")

        results.append({
            "stage": name,
            "correlation": corr,
            "max_diff": max_diff,
            "mse": mse,
            "status": status,
        })

    print("-" * 60)

    # Summary
    print("\nSummary:")
    ok_count = sum(1 for r in results if r.get("status") == "OK")
    warn_count = sum(1 for r in results if r.get("status") == "WARN")
    fail_count = sum(1 for r in results if r.get("status") == "FAIL")

    print(f"  OK (corr > 0.999): {ok_count}")
    print(f"  WARN (0.99 < corr < 0.999): {warn_count}")
    print(f"  FAIL (corr < 0.99): {fail_count}")

    if fail_count > 0:
        print("\nAction needed: Check stages marked FAIL for implementation bugs")
    elif warn_count > 0:
        print("\nMinor differences detected - may be acceptable numerical precision differences")
    else:
        print("\nAll stages match! Text encoding pipeline is working correctly.")

    return results


def compare_two_prompts(
    prompt1: str,
    prompt2: str,
    weights_path: str,
    gemma_path: str,
):
    """Compare embeddings for two different prompts (within MLX)."""
    import mlx.core as mx
    from scripts.generate import load_tokenizer, create_chat_prompt
    from LTX_2_MLX.model.text_encoder.gemma3 import (
        Gemma3Model,
        Gemma3Config,
        load_gemma3_weights,
    )
    from LTX_2_MLX.model.text_encoder.encoder import (
        create_text_encoder,
        load_text_encoder_weights,
    )

    print("=" * 60)
    print("Comparing Two Prompts (Text Differentiation Check)")
    print("=" * 60)
    print(f"Prompt 1: {prompt1}")
    print(f"Prompt 2: {prompt2}")

    # Load models
    print("\nLoading models...")
    tokenizer = load_tokenizer(gemma_path)
    tokenizer.padding_side = "right"

    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, gemma_path, use_fp16=False)

    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, weights_path)
    mx.eval(text_encoder.parameters())

    # Process both prompts
    results = {}
    for prompt in [prompt1, prompt2]:
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

        _, all_hidden_states = gemma(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        mx.eval(all_hidden_states[-1])

        encoded = text_encoder.feature_extractor.extract_from_hidden_states(
            hidden_states=all_hidden_states,
            attention_mask=attention_mask,
            padding_side="right",
        )
        mx.eval(encoded)

        large_value = 1e9
        connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
        connector_mask = connector_mask.reshape(1, 1, 1, 256)

        final_embedding, _ = text_encoder.embeddings_connector(encoded, connector_mask)
        mx.eval(final_embedding)

        results[prompt] = {
            "layer_0": np.array(all_hidden_states[0]),
            "layer_48": np.array(all_hidden_states[48]),
            "feature_extractor": np.array(encoded),
            "final_embedding": np.array(final_embedding),
            "num_tokens": int(attention_mask.sum()),
        }

    # Compare
    print("\n" + "-" * 60)
    print(f"{'Stage':<35} {'Correlation':>12} {'Assessment':>15}")
    print("-" * 60)

    stages = [
        ("layer_0", "Gemma Layer 0 (Input)"),
        ("layer_48", "Gemma Layer 48 (Final)"),
        ("feature_extractor", "Feature Extractor Output"),
        ("final_embedding", "Final Embedding"),
    ]

    for key, name in stages:
        arr1 = results[prompt1][key]
        arr2 = results[prompt2][key]

        # Compare only valid tokens (use min of both)
        n1 = results[prompt1]["num_tokens"]
        n2 = results[prompt2]["num_tokens"]
        n_min = min(n1, n2)

        # For layer outputs, compare valid token region
        if key in ["layer_0", "layer_48"]:
            arr1_valid = arr1[0, :n_min, :]
            arr2_valid = arr2[0, :n_min, :]
        else:
            arr1_valid = arr1[0, :n_min, :]
            arr2_valid = arr2[0, :n_min, :]

        corr = compute_correlation(arr1_valid, arr2_valid)

        if corr < 0.9:
            assessment = "DIFFERENTIATED"
        elif corr < 0.99:
            assessment = "SOME DIFF"
        else:
            assessment = "TOO SIMILAR"

        print(f"{name:<35} {corr:>11.6f} {assessment:>15}")

    print("-" * 60)

    # Interpretation
    final_corr = compute_correlation(
        results[prompt1]["final_embedding"],
        results[prompt2]["final_embedding"],
    )

    print(f"\nFinal embedding correlation: {final_corr:.6f}")
    if final_corr > 0.99:
        print("WARNING: Embeddings are nearly identical!")
        print("The model will produce similar outputs for both prompts.")
        print("Consider using --early-layers-only flag for better differentiation.")
    elif final_corr > 0.9:
        print("Some differentiation exists, but embeddings are still quite similar.")
    else:
        print("Good differentiation between prompts.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare text embeddings between MLX and PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["mlx", "pytorch", "compare", "diff"],
        required=True,
        help="Mode: 'mlx' to generate MLX embeddings, 'pytorch' to generate PyTorch embeddings, "
             "'compare' to compare saved embeddings, 'diff' to compare two prompts in MLX",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A blue ball bouncing on green grass",
        help="Text prompt to encode",
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        default="A red ball bouncing on green grass",
        help="Second prompt for diff mode",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to LTX-2 weights",
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
        help="Path to Gemma 3 weights",
    )
    parser.add_argument(
        "--mlx-output",
        type=str,
        default="mlx_embeddings.npz",
        help="Output path for MLX embeddings",
    )
    parser.add_argument(
        "--pytorch-output",
        type=str,
        default="pytorch_embeddings.npz",
        help="Output path for PyTorch embeddings",
    )
    parser.add_argument(
        "--early-layers",
        action="store_true",
        help="Also save early layers embedding",
    )

    args = parser.parse_args()

    if args.mode == "mlx":
        generate_mlx_embeddings(
            prompt=args.prompt,
            weights_path=args.weights,
            gemma_path=args.gemma_path,
            output_path=args.mlx_output,
            use_early_layers=args.early_layers,
        )

    elif args.mode == "pytorch":
        generate_pytorch_embeddings(
            prompt=args.prompt,
            weights_path=args.weights,
            gemma_path=args.gemma_path,
            output_path=args.pytorch_output,
        )

    elif args.mode == "compare":
        compare_embeddings(args.mlx_output, args.pytorch_output)

    elif args.mode == "diff":
        compare_two_prompts(
            prompt1=args.prompt,
            prompt2=args.prompt2,
            weights_path=args.weights,
            gemma_path=args.gemma_path,
        )


if __name__ == "__main__":
    main()
