#!/usr/bin/env python3
"""
Pre-compute text embeddings using PyTorch and the original LTX-2 text encoder.

This script uses the official LTX-2 PyTorch implementation to encode prompts,
then saves them for use with the MLX generation pipeline.

Requirements:
    pip install torch transformers safetensors

Usage:
    python scripts/encode_with_pytorch.py "Your prompt here" \
        --gemma-path weights/gemma-3-12b \
        --weights weights/ltx-2/ltx-2-19b-distilled.safetensors \
        --output prompt_embedding.npz

    python scripts/generate.py --embedding prompt_embedding.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def encode_prompt_pytorch(
    prompt: str,
    gemma_path: str,
    weights_path: str,
    max_tokens: int = 256,
    device: str = "mps",
) -> tuple:
    """
    Encode prompt using PyTorch LTX-2 text encoder.

    Returns:
        Tuple of (embedding [1, seq, 4096], attention_mask [1, seq]).
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    # Add LTX-2 to path
    ltx2_path = Path(__file__).parent.parent / "LTX-2"
    if ltx2_path.exists():
        sys.path.insert(0, str(ltx2_path / "packages" / "ltx-core" / "src"))
        sys.path.insert(0, str(ltx2_path / "packages" / "ltx-pipelines" / "src"))

    try:
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.text_encoders.gemma.encoders.av_encoder import (
            AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            AVGemmaTextEncoderModelConfigurator,
        )
        from ltx_core.text_encoders.gemma.encoders.base_encoder import (
            module_ops_from_gemma_root,
        )
    except ImportError as e:
        print(f"Error: Could not import LTX-2 modules: {e}")
        print("Make sure LTX-2 submodule is initialized:")
        print("  git submodule update --init --recursive")
        sys.exit(1)

    print(f"Loading Gemma from {gemma_path}...")
    print(f"Loading LTX-2 weights from {weights_path}...")

    # Build text encoder
    key_ops = module_ops_from_gemma_root(gemma_path, torch.bfloat16)
    key_ops.update(AV_GEMMA_TEXT_ENCODER_KEY_OPS)

    builder = SingleGPUModelBuilder(
        model_path=weights_path,
        dtype=torch.bfloat16,
        key_ops=key_ops,
        model_configurator=AVGemmaTextEncoderModelConfigurator(gemma_path),
    )

    text_encoder = builder.build(device=torch.device(device), dtype=torch.bfloat16)
    print("Text encoder loaded!")

    # Encode prompt
    print(f"Encoding: {prompt}")

    with torch.inference_mode():
        output = text_encoder.encode([prompt], max_sequence_length=max_tokens)

    # Get video encoding (already projected to 4096 dim)
    embedding = output.video_encoding.cpu().float().numpy()
    mask = output.video_attention_mask.cpu().numpy()

    print(f"Embedding shape: {embedding.shape}")

    return embedding, mask


def main():
    parser = argparse.ArgumentParser(
        description="Encode prompts using PyTorch LTX-2 text encoder"
    )
    parser.add_argument("prompt", type=str, help="Text prompt to encode")
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma",
        help="Path to Gemma model directory",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to LTX-2 weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prompt_embedding.npz",
        help="Output path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="PyTorch device (mps, cuda, cpu)",
    )

    args = parser.parse_args()

    embedding, mask = encode_prompt_pytorch(
        args.prompt,
        args.gemma_path,
        args.weights,
        args.max_tokens,
        args.device,
    )

    print(f"Saving to {args.output}...")
    np.savez(
        args.output,
        embedding=embedding,
        attention_mask=mask,
        prompt=args.prompt,
    )
    print("Done!")
    print(f"\nUse with: python scripts/generate.py --embedding {args.output}")


if __name__ == "__main__":
    main()
