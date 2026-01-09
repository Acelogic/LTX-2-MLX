#!/usr/bin/env python3
"""
Download Gemma 3 12B safetensors without using the HuggingFace CLI.

This script uses the huggingface_hub library directly.

Requirements:
    pip install huggingface_hub

Usage:
    # First, accept the license at: https://huggingface.co/google/gemma-3-12b-it
    # Then run:
    python scripts/download_gemma.py --token YOUR_HF_TOKEN

    # Or set environment variable:
    export HF_TOKEN=your_token_here
    python scripts/download_gemma.py
"""

import argparse
import os
from pathlib import Path


def download_gemma(token: str, output_dir: str = "weights/gemma-3-12b"):
    """Download Gemma 3 12B safetensors."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required.")
        print("Install with: pip install huggingface_hub")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "google/gemma-3-12b-it"

    print(f"Downloading Gemma 3 12B to {output_dir}...")
    print(f"Repository: {repo_id}")
    print()
    print("Note: You must accept the license at:")
    print(f"  https://huggingface.co/{repo_id}")
    print()

    try:
        # Download the full model
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            token=token,
            allow_patterns=[
                "*.safetensors",
                "config.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
            ignore_patterns=[
                "*.gguf",
                "*.bin",
                "*.pt",
            ],
        )
        print(f"\nDownload complete: {local_dir}")
        return True

    except Exception as e:
        print(f"\nError downloading: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've accepted the license at:")
        print(f"   https://huggingface.co/{repo_id}")
        print("2. Verify your token is valid at:")
        print("   https://huggingface.co/settings/tokens")
        print("3. Ensure you have enough disk space (~25GB)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Gemma 3 12B for LTX-2 text encoding"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/gemma-3-12b",
        help="Output directory",
    )

    args = parser.parse_args()

    if not args.token:
        print("Error: HuggingFace token required.")
        print()
        print("Get a token at: https://huggingface.co/settings/tokens")
        print()
        print("Then either:")
        print("  python scripts/download_gemma.py --token YOUR_TOKEN")
        print("Or:")
        print("  export HF_TOKEN=your_token")
        print("  python scripts/download_gemma.py")
        return

    download_gemma(args.token, args.output)


if __name__ == "__main__":
    main()
