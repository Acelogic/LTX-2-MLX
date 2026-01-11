#!/usr/bin/env python3
"""
Script to download Gemma 3 12B weights from Hugging Face.
Usage:
    python scripts/download_gemma.py --token YOUR_HF_TOKEN
    
    # Or with env var:
    export HF_TOKEN=your_token
    python scripts/download_gemma.py
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login

def main():
    parser = argparse.ArgumentParser(description="Download Gemma 3 12B weights")
    parser.add_argument("--token", type=str, help="Hugging Face token (optional if already logged in)")
    parser.add_argument("--repo-id", type=str, default="google/gemma-3-12b-it", help="Hugging Face repo ID")
    parser.add_argument("--output-dir", type=str, default="weights/gemma-3-12b", help="Output directory")
    args = parser.parse_args()

    # Handle token
    token = args.token or os.environ.get("HF_TOKEN")
    
    if token:
        print(f"Logging in to Hugging Face...")
        login(token=token)
    
    print(f"Downloading {args.repo_id} to {args.output_dir}...")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "original/*"],
        )
        print(f"\nSuccess! Weights downloaded to {args.output_dir}")
        print("\nYou can now run generation with:")
        print(f"  --gemma-path {args.output_dir}")
        
    except Exception as e:
        print(f"\nError downloading weights: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have accepted the Gemma 3 license at: https://huggingface.co/google/gemma-3-12b-it")
        print("2. Make sure your token has read access")
        print("3. Try running: pip install --upgrade huggingface_hub")
        sys.exit(1)

if __name__ == "__main__":
    main()
