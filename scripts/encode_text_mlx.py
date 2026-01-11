#!/usr/bin/env python3
"""
Standalone script to encode text prompts using Gemma 3 for LTX-2.
Useful for pre-computing embeddings to save memory/time during generation.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import mlx.core as mx

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.text_encoder.gemma3 import Gemma3Model, load_gemma3_weights
from LTX_2_MLX.model.text_encoder.encoder import load_text_encoder_weights, create_text_encoder
from scripts.generate import load_tokenizer, encode_with_gemma

def main():
    parser = argparse.ArgumentParser(description="Encode text prompts for LTX-2")
    parser.add_argument("prompt", type=str, help="Text prompt to encode")
    parser.add_argument("--gemma-path", type=str, default="weights/gemma-3-12b", help="Path to Gemma weights")
    parser.add_argument("--ltx-weights", type=str, required=True, help="Path to LTX-2 weights (for text projection)")
    parser.add_argument("--output", type=str, default="prompt_embedding.npz", help="Output .npz file")
    args = parser.parse_args()

    print(f"Encoding prompt: '{args.prompt}'")
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.gemma_path)
    if tokenizer is None:
        sys.exit(1)

    # 2. Load Gemma
    print("Loading Gemma 3...")
    gemma = Gemma3Model(None)  # Config loaded from weights
    load_gemma3_weights(gemma, args.gemma_path, use_fp16=True) # Default to FP16 

    # 3. Load Projection Layers
    print("Loading LTX-2 text encoder projection...")
    text_encoder_state = load_text_encoder_weights(args.ltx_weights)
    text_encoder = create_text_encoder(text_encoder_state)

    # 4. Encode
    print("Running encoding...")
    # Using the helper from generate.py logic (re-implemented here strictly or importing? 
    # Importing is risky if generate.py has side effects. Let's use the function if clean.)
    
    # Actually, generate.py's encode_with_gemma handles the full pipeline including projection
    # but requires a 'gemma' instance passed in.
    
    try:
        encoding, mask = encode_with_gemma(
            gemma, 
            text_encoder, 
            tokenizer, 
            args.prompt, 
            device="mps" # MPS/MLX 
        )
        
        # 5. Save
        print(f"Saving to {args.output}...")
        np.savez_compressed(
            args.output,
            text_encoding=np.array(encoding),
            text_mask=np.array(mask)
        )
        print("Done!")
        
    except Exception as e:
        print(f"Encoding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
