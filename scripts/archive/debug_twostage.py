#!/usr/bin/env python3
"""Debug script to investigate why two-stage pipeline produces noise."""

import mlx.core as mx
import numpy as np
from LTX_2_MLX.components.schedulers import LTX2Scheduler
from LTX_2_MLX.pipelines.two_stage import TwoStageCFGConfig

# Test sigma schedule generation
print("=== Testing LTX2Scheduler ===")
scheduler = LTX2Scheduler()

# Create dummy latent (similar to what the pipeline uses)
dummy_latent = mx.random.normal((1, 16, 32, 44, 128))  # [B, F, H, W, C]

# Generate sigmas for 15 steps (same as two-stage Stage 1)
sigmas = scheduler.execute(steps=15, latent=dummy_latent)
print(f"Generated {len(sigmas)} sigma values for 15 steps")
print(f"Sigma shape: {sigmas.shape}")
print(f"Sigma values: {sigmas}")
print(f"Sigma dtype: {sigmas.dtype}")

# Check number of denoising iterations
num_iterations = len(sigmas) - 1
print(f"\nNumber of denoising iterations: {num_iterations}")

if num_iterations == 0:
    print("❌ ERROR: No denoising iterations! This explains the noise output.")
else:
    print(f"✅ OK: {num_iterations} iterations expected")

# Test TwoStageCFGConfig
print("\n=== Testing TwoStageCFGConfig ===")
try:
    config = TwoStageCFGConfig(
        height=512,
        width=704,
        num_frames=97,
        seed=42,
        fps=24.0,
        num_inference_steps=15,
        cfg_scale=5.0,
        dtype=mx.float16,
    )
    print(f"✅ Config created successfully")
    print(f"  Stage 1 resolution: {config.height//2}x{config.width//2}")
    print(f"  Stage 2 resolution: {config.height}x{config.width}")
    print(f"  Inference steps: {config.num_inference_steps}")
    print(f"  CFG scale: {config.cfg_scale}")
except Exception as e:
    print(f"❌ Config creation failed: {e}")

print("\n=== Debug Complete ===")
