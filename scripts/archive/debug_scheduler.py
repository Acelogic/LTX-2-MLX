#!/usr/bin/env python3
"""Debug scheduler with different latent shapes."""

import mlx.core as mx
import math
from LTX_2_MLX.components.schedulers import LTX2Scheduler

scheduler = LTX2Scheduler()

print("=== Testing Scheduler with Different Latent Shapes ===\n")

# Test 1: 3D patchified latent (what the pipeline actually uses)
print("Test 1: 3D Patchified Latent [B, N, D]")
latent_3d = mx.zeros((1, 1408, 3840))  # Typical patchified shape
print(f"  Shape: {latent_3d.shape}")
print(f"  shape[2:]: {latent_3d.shape[2:]}")
tokens_3d = math.prod(latent_3d.shape[2:])
print(f"  Tokens (prod of shape[2:]): {tokens_3d}")

sigmas_3d = scheduler.execute(steps=15, latent=latent_3d)
print(f"  Sigmas: {sigmas_3d[:3]}...{sigmas_3d[-3:]}")
print(f"  Has NaN: {mx.any(mx.isnan(sigmas_3d))}\n")

# Test 2: 5D unpatchified latent
print("Test 2: 5D Unpatchified Latent [B, C, F, H, W]")
latent_5d = mx.zeros((1, 128, 16, 32, 44))
print(f"  Shape: {latent_5d.shape}")
print(f"  shape[2:]: {latent_5d.shape[2:]}")
tokens_5d = math.prod(latent_5d.shape[2:])
print(f"  Tokens (prod of shape[2:]): {tokens_5d}")

sigmas_5d = scheduler.execute(steps=15, latent=latent_5d)
print(f"  Sigmas: {sigmas_5d[:3]}...{sigmas_5d[-3:]}")
print(f"  Has NaN: {mx.any(mx.isnan(sigmas_5d))}\n")

# Test 3: No latent (uses MAX_SHIFT_ANCHOR)
print("Test 3: No latent (uses default MAX_SHIFT_ANCHOR=4096)")
sigmas_none = scheduler.execute(steps=15, latent=None)
print(f"  Sigmas: {sigmas_none[:3]}...{sigmas_none[-3:]}")
print(f"  Has NaN: {mx.any(mx.isnan(sigmas_none))}\n")

# Test 4: Let's manually calculate what happens with large token counts
print("=== Manual Calculation Debug ===")
from LTX_2_MLX.components.schedulers import BASE_SHIFT_ANCHOR, MAX_SHIFT_ANCHOR

for tokens in [3840, 22528, 180224, 4096]:
    print(f"\nTokens: {tokens}")

    # Calculate shift
    x1 = BASE_SHIFT_ANCHOR  # 1024
    x2 = MAX_SHIFT_ANCHOR  # 4096
    max_shift = 2.05
    base_shift = 0.95
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = tokens * mm + b

    print(f"  sigma_shift = {sigma_shift}")
    print(f"  exp(sigma_shift) = {math.exp(sigma_shift) if sigma_shift < 100 else 'overflow'}")

    # Test with a single sigma value
    test_sigma = 0.5
    if sigma_shift < 100:
        exp_shift = math.exp(sigma_shift)
        transformed = exp_shift / (exp_shift + (1.0 / test_sigma - 1.0))
        print(f"  transformed sigma (0.5) = {transformed}")
    else:
        print(f"  transformed sigma (0.5) = NaN (overflow)")
