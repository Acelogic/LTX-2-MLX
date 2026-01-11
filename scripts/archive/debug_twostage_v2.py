#!/usr/bin/env python3
"""Debug script v2 - Test with actual patchified latent shape."""

import mlx.core as mx
from LTX_2_MLX.components.schedulers import LTX2Scheduler
from LTX_2_MLX.conditioning.tools import VideoLatentTools
from LTX_2_MLX.types import VideoLatentShape, VideoPixelShape

print("=== Testing Scheduler with Actual Patchified Latent ===\n")

# Create stage 1 output shape (half resolution for two-stage)
stage_1_pixel_shape = VideoPixelShape(
    batch=1,
    frames=97,
    height=256,  # 512 // 2
    width=352,   # 704 // 2
    fps=24.0,
)

stage_1_latent_shape = VideoLatentShape.from_pixel_shape(
    stage_1_pixel_shape, latent_channels=128
)

print(f"Stage 1 pixel shape: {stage_1_pixel_shape}")
print(f"Stage 1 latent shape (unpatchified): {stage_1_latent_shape}")
print(f"  B={stage_1_latent_shape.batch}, C={stage_1_latent_shape.channels}, ")
print(f"  F={stage_1_latent_shape.frames}, H={stage_1_latent_shape.height}, W={stage_1_latent_shape.width}")

# Create video tools to get patchified shape
video_tools = VideoLatentTools(
    target_shape=stage_1_latent_shape,
    patch_size=(1, 2, 2),
    fps=24.0,
)

# Create initial state (this will be patchified)
initial_state = video_tools.create_initial_state(dtype=mx.float16)

print(f"\nPatchified latent shape: {initial_state.latent.shape}")
print(f"  B={initial_state.latent.shape[0]}, N={initial_state.latent.shape[1]}, D={initial_state.latent.shape[2]}")

# Calculate token count the way scheduler does
tokens_calculated = 1
for dim in initial_state.latent.shape[2:]:
    tokens_calculated *= dim
print(f"\nTokens calculated by scheduler (shape[2:]): {tokens_calculated}")

# Test scheduler
scheduler = LTX2Scheduler()
sigmas = scheduler.execute(steps=15, latent=initial_state.latent)

print(f"\nScheduler output:")
print(f"  Sigma shape: {sigmas.shape}")
print(f"  Sigma dtype: {sigmas.dtype}")
print(f"  First 5 sigmas: {sigmas[:5]}")
print(f"  Last 5 sigmas: {sigmas[-5:]}")

# Check for NaN
has_nan = mx.any(mx.isnan(sigmas))
print(f"\nHas NaN values: {has_nan}")

if has_nan:
    print("❌ FAILED: Sigma schedule contains NaN!")
else:
    print("✅ PASSED: Sigma schedule is valid")

# Additional debug: Try without latent argument
print("\n=== Testing scheduler WITHOUT latent (should use default) ===")
sigmas_no_latent = scheduler.execute(steps=15, latent=None)
print(f"  First 5 sigmas: {sigmas_no_latent[:5]}")
print(f"  Last 5 sigmas: {sigmas_no_latent[-5:]}")
has_nan_no_latent = mx.any(mx.isnan(sigmas_no_latent))
print(f"  Has NaN: {has_nan_no_latent}")
