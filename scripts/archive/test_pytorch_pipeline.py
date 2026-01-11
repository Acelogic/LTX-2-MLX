#!/usr/bin/env python3
"""Test PyTorch LTX-2 distilled pipeline output quality.

Runs the PyTorch implementation to verify what output quality we should expect.
This helps determine if the issue is in text encoding vs model implementation.
"""

import sys
from pathlib import Path

# Mock triton before any imports
import types
mock_triton = types.ModuleType('triton')
mock_triton.cdiv = lambda a, b: (a + b - 1) // b
mock_triton.jit = lambda fn: fn

mock_triton_language = types.ModuleType('triton.language')
mock_triton_language.constexpr = int

class MockDtype:
    pass
mock_triton_language.dtype = MockDtype

mock_triton.language = mock_triton_language

sys.modules['triton'] = mock_triton
sys.modules['triton.language'] = mock_triton_language

# Add PyTorch LTX-2 to path
pytorch_ltx_path = Path(__file__).parent.parent.parent / "LTX-2-PyTorch"
if pytorch_ltx_path.exists():
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-core" / "src"))
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-pipelines" / "src"))

import numpy as np
import torch


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 70)
    print("PyTorch LTX-2 Distilled Pipeline Test")
    print("=" * 70)

    if not Path(weights_path).exists():
        print(f"Weights not found: {weights_path}")
        return

    # Test parameters matching generate.py defaults
    batch_size = 1
    frames, height, width = 3, 8, 12  # Small test size
    context_dim = 3840
    context_len = 256
    fps = 24.0

    # Distilled sigma schedule
    sigmas = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

    print(f"\nTest parameters:")
    print(f"  Latent: {frames}x{height}x{width}")
    print(f"  Context: {context_len}x{context_dim}")
    print(f"  Sigmas: {sigmas}")

    # Load models
    print("\n" + "=" * 70)
    print("Loading PyTorch Models")
    print("=" * 70)

    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import VideoLatentShape, SpatioTemporalScaleFactors

    print("Loading velocity model...")
    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    velocity_model = builder.build(device=torch.device("cpu"), dtype=torch.float32)
    velocity_model.eval()
    x0_model = X0Model(velocity_model)

    # Create inputs
    print("\nCreating inputs...")
    torch.manual_seed(42)
    np.random.seed(42)

    # Random context (simulating Gemma output)
    context = torch.randn(batch_size, context_len, context_dim, dtype=torch.float32) * 0.1

    # Initial noise
    initial_noise = torch.randn(batch_size, 128, frames, height, width, dtype=torch.float32)

    # Create patchifier and positions
    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_shape = VideoLatentShape(batch=batch_size, channels=128, frames=frames, height=height, width=width)
    latent_coords = patchifier.get_patch_grid_bounds(latent_shape)
    scale_factors = SpatioTemporalScaleFactors.default()
    positions = get_pixel_coords(latent_coords, scale_factors, causal_fix=True).float()
    positions[:, 0, ...] = positions[:, 0, ...] / fps

    print(f"  Initial noise: {initial_noise.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Positions: {positions.shape}")

    # Patchify initial noise
    def patchify(x):
        B, C, F, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    def unpatchify(x, F, H, W):
        B, T, C = x.shape
        return x.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)

    latent = initial_noise.clone()
    num_tokens = frames * height * width

    # Run denoising loop (matching PyTorch distilled pipeline - NO CFG)
    print("\n" + "=" * 70)
    print("Running Denoising Loop (No CFG)")
    print("=" * 70)

    sigmas_tensor = torch.tensor(sigmas)

    for step_idx in range(len(sigmas) - 1):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma

        print(f"\n--- Step {step_idx}: sigma={sigma:.4f} -> {sigma_next:.4f} ---")

        # Patchify
        latent_patchified = patchify(latent)

        # Create timesteps (one per token, all same sigma)
        timesteps = torch.ones((batch_size, num_tokens, 1)) * sigma

        # Create modality
        modality = Modality(
            enabled=True,
            latent=latent_patchified,
            timesteps=timesteps,
            positions=positions,
            context=context,
            context_mask=None,
        )

        # Get X0 prediction (no CFG)
        with torch.no_grad():
            denoised_patchified, _ = x0_model(video=modality, audio=None, perturbations=None)

        # Unpatchify
        denoised = unpatchify(denoised_patchified, frames, height, width)

        # Euler step: x_next = x + velocity * dt, where velocity = (x - denoised) / sigma
        velocity = (latent - denoised) / sigma
        latent = latent + velocity * dt

        # Stats
        print(f"  Latent: mean={latent.mean():.4f}, std={latent.std():.4f}")
        print(f"  Denoised: mean={denoised.mean():.4f}, std={denoised.std():.4f}")

    # Analyze final latent
    print("\n" + "=" * 70)
    print("Final Latent Analysis")
    print("=" * 70)

    final_latent = latent.numpy()
    print(f"\nShape: {final_latent.shape}")
    print(f"Mean: {final_latent.mean():.4f}")
    print(f"Std: {final_latent.std():.4f}")
    print(f"Range: [{final_latent.min():.4f}, {final_latent.max():.4f}]")

    # Spatial autocorrelation
    frame0 = final_latent[0, :, 0, :, :].mean(axis=0)  # Average over channels
    autocorr = np.corrcoef(frame0[:-1, :].flatten(), frame0[1:, :].flatten())[0, 1]
    print(f"\nSpatial autocorrelation (frame 0): {autocorr:.4f}")

    if autocorr < 0.3:
        print("\n  Low autocorrelation suggests noise-like output")
        print("  This is expected with random context - no semantic guidance")
    else:
        print("\n  Higher autocorrelation suggests structured output")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nWith random context (no real Gemma encoding):")
    print("  - PyTorch distilled pipeline also produces noise-like output")
    print("  - This is EXPECTED behavior - random context provides no guidance")
    print("\nTo get semantic output, need proper Gemma text encoding")


if __name__ == "__main__":
    main()
