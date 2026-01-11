#!/usr/bin/env python3
"""Compare full model forward pass MLX vs PyTorch.

This tests the complete inference path through Modality, including
preprocessor and all transformer blocks.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add PyTorch LTX-2 to path
pytorch_ltx_path = Path(__file__).parent.parent.parent / "LTX-2-PyTorch"
if pytorch_ltx_path.exists():
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-core" / "src"))
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-pipelines" / "src"))

import numpy as np
import torch
import mlx.core as mx


def compare_arrays(name, a, b, rtol=0.01, atol=0.01):
    """Compare two arrays and print results."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH - {a.shape} vs {b.shape}")
        return False

    abs_diff = np.abs(a - b)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    try:
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    except:
        corr = float('nan')

    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "OK" if close else "DIFF"

    print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.4f}")
    if not close:
        print(f"    MLX: mean={a.mean():.6f}, std={a.std():.6f}, range=[{a.min():.4f}, {a.max():.4f}]")
        print(f"    PT:  mean={b.mean():.6f}, std={b.std():.6f}, range=[{b.min():.4f}, {b.max():.4f}]")

    return close, corr


def create_pixel_space_positions(frames, height, width, fps=24.0, time_scale=8, spatial_scale=32):
    """Create pixel-space positions matching PyTorch production pipeline."""
    batch_size = 1

    frame_coords = np.arange(0, frames)
    height_coords = np.arange(0, height)
    width_coords = np.arange(0, width)

    grid_f, grid_h, grid_w = np.meshgrid(frame_coords, height_coords, width_coords, indexing="ij")

    patch_starts = np.stack([grid_f, grid_h, grid_w], axis=0)
    patch_ends = patch_starts + 1

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_tokens = frames * height * width
    latent_coords = latent_coords.reshape(3, num_tokens, 2)
    latent_coords = latent_coords[None, ...]

    scale_factors = np.array([time_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_factors

    pixel_coords[:, 0, ...] = np.maximum(pixel_coords[:, 0, ...] + 1 - time_scale, 0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

    return pixel_coords.astype(np.float32)


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 70)
    print("Full Model Forward Pass Comparison")
    print("=" * 70)

    if not Path(weights_path).exists():
        print(f"Weights not found: {weights_path}")
        return

    # Test parameters
    np.random.seed(42)
    batch_size = 1
    latent_channels = 128
    frames, height, width = 3, 8, 12
    context_dim = 3840
    context_len = 256
    sigma = 1.0

    # Create inputs
    latent = np.random.randn(batch_size, latent_channels, frames, height, width).astype(np.float32) * 0.1
    context = np.random.randn(batch_size, context_len, context_dim).astype(np.float32) * 0.1

    def patchify(x):
        B, C, F, H, W = x.shape
        return x.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    def unpatchify(x, F, H, W):
        B, T, C = x.shape
        return x.reshape(B, F, H, W, C).transpose(0, 4, 1, 2, 3)

    latent_patchified = patchify(latent)
    positions = create_pixel_space_positions(frames, height, width)

    print(f"\nInput shapes:")
    print(f"  Latent (patchified): {latent_patchified.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Positions: {positions.shape}")
    print(f"  Sigma: {sigma}")

    # Load PyTorch model
    print("\n" + "=" * 70)
    print("Loading models")
    print("=" * 70)

    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.model import X0Model as PTX0Model
    from ltx_core.model.transformer.modality import Modality as PTModality

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pt_velocity_model = builder.build(device=torch.device("cpu"), dtype=torch.float32)
    pt_velocity_model.eval()
    pt_x0_model = PTX0Model(pt_velocity_model)

    # Load MLX model
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, X0Model as MLXX0Model
    from LTX_2_MLX.model.transformer.model import Modality as MLXModality
    from LTX_2_MLX.loader import load_transformer_weights

    mlx_velocity_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        compute_dtype=mx.float32,
    )
    load_transformer_weights(mlx_velocity_model, weights_path)
    mlx_x0_model = MLXX0Model(mlx_velocity_model)

    # Create Modality objects
    print("\n" + "=" * 70)
    print("Testing Velocity Model (raw transformer output)")
    print("=" * 70)

    # MLX Modality
    mlx_modality = MLXModality(
        latent=mx.array(latent_patchified),
        context=mx.array(context),
        context_mask=None,
        timesteps=mx.array([sigma]),
        positions=mx.array(positions),
        enabled=True,
    )

    # PyTorch Modality
    # PyTorch expects timesteps as (B, T, 1) = denoise_mask * sigma
    num_tokens = latent_patchified.shape[1]
    pt_timesteps = np.ones((batch_size, num_tokens, 1), dtype=np.float32) * sigma

    pt_modality = PTModality(
        enabled=True,
        latent=torch.from_numpy(latent_patchified),
        timesteps=torch.from_numpy(pt_timesteps),
        positions=torch.from_numpy(positions),
        context=torch.from_numpy(context),
        context_mask=None,
    )

    # Run velocity models
    print("\nRunning velocity models...")
    mlx_velocity = mlx_velocity_model(mlx_modality)
    mx.eval(mlx_velocity)

    with torch.no_grad():
        pt_velocity, _ = pt_velocity_model(video=pt_modality, audio=None, perturbations=None)

    match, corr = compare_arrays("Velocity output", np.array(mlx_velocity), pt_velocity.numpy())

    # Test X0 model
    print("\n" + "=" * 70)
    print("Testing X0 Model (denoised prediction)")
    print("=" * 70)

    mlx_denoised = mlx_x0_model(mlx_modality)
    mx.eval(mlx_denoised)

    with torch.no_grad():
        pt_denoised, _ = pt_x0_model(video=pt_modality, audio=None, perturbations=None)

    match, corr = compare_arrays("X0 (denoised) output", np.array(mlx_denoised), pt_denoised.numpy())

    # Unpatchify and check spatial structure
    mlx_denoised_5d = unpatchify(np.array(mlx_denoised), frames, height, width)
    pt_denoised_5d = unpatchify(pt_denoised.numpy(), frames, height, width)

    match, corr = compare_arrays("X0 (unpatchified)", mlx_denoised_5d, pt_denoised_5d)

    # Check spatial autocorrelation
    print("\n" + "=" * 70)
    print("Spatial Autocorrelation Analysis")
    print("=" * 70)

    mlx_frame0 = mlx_denoised_5d[0, :, 0, :, :].mean(axis=0)  # Average over channels
    pt_frame0 = pt_denoised_5d[0, :, 0, :, :].mean(axis=0)

    mlx_autocorr = np.corrcoef(mlx_frame0[:-1, :].flatten(), mlx_frame0[1:, :].flatten())[0, 1]
    pt_autocorr = np.corrcoef(pt_frame0[:-1, :].flatten(), pt_frame0[1:, :].flatten())[0, 1]

    print(f"\n  MLX denoised spatial autocorr: {mlx_autocorr:.4f}")
    print(f"  PT denoised spatial autocorr: {pt_autocorr:.4f}")

    if mlx_autocorr < 0.3:
        print(f"\n  WARNING: Low spatial autocorrelation ({mlx_autocorr:.2f}) suggests noise-like output")
        print(f"  Expected: ~0.5+ for coherent predictions")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if corr > 0.99:
        print("\n  MLX and PyTorch produce MATCHING outputs!")
        print("  The model implementation is correct.")
        if mlx_autocorr < 0.3:
            print("\n  BUT: Both produce low spatial autocorrelation.")
            print("  This could indicate:")
            print("    1. Test inputs (random noise) produce noise-like outputs")
            print("    2. Single forward pass at sigma=1.0 expected to be noisy")
            print("    3. Need to check full denoising loop over multiple steps")
    else:
        print(f"\n  WARNING: Outputs differ (corr={corr:.4f})")


if __name__ == "__main__":
    main()
