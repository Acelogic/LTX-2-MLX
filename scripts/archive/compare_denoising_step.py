#!/usr/bin/env python3
"""Compare a single denoising step between MLX and PyTorch."""

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


def compare_arrays(name, mlx_arr, pt_arr, rtol=0.01, atol=0.01):
    """Compare MLX and PyTorch arrays."""
    mlx_np = np.array(mlx_arr)
    pt_np = pt_arr.detach().numpy() if hasattr(pt_arr, 'detach') else pt_arr

    if mlx_np.shape != pt_np.shape:
        print(f"  {name}: SHAPE MISMATCH - MLX {mlx_np.shape} vs PT {pt_np.shape}")
        return False

    max_diff = np.abs(mlx_np - pt_np).max()
    mean_diff = np.abs(mlx_np - pt_np).mean()
    corr = np.corrcoef(mlx_np.flatten(), pt_np.flatten())[0, 1]
    close = np.allclose(mlx_np, pt_np, rtol=rtol, atol=atol)

    status = "✓" if close else "✗"
    print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.4f}")

    if not close:
        print(f"    MLX: mean={mlx_np.mean():.6f}, std={mlx_np.std():.6f}, range=[{mlx_np.min():.4f}, {mlx_np.max():.4f}]")
        print(f"    PT:  mean={pt_np.mean():.6f}, std={pt_np.std():.6f}, range=[{pt_np.min():.4f}, {pt_np.max():.4f}]")

    return close


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 70)
    print("Single Denoising Step Comparison")
    print("=" * 70)

    # Create deterministic test inputs
    np.random.seed(42)

    batch_size = 1
    latent_channels = 128
    frames, height, width = 3, 8, 12
    context_dim = 3840
    context_len = 256
    sigma = 0.9  # Test at this noise level

    # Create latent (noisy)
    latent_bcfhw = np.random.randn(batch_size, latent_channels, frames, height, width).astype(np.float32) * sigma
    context = np.random.randn(batch_size, context_len, context_dim).astype(np.float32) * 0.1

    # Patchify function
    def patchify(x):
        # [B, C, F, H, W] -> [B, F*H*W, C]
        B, C, F, H, W = x.shape
        return x.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    def unpatchify(x, f, h, w):
        # [B, F*H*W, C] -> [B, C, F, H, W]
        B, T, C = x.shape
        return x.reshape(B, f, h, w, C).transpose(0, 4, 1, 2, 3)

    latent_patchified = patchify(latent_bcfhw)

    # Create positions
    t_coords = np.arange(frames)
    h_coords = np.arange(height)
    w_coords = np.arange(width)
    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    positions = np.stack([t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], axis=0)[None].astype(np.float32)

    print(f"\nInput shapes:")
    print(f"  Latent (BCFHW): {latent_bcfhw.shape}")
    print(f"  Latent (patchified): {latent_patchified.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Positions: {positions.shape}")
    print(f"  Sigma: {sigma}")

    # =================== Load PyTorch Model ===================
    print("\n" + "=" * 70)
    print("Loading PyTorch Model")
    print("=" * 70)

    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.model import X0Model as PTX0Model
    from ltx_core.model.transformer.rope import precompute_freqs_cis as pt_precompute_freqs_cis
    from ltx_core.model.transformer.rope import generate_freq_grid_np
    from ltx_core.model.transformer.rope import LTXRopeType as PTRopeType
    from ltx_core.model.transformer.modality import Modality as PTModality
    from ltx_core.utils import to_denoised as pt_to_denoised

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pt_velocity_model = builder.build(device=torch.device("cpu"), dtype=torch.float32)
    pt_velocity_model.eval()
    pt_x0_model = PTX0Model(pt_velocity_model)

    # =================== Load MLX Model ===================
    print("\nLoading MLX Model")

    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality as MLXModality, X0Model as MLXX0Model
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.model.transformer.rope import precompute_freqs_cis as mlx_precompute_freqs_cis
    from LTX_2_MLX.model.transformer.rope import LTXRopeType

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

    # =================== Run Through Models ===================
    print("\n" + "=" * 70)
    print("Running Single Step")
    print("=" * 70)

    # Create RoPE
    fps = 24.0

    # MLX RoPE
    mlx_positions = mx.array(positions)
    mlx_positions_with_bounds = mx.stack([mlx_positions, mlx_positions + 1], axis=-1)
    mlx_temporal = mlx_positions_with_bounds[:, 0:1, :, :] / fps
    mlx_spatial = mlx_positions_with_bounds[:, 1:, :, :]
    mlx_positions_combined = mx.concatenate([mlx_temporal, mlx_spatial], axis=1)

    mlx_rope = mlx_precompute_freqs_cis(
        mlx_positions_combined,
        dim=4096,
        out_dtype=mx.float32,
        theta=10000.0,
        max_pos=[128, 128, 128],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=LTXRopeType.SPLIT,
    )

    # PyTorch RoPE
    pt_positions = torch.from_numpy(positions)
    pt_positions_with_bounds = torch.stack([pt_positions, pt_positions + 1], dim=-1)
    pt_temporal = pt_positions_with_bounds[:, 0:1, :, :] / fps
    pt_spatial = pt_positions_with_bounds[:, 1:, :, :]
    pt_positions_combined = torch.cat([pt_temporal, pt_spatial], dim=1)

    with torch.no_grad():
        pt_rope = pt_precompute_freqs_cis(
            pt_positions_combined,
            dim=4096,
            out_dtype=torch.float32,
            theta=10000.0,
            max_pos=[128, 128, 128],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=PTRopeType.SPLIT,
            freq_grid_generator=generate_freq_grid_np,
        )

    # Create modalities
    mlx_modality = MLXModality(
        latent=mx.array(latent_patchified),
        context=mx.array(context),
        context_mask=None,
        timesteps=mx.array([sigma]),
        positions=mlx_positions,
        enabled=True,
    )

    # For PyTorch, we need denoise_mask of all ones
    denoise_mask = np.ones((batch_size, 1, frames, height, width), dtype=np.float32)
    denoise_mask_patchified = patchify(denoise_mask)
    timesteps_pt = denoise_mask_patchified * sigma  # [B, T, 1] * scalar

    pt_modality = PTModality(
        enabled=True,
        latent=torch.from_numpy(latent_patchified),
        timesteps=torch.from_numpy(timesteps_pt),
        positions=pt_positions,
        context=torch.from_numpy(context),
        context_mask=None,
    )

    print("\nTimesteps check:")
    print(f"  MLX timesteps: shape={mlx_modality.timesteps.shape}, value={float(mlx_modality.timesteps[0]):.4f}")
    print(f"  PT timesteps: shape={pt_modality.timesteps.shape}, unique values={np.unique(pt_modality.timesteps.numpy())}")

    # Run X0 models
    print("\nRunning X0 models...")

    with torch.no_grad():
        pt_denoised_patchified, _ = pt_x0_model(video=pt_modality, audio=None, perturbations=None)

    mlx_denoised_patchified = mlx_x0_model(mlx_modality)
    mx.eval(mlx_denoised_patchified)

    print("\nComparing outputs:")
    compare_arrays("X0 output (patchified)", mlx_denoised_patchified, pt_denoised_patchified)

    # Unpatchify
    mlx_denoised = unpatchify(np.array(mlx_denoised_patchified), frames, height, width)
    pt_denoised = unpatchify(pt_denoised_patchified.numpy(), frames, height, width)

    compare_arrays("X0 output (unpatchified)", mlx_denoised, pt_denoised)

    # Now compare the Euler step
    print("\n" + "=" * 70)
    print("Comparing Euler Step")
    print("=" * 70)

    sigma_next = 0.725  # Next sigma in schedule
    dt = sigma_next - sigma

    # PyTorch velocity
    pt_velocity = (torch.from_numpy(latent_bcfhw) - torch.from_numpy(pt_denoised)) / sigma
    pt_next = torch.from_numpy(latent_bcfhw) + pt_velocity * dt

    # MLX velocity
    mlx_velocity = (mx.array(latent_bcfhw) - mx.array(mlx_denoised)) / sigma
    mlx_next = mx.array(latent_bcfhw) + mlx_velocity * dt
    mx.eval(mlx_next)

    print(f"\n  dt = {dt:.4f} (sigma: {sigma} -> {sigma_next})")
    compare_arrays("velocity", mlx_velocity, pt_velocity)
    compare_arrays("next latent", mlx_next, pt_next)

    # Analyze the denoised output
    print("\n" + "=" * 70)
    print("Analyzing Denoised Output Quality")
    print("=" * 70)

    # Spatial autocorrelation of frame 0
    mlx_frame0 = mlx_denoised[0, :, 0, :, :]  # [C, H, W]
    mlx_gray = mlx_frame0.mean(axis=0)  # [H, W]
    autocorr = np.corrcoef(mlx_gray[:-1, :].flatten(), mlx_gray[1:, :].flatten())[0, 1]
    print(f"\n  MLX denoised frame 0 spatial autocorr: {autocorr:.4f}")

    pt_frame0 = pt_denoised[0, :, 0, :, :]  # [C, H, W]
    pt_gray = pt_frame0.mean(axis=0)  # [H, W]
    autocorr_pt = np.corrcoef(pt_gray[:-1, :].flatten(), pt_gray[1:, :].flatten())[0, 1]
    print(f"  PT denoised frame 0 spatial autocorr: {autocorr_pt:.4f}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nIf all comparisons show high correlation (~1.0), then MLX matches PyTorch")
    print("for a single denoising step. Issues may lie in multi-step accumulation")
    print("or other pipeline differences.")


if __name__ == "__main__":
    main()
