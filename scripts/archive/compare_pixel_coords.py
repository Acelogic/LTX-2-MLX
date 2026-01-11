#!/usr/bin/env python3
"""Compare MLX vs PyTorch with pixel-space coordinates (production format).

The previous comparison used latent-space coordinates (0, 1, 2, ...).
This script tests with pixel-space coordinates (0, 32, 64, ...) to verify
that both implementations produce identical outputs when using production
coordinate formats.
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
        print(f"    MLX: mean={a.mean():.6f}, std={a.std():.6f}")
        print(f"    PT:  mean={b.mean():.6f}, std={b.std():.6f}")

    return close


def create_pixel_space_positions(frames, height, width, fps=24.0, time_scale=8, spatial_scale=32):
    """Create pixel-space positions matching PyTorch production pipeline.

    This replicates what VideoLatentTools.create_initial_state() does:
    1. get_patch_grid_bounds() -> latent coords [0, 1], [1, 2], etc.
    2. get_pixel_coords() -> scale by VAE factors (8 for time, 32 for spatial)
    3. Apply causal_fix for temporal dimension
    4. Divide temporal by fps
    """
    batch_size = 1

    # Generate latent coordinate grid
    frame_coords = np.arange(0, frames)
    height_coords = np.arange(0, height)
    width_coords = np.arange(0, width)

    grid_f, grid_h, grid_w = np.meshgrid(frame_coords, height_coords, width_coords, indexing="ij")

    # Stack to get start coordinates
    patch_starts = np.stack([grid_f, grid_h, grid_w], axis=0)  # (3, F, H, W)
    patch_ends = patch_starts + 1  # (3, F, H, W) - patch size is 1

    # Stack [start, end) bounds
    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)  # (3, F, H, W, 2)

    # Flatten spatial dims: (3, F*H*W, 2)
    num_tokens = frames * height * width
    latent_coords = latent_coords.reshape(3, num_tokens, 2)

    # Add batch dimension: (1, 3, num_tokens, 2)
    latent_coords = latent_coords[None, ...]

    # Convert to pixel space (like get_pixel_coords)
    scale_factors = np.array([time_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_factors

    # Apply causal_fix for temporal dimension
    # temporal = temporal + 1 - time_scale, clamped to 0
    pixel_coords[:, 0, ...] = np.maximum(pixel_coords[:, 0, ...] + 1 - time_scale, 0)

    # Divide temporal by fps
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

    return pixel_coords.astype(np.float32)


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 70)
    print("Pixel-Space Coordinates Comparison")
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
    sigma = 1.0  # Initial noise level

    # Create inputs
    latent = np.random.randn(batch_size, latent_channels, frames, height, width).astype(np.float32) * 0.1
    context = np.random.randn(batch_size, context_len, context_dim).astype(np.float32) * 0.1

    # Patchify latent
    def patchify(x):
        B, C, F, H, W = x.shape
        return x.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    latent_patchified = patchify(latent)

    # Create PIXEL-SPACE positions (not latent-space!)
    positions = create_pixel_space_positions(frames, height, width)

    print(f"\nInput shapes:")
    print(f"  Latent (patchified): {latent_patchified.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Positions: {positions.shape}")
    print(f"  Sigma: {sigma}")

    print(f"\nPosition samples (pixel-space):")
    print(f"  Temporal first 5 starts: {positions[0, 0, :5, 0]}")  # Should be 0/24, 0/24, 0/24, ... then 1/24...
    print(f"  Spatial (H) first 5 starts: {positions[0, 1, :5, 0]}")  # Should be 0, 32, 64, ... or pattern
    print(f"  Spatial (W) first 5 starts: {positions[0, 2, :5, 0]}")  # Should be 0, 32, 64, ...

    # Load PyTorch model
    print("\n" + "=" * 70)
    print("Loading models")
    print("=" * 70)

    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.rope import precompute_freqs_cis, generate_freq_grid_np
    from ltx_core.model.transformer.rope import LTXRopeType as PTRopeType

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pt_model = builder.build(device=torch.device("cpu"), dtype=torch.float32)
    pt_model.eval()

    # Load MLX model
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.model.transformer.rope import precompute_freqs_cis as mlx_precompute_freqs_cis
    from LTX_2_MLX.model.transformer.rope import LTXRopeType

    mlx_model = LTXModel(
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
    load_transformer_weights(mlx_model, weights_path)

    # Prepare inputs
    print("\nPreparing inputs...")

    # === MLX ===
    mlx_x = mx.array(latent_patchified)
    mlx_x = mlx_model.patchify_proj(mlx_x)
    mx.eval(mlx_x)

    mlx_context = mx.array(context)
    mlx_context = mlx_model.caption_projection(mlx_context)
    mlx_context = mlx_context.reshape(1, -1, mlx_x.shape[-1])
    mx.eval(mlx_context)

    # Timestep
    timestep_scale = 1000.0
    mlx_timestep = mx.array([sigma * timestep_scale])
    mlx_ts_emb, mlx_embedded = mlx_model.adaln_single(mlx_timestep)
    mx.eval(mlx_ts_emb, mlx_embedded)

    # Broadcast timestep to all tokens
    num_tokens = mlx_x.shape[1]
    mlx_ts = mlx_ts_emb.reshape(1, 1, 6, -1)
    mlx_ts = mx.broadcast_to(mlx_ts, (1, num_tokens, 6, mlx_ts.shape[-1]))
    mx.eval(mlx_ts)

    # RoPE with pixel-space positions
    mlx_positions = mx.array(positions)
    mlx_rope = mlx_precompute_freqs_cis(
        mlx_positions,
        dim=4096,
        out_dtype=mx.float32,
        theta=10000.0,
        max_pos=[128, 128, 128],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=LTXRopeType.SPLIT,
    )
    mx.eval(mlx_rope[0], mlx_rope[1])

    # === PyTorch ===
    pt_x = torch.from_numpy(latent_patchified)
    with torch.no_grad():
        pt_x = pt_model.patchify_proj(pt_x)

    pt_context = torch.from_numpy(context)
    with torch.no_grad():
        pt_context = pt_model.caption_projection(pt_context)
        pt_context = pt_context.view(1, -1, pt_x.shape[-1])

    with torch.no_grad():
        pt_timestep = torch.tensor([sigma * timestep_scale])
        pt_ts_emb, pt_embedded = pt_model.adaln_single(pt_timestep.flatten(), hidden_dtype=torch.float32)

    pt_ts = pt_ts_emb.view(1, 1, 6, -1).expand(1, num_tokens, 6, -1).contiguous()

    # PyTorch RoPE with same pixel-space positions
    pt_positions = torch.from_numpy(positions)
    with torch.no_grad():
        pt_rope = precompute_freqs_cis(
            pt_positions,
            dim=4096,
            out_dtype=torch.float32,
            theta=10000.0,
            max_pos=[128, 128, 128],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=PTRopeType.SPLIT,
            freq_grid_generator=generate_freq_grid_np,
        )

    # Compare inputs
    print("\n" + "=" * 70)
    print("Input Comparison")
    print("=" * 70)

    compare_arrays("projected x", np.array(mlx_x), pt_x.numpy())
    compare_arrays("projected context", np.array(mlx_context), pt_context.numpy())
    compare_arrays("timestep emb", np.array(mlx_ts), pt_ts.numpy())
    compare_arrays("RoPE cos", np.array(mlx_rope[0]), pt_rope[0].numpy())
    compare_arrays("RoPE sin", np.array(mlx_rope[1]), pt_rope[1].numpy())

    # Run through first block to compare
    print("\n" + "=" * 70)
    print("First Block Output")
    print("=" * 70)

    from LTX_2_MLX.model.transformer.attention import rms_norm
    from ltx_core.utils import rms_norm as pt_rms_norm

    mlx_block = mlx_model.transformer_blocks[0]
    pt_block = pt_model.transformer_blocks[0]

    # Self-attention
    table_slice = mlx_block.scale_shift_table[0:3]
    ada_values = table_slice[None, None, :, :] + mlx_ts[:, :, 0:3, :]
    shift_msa = ada_values[:, :, 0, :]
    scale_msa = ada_values[:, :, 1, :]
    gate_msa = ada_values[:, :, 2, :]

    norm_vx = rms_norm(mlx_x, eps=1e-6) * (1 + scale_msa) + shift_msa
    mlx_attn = mlx_block.attn1(norm_vx, pe=mlx_rope)
    mlx_x1 = mlx_x + mlx_attn * gate_msa
    mx.eval(mlx_x1)

    # Cross-attention
    mlx_x2 = mlx_x1 + mlx_block.attn2(rms_norm(mlx_x1, eps=1e-6), context=mlx_context, mask=None)
    mx.eval(mlx_x2)

    # FFN
    ff_table = mlx_block.scale_shift_table[3:6]
    ff_ada = ff_table[None, None, :, :] + mlx_ts[:, :, 3:6, :]
    shift_mlp = ff_ada[:, :, 0, :]
    scale_mlp = ff_ada[:, :, 1, :]
    gate_mlp = ff_ada[:, :, 2, :]

    vx_scaled = rms_norm(mlx_x2, eps=1e-6) * (1 + scale_mlp) + shift_mlp
    mlx_out = mlx_x2 + mlx_block.ff(vx_scaled) * gate_mlp
    mx.eval(mlx_out)

    # PyTorch block
    with torch.no_grad():
        pt_table = pt_block.scale_shift_table[0:3].unsqueeze(0).unsqueeze(0)
        pt_ada = pt_table + pt_ts[:, :, 0:3, :]
        pt_shift_msa = pt_ada[:, :, 0, :]
        pt_scale_msa = pt_ada[:, :, 1, :]
        pt_gate_msa = pt_ada[:, :, 2, :]

        pt_norm_vx = pt_rms_norm(pt_x, eps=1e-6) * (1 + pt_scale_msa) + pt_shift_msa
        pt_attn = pt_block.attn1(pt_norm_vx, pe=pt_rope)
        pt_x1 = pt_x + pt_attn * pt_gate_msa

        pt_x2 = pt_x1 + pt_block.attn2(pt_rms_norm(pt_x1, eps=1e-6), context=pt_context, mask=None)

        pt_ff_table = pt_block.scale_shift_table[3:6].unsqueeze(0).unsqueeze(0)
        pt_ff_ada = pt_ff_table + pt_ts[:, :, 3:6, :]
        pt_shift_mlp = pt_ff_ada[:, :, 0, :]
        pt_scale_mlp = pt_ff_ada[:, :, 1, :]
        pt_gate_mlp = pt_ff_ada[:, :, 2, :]

        pt_vx_scaled = pt_rms_norm(pt_x2, eps=1e-6) * (1 + pt_scale_mlp) + pt_shift_mlp
        pt_out = pt_x2 + pt_block.ff(pt_vx_scaled) * pt_gate_mlp

    compare_arrays("After self-attn", np.array(mlx_x1), pt_x1.numpy())
    compare_arrays("After cross-attn", np.array(mlx_x2), pt_x2.numpy())
    compare_arrays("After FFN", np.array(mlx_out), pt_out.numpy())

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nIf block 0 matches with pixel-space coords, the model itself is correct.")
    print("Issue would then be in how generate.py sets up the pipeline.")


if __name__ == "__main__":
    main()
