#!/usr/bin/env python3
"""Compare all transformer blocks between MLX and PyTorch.

This script runs through all 48 blocks to find which one diverges.
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


def compare_arrays(name, a, b, rtol=0.01, atol=0.01, verbose=True):
    """Compare two arrays and print results."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH - {a.shape} vs {b.shape}")
        return False, float('inf'), 0

    abs_diff = np.abs(a - b)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    try:
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    except:
        corr = float('nan')

    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "✓" if close else "✗"

    if verbose:
        print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.4f}")
        if not close:
            print(f"    MLX: mean={a.mean():.6f}, std={a.std():.6f}")
            print(f"    PT:  mean={b.mean():.6f}, std={b.std():.6f}")

    return close, max_diff, corr


def create_test_inputs():
    """Create test inputs."""
    np.random.seed(42)

    batch_size = 1
    latent_channels = 128
    frames, height, width = 3, 8, 12
    context_dim = 3840
    context_len = 256

    latent = np.random.randn(batch_size, latent_channels, frames, height, width).astype(np.float32) * 0.1
    context = np.random.randn(batch_size, context_len, context_dim).astype(np.float32) * 0.1
    timestep = np.array([1.0], dtype=np.float32)

    t_coords = np.arange(frames)
    h_coords = np.arange(height)
    w_coords = np.arange(width)
    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    positions = np.stack([t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], axis=0)[None].astype(np.float32)

    return {
        'latent': latent,
        'context': context,
        'timestep': timestep,
        'positions': positions,
    }


def patchify_latent(latent):
    B, C, F, H, W = latent.shape
    return latent.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)


def run_mlx_block(block, x, context, timesteps, rope, context_mask, norm_eps=1e-6):
    """Run a single MLX transformer block."""
    from LTX_2_MLX.model.transformer.attention import rms_norm

    vx = x
    batch_size = vx.shape[0]

    # Get AdaLN values for self-attention
    table_slice = block.scale_shift_table[0:3]
    ada_values = table_slice[None, None, :, :] + timesteps[:, :, 0:3, :]
    shift_msa = ada_values[:, :, 0, :]
    scale_msa = ada_values[:, :, 1, :]
    gate_msa = ada_values[:, :, 2, :]

    # Self-attention
    norm_vx = rms_norm(vx, eps=norm_eps) * (1 + scale_msa) + shift_msa
    vx = vx + block.attn1(norm_vx, pe=rope) * gate_msa

    # Cross-attention
    vx = vx + block.attn2(rms_norm(vx, eps=norm_eps), context=context, mask=context_mask)

    # Get AdaLN values for FFN
    ff_table = block.scale_shift_table[3:6]
    ff_ada = ff_table[None, None, :, :] + timesteps[:, :, 3:6, :]
    shift_mlp = ff_ada[:, :, 0, :]
    scale_mlp = ff_ada[:, :, 1, :]
    gate_mlp = ff_ada[:, :, 2, :]

    # FFN
    vx_scaled = rms_norm(vx, eps=norm_eps) * (1 + scale_mlp) + shift_mlp
    vx = vx + block.ff(vx_scaled) * gate_mlp

    return vx


def run_pt_block(block, x, context, timesteps, rope, context_mask, norm_eps=1e-6):
    """Run a single PyTorch transformer block."""
    from ltx_core.utils import rms_norm

    vx = x
    batch_size = vx.shape[0]

    # Get AdaLN values for self-attention
    table_slice = block.scale_shift_table[0:3].unsqueeze(0).unsqueeze(0)
    ada_values = table_slice + timesteps[:, :, 0:3, :]
    shift_msa = ada_values[:, :, 0, :]
    scale_msa = ada_values[:, :, 1, :]
    gate_msa = ada_values[:, :, 2, :]

    # Self-attention
    norm_vx = rms_norm(vx, eps=norm_eps) * (1 + scale_msa) + shift_msa
    vx = vx + block.attn1(norm_vx, pe=rope) * gate_msa

    # Cross-attention
    vx = vx + block.attn2(rms_norm(vx, eps=norm_eps), context=context, mask=context_mask)

    # Get AdaLN values for FFN
    ff_table = block.scale_shift_table[3:6].unsqueeze(0).unsqueeze(0)
    ff_ada = ff_table + timesteps[:, :, 3:6, :]
    shift_mlp = ff_ada[:, :, 0, :]
    scale_mlp = ff_ada[:, :, 1, :]
    gate_mlp = ff_ada[:, :, 2, :]

    # FFN
    vx_scaled = rms_norm(vx, eps=norm_eps) * (1 + scale_mlp) + shift_mlp
    vx = vx + block.ff(vx_scaled) * gate_mlp

    return vx


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 60)
    print("All Blocks Comparison")
    print("=" * 60)

    if not Path(weights_path).exists():
        print(f"Weights not found: {weights_path}")
        return

    # Create test inputs
    inputs = create_test_inputs()
    latent_patchified = patchify_latent(inputs['latent'])

    # Load PyTorch model
    print("\nLoading PyTorch model...")
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
    print("Loading MLX model...")
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

    # Patchify and project
    mlx_x = mx.array(latent_patchified)
    mlx_x = mlx_model.patchify_proj(mlx_x)
    mx.eval(mlx_x)

    pt_x = torch.from_numpy(latent_patchified)
    with torch.no_grad():
        pt_x = pt_model.patchify_proj(pt_x)

    # Context
    mlx_context = mx.array(inputs['context'])
    mlx_context = mlx_model.caption_projection(mlx_context)
    mlx_context = mlx_context.reshape(1, -1, mlx_x.shape[-1])
    mx.eval(mlx_context)

    pt_context = torch.from_numpy(inputs['context'])
    with torch.no_grad():
        pt_context = pt_model.caption_projection(pt_context)
        pt_context = pt_context.view(1, -1, pt_x.shape[-1])

    # Timestep
    timestep_scale = 1000.0
    mlx_timestep = mx.array(inputs['timestep'] * timestep_scale)
    mlx_ts_emb, mlx_embedded = mlx_model.adaln_single(mlx_timestep)
    mx.eval(mlx_ts_emb, mlx_embedded)

    with torch.no_grad():
        pt_timestep = torch.tensor(inputs['timestep'] * timestep_scale)
        pt_ts_emb, pt_embedded = pt_model.adaln_single(pt_timestep.flatten(), hidden_dtype=torch.float32)

    # Reshape timesteps for blocks
    num_tokens = mlx_x.shape[1]
    mlx_ts = mlx_ts_emb.reshape(1, 1, 6, -1)
    mlx_ts = mx.broadcast_to(mlx_ts, (1, num_tokens, 6, mlx_ts.shape[-1]))
    mx.eval(mlx_ts)

    pt_ts = pt_ts_emb.view(1, 1, 6, -1).expand(1, num_tokens, 6, -1).contiguous()

    # RoPE
    positions_3d = mx.array(inputs['positions'])
    positions_with_bounds = mx.stack([positions_3d, positions_3d + 1], axis=-1)
    fps = 24.0
    temporal = positions_with_bounds[:, 0:1, :, :] / fps
    spatial = positions_with_bounds[:, 1:, :, :]
    mlx_positions = mx.concatenate([temporal, spatial], axis=1)

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

    pt_positions_3d = torch.from_numpy(inputs['positions'])
    pt_positions = torch.stack([pt_positions_3d, pt_positions_3d + 1], dim=-1)
    pt_temporal = pt_positions[:, 0:1, :, :] / fps
    pt_spatial = pt_positions[:, 1:, :, :]
    pt_positions = torch.cat([pt_temporal, pt_spatial], dim=1)

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

    # Run through all blocks
    print("\n" + "=" * 60)
    print("Running through all 48 blocks...")
    print("=" * 60)

    first_divergence = None
    divergence_corrs = []

    for i in range(48):
        mlx_block = mlx_model.transformer_blocks[i]
        pt_block = pt_model.transformer_blocks[i]

        mlx_x = run_mlx_block(mlx_block, mlx_x, mlx_context, mlx_ts, mlx_rope, None)
        mx.eval(mlx_x)

        with torch.no_grad():
            pt_x = run_pt_block(pt_block, pt_x, pt_context, pt_ts, pt_rope, None)

        match, max_diff, corr = compare_arrays(f"Block {i:2d}", np.array(mlx_x), pt_x.numpy(), verbose=True)
        divergence_corrs.append((i, corr))

        if not match and first_divergence is None:
            first_divergence = i

    # Final projection
    print("\n" + "=" * 60)
    print("Final Projection")
    print("=" * 60)

    # Apply final scale/shift
    mlx_table = mlx_model.scale_shift_table
    pt_table = pt_model.scale_shift_table

    compare_arrays("final scale_shift_table", np.array(mlx_table), pt_table.detach().numpy())

    # Final embedded timestep (for scale/shift)
    # embedded_timestep shape: (B, inner_dim) - needs to be (B, 1, inner_dim)
    mlx_final_ts = mlx_embedded[:, None, :]  # (B, 1, D)
    pt_final_ts = pt_embedded.unsqueeze(1)  # (B, 1, D)

    print(f"  embedded_timestep MLX shape: {mlx_final_ts.shape}")
    print(f"  embedded_timestep PT shape: {pt_final_ts.shape}")
    print(f"  scale_shift_table MLX shape: {mlx_table.shape}")
    print(f"  scale_shift_table PT shape: {pt_table.shape}")

    # scale_shift_table: (2, inner_dim)
    # scale_shift_values: (B, 1, 2, inner_dim)
    mlx_scale_shift = mlx_table[None, None, :, :] + mlx_final_ts[:, :, None, :]
    mlx_shift = mlx_scale_shift[:, :, 0, :]  # (B, 1, inner_dim)
    mlx_scale = mlx_scale_shift[:, :, 1, :]  # (B, 1, inner_dim)
    mx.eval(mlx_shift, mlx_scale)

    with torch.no_grad():
        pt_scale_shift = pt_table[None, None].to(pt_final_ts.dtype) + pt_final_ts[:, :, None, :]
        pt_shift = pt_scale_shift[:, :, 0, :]
        pt_scale = pt_scale_shift[:, :, 1, :]

    compare_arrays("final shift", np.array(mlx_shift), pt_shift.detach().numpy())
    compare_arrays("final scale", np.array(mlx_scale), pt_scale.detach().numpy())

    # Apply final norm and scale/shift using model's norm_out
    mlx_normed = mlx_model.norm_out(mlx_x)
    mlx_out = mlx_normed * (1 + mlx_scale) + mlx_shift
    mx.eval(mlx_out)

    with torch.no_grad():
        pt_normed = pt_model.norm_out(pt_x)
        pt_out = pt_normed * (1 + pt_scale) + pt_shift

    compare_arrays("after final norm", np.array(mlx_out), pt_out.numpy())

    # Final linear projection
    mlx_final = mlx_model.proj_out(mlx_out)
    mx.eval(mlx_final)

    with torch.no_grad():
        pt_final = pt_model.proj_out(pt_out)

    compare_arrays("final output", np.array(mlx_final), pt_final.numpy())

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if first_divergence is not None:
        print(f"\nFirst divergence at block {first_divergence}")
    else:
        print("\nAll blocks match!")

    # Find where correlation drops significantly
    print("\nCorrelation trend (last 10 blocks):")
    for i, corr in divergence_corrs[-10:]:
        print(f"  Block {i}: {corr:.4f}")


if __name__ == "__main__":
    main()
