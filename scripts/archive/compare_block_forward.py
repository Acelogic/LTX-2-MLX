#!/usr/bin/env python3
"""Compare single transformer block forward pass between MLX and PyTorch.

This script traces through a single block and compares intermediate values
to find exactly where the computation diverges.
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

# Add dtype class to prevent torch._dynamo error
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

    # Handle potential NaN in correlation
    try:
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    except:
        corr = float('nan')

    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "✓" if close else "✗"

    print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.4f}")
    if not close:
        print(f"    MLX: mean={a.mean():.6f}, std={a.std():.6f}, range=[{a.min():.4f}, {a.max():.4f}]")
        print(f"    PT:  mean={b.mean():.6f}, std={b.std():.6f}, range=[{b.min():.4f}, {b.max():.4f}]")

    return close


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
        'frames': frames,
        'height': height,
        'width': width,
    }


def patchify_latent(latent):
    """Patchify latent from (B, C, F, H, W) to (B, F*H*W, C)."""
    B, C, F, H, W = latent.shape
    return latent.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 60)
    print("Block Forward Pass Comparison")
    print("=" * 60)

    if not Path(weights_path).exists():
        print(f"Weights not found: {weights_path}")
        return

    # Create test inputs
    inputs = create_test_inputs()
    latent_patchified = patchify_latent(inputs['latent'])

    # ================================================================
    # Load PyTorch model
    # ================================================================
    print("\nLoading PyTorch model...")
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.rope import precompute_freqs_cis, generate_freq_grid_np
    from ltx_core.utils import rms_norm as pt_rms_norm

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pt_model = builder.build(device=torch.device("cpu"), dtype=torch.float32)
    pt_model.eval()

    # ================================================================
    # Load MLX model
    # ================================================================
    print("Loading MLX model...")
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.model.transformer.attention import rms_norm as mlx_rms_norm
    from LTX_2_MLX.model.transformer.rope import precompute_freqs_cis as mlx_precompute_freqs_cis

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

    # ================================================================
    # Compare: patchify_proj output
    # ================================================================
    print("\n" + "=" * 60)
    print("Stage 1: After patchify_proj")
    print("=" * 60)

    mlx_x = mx.array(latent_patchified)
    mlx_x = mlx_model.patchify_proj(mlx_x)
    mx.eval(mlx_x)

    pt_x = torch.from_numpy(latent_patchified)
    with torch.no_grad():
        pt_x = pt_model.patchify_proj(pt_x)

    compare_arrays("x after patchify", np.array(mlx_x), pt_x.numpy())

    # ================================================================
    # Compare: caption_projection output
    # ================================================================
    print("\n" + "=" * 60)
    print("Stage 2: After caption_projection")
    print("=" * 60)

    mlx_context = mx.array(inputs['context'])
    mlx_context = mlx_model.caption_projection(mlx_context)
    mlx_context = mlx_context.reshape(1, -1, mlx_x.shape[-1])  # Match PyTorch shape
    mx.eval(mlx_context)

    pt_context = torch.from_numpy(inputs['context'])
    with torch.no_grad():
        pt_context = pt_model.caption_projection(pt_context)
        pt_context = pt_context.view(1, -1, pt_x.shape[-1])

    compare_arrays("context after projection", np.array(mlx_context), pt_context.numpy())

    # ================================================================
    # Compare: timestep embedding
    # ================================================================
    print("\n" + "=" * 60)
    print("Stage 3: Timestep embedding")
    print("=" * 60)

    timestep_scale = 1000.0

    mlx_timestep = mx.array(inputs['timestep'] * timestep_scale)
    mlx_ts_emb, mlx_embedded = mlx_model.adaln_single(mlx_timestep)
    mx.eval(mlx_ts_emb, mlx_embedded)
    # Shape: (B, 6, D)
    mlx_ts_emb_np = np.array(mlx_ts_emb)

    with torch.no_grad():
        pt_timestep = torch.tensor(inputs['timestep'] * timestep_scale)
        pt_ts_emb, pt_embedded = pt_model.adaln_single(pt_timestep.flatten(), hidden_dtype=torch.float32)
    pt_ts_emb_np = pt_ts_emb.numpy()

    compare_arrays("timestep embedding", mlx_ts_emb_np, pt_ts_emb_np)

    # ================================================================
    # Compare: RoPE computation
    # ================================================================
    print("\n" + "=" * 60)
    print("Stage 4: RoPE position embeddings")
    print("=" * 60)

    # MLX expects positions with bounds: (B, 3, T, 2)
    positions_3d = mx.array(inputs['positions'])
    positions_with_bounds = mx.stack([positions_3d, positions_3d + 1], axis=-1)

    # Apply temporal scaling (fps=24)
    fps = 24.0
    temporal = positions_with_bounds[:, 0:1, :, :] / fps
    spatial = positions_with_bounds[:, 1:, :, :]
    mlx_positions = mx.concatenate([temporal, spatial], axis=1)

    # Compute MLX RoPE
    from LTX_2_MLX.model.transformer.rope import LTXRopeType
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
    mlx_cos, mlx_sin = np.array(mlx_rope[0]), np.array(mlx_rope[1])

    # Compute PyTorch RoPE
    from ltx_core.model.transformer.rope import LTXRopeType as PTRopeType

    pt_positions_3d = torch.from_numpy(inputs['positions'])
    pt_positions = torch.stack([pt_positions_3d, pt_positions_3d + 1], dim=-1)
    # Apply temporal scaling
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
    pt_cos, pt_sin = pt_rope[0].numpy(), pt_rope[1].numpy()

    compare_arrays("RoPE cos", mlx_cos, pt_cos)
    compare_arrays("RoPE sin", mlx_sin, pt_sin)

    # ================================================================
    # Compare: Block 0 forward pass (step by step)
    # ================================================================
    print("\n" + "=" * 60)
    print("Stage 5: Block 0 Forward Pass (detailed)")
    print("=" * 60)

    mlx_block = mlx_model.transformer_blocks[0]
    pt_block = pt_model.transformer_blocks[0]

    # Get fresh copies of x after patchify
    mlx_x = mx.array(latent_patchified)
    mlx_x = mlx_model.patchify_proj(mlx_x)
    mx.eval(mlx_x)

    pt_x = torch.from_numpy(latent_patchified)
    with torch.no_grad():
        pt_x = pt_model.patchify_proj(pt_x)

    # Prepare timesteps for transformer blocks
    # MLX expects (B, T, 6, D) format
    num_tokens = mlx_x.shape[1]
    mlx_ts = mlx_ts_emb.reshape(1, 1, 6, -1)
    mlx_ts = mx.broadcast_to(mlx_ts, (1, num_tokens, 6, mlx_ts.shape[-1]))
    mx.eval(mlx_ts)

    # PyTorch expects (B, T, 6*D) which gets split
    pt_ts = pt_ts_emb.view(1, 1, -1).expand(1, num_tokens, -1)
    pt_ts = pt_ts.view(1, num_tokens, 6, -1)

    print(f"\n  Input x shape: MLX={mlx_x.shape}, PT={pt_x.shape}")
    print(f"  Timestep shape: MLX={mlx_ts.shape}, PT={pt_ts.shape}")

    # --- Compare RMSNorm ---
    print("\n  --- RMSNorm before self-attention ---")
    mlx_norm = mlx_rms_norm(mlx_x, eps=1e-6)
    mx.eval(mlx_norm)

    with torch.no_grad():
        pt_norm = pt_rms_norm(pt_x, eps=1e-6)

    compare_arrays("RMSNorm(x)", np.array(mlx_norm), pt_norm.numpy())

    # --- Compare scale_shift_table slice ---
    print("\n  --- AdaLN scale/shift/gate (first 3 values) ---")

    mlx_table = np.array(mlx_block.scale_shift_table)
    pt_table = pt_block.scale_shift_table.detach().numpy()

    compare_arrays("scale_shift_table[0:3]", mlx_table[0:3], pt_table[0:3])

    # --- Compare computed AdaLN values ---
    # MLX: table_slice[None, None, :, :] + timestep[:, :, start:end, :]
    mlx_table_slice = mx.array(mlx_table[0:3])
    mlx_ada = mlx_table_slice[None, None, :, :] + mlx_ts[:, :, 0:3, :]
    mx.eval(mlx_ada)
    mlx_shift = np.array(mlx_ada[:, :, 0, :])
    mlx_scale = np.array(mlx_ada[:, :, 1, :])
    mlx_gate = np.array(mlx_ada[:, :, 2, :])

    # PyTorch: scale_shift_table[indices].unsqueeze(0).unsqueeze(0) + timestep.reshape(batch_size, timestep.shape[1], 6, -1)[:, :, indices, :]
    with torch.no_grad():
        pt_table_slice = pt_block.scale_shift_table[0:3].unsqueeze(0).unsqueeze(0)
        pt_ts_reshaped = pt_ts
        pt_ada = pt_table_slice + pt_ts_reshaped[:, :, 0:3, :]
    pt_shift = pt_ada[:, :, 0, :].numpy()
    pt_scale = pt_ada[:, :, 1, :].numpy()
    pt_gate = pt_ada[:, :, 2, :].numpy()

    compare_arrays("shift_msa", mlx_shift, pt_shift)
    compare_arrays("scale_msa", mlx_scale, pt_scale)
    compare_arrays("gate_msa", mlx_gate, pt_gate)

    # --- Compare scaled norm ---
    print("\n  --- After AdaLN modulation ---")
    mlx_norm_scaled = mlx_rms_norm(mlx_x, eps=1e-6) * (1 + mx.array(mlx_scale)) + mx.array(mlx_shift)
    mx.eval(mlx_norm_scaled)

    with torch.no_grad():
        pt_norm_scaled = pt_rms_norm(pt_x, eps=1e-6) * (1 + torch.tensor(pt_scale)) + torch.tensor(pt_shift)

    compare_arrays("norm_x (after AdaLN)", np.array(mlx_norm_scaled), pt_norm_scaled.numpy())

    # --- Compare Q, K, V projections ---
    print("\n  --- Q, K, V projections (self-attention) ---")

    mlx_q = mlx_block.attn1.to_q(mlx_norm_scaled)
    mlx_k = mlx_block.attn1.to_k(mlx_norm_scaled)
    mlx_v = mlx_block.attn1.to_v(mlx_norm_scaled)
    mx.eval(mlx_q, mlx_k, mlx_v)

    with torch.no_grad():
        pt_q = pt_block.attn1.to_q(pt_norm_scaled)
        pt_k = pt_block.attn1.to_k(pt_norm_scaled)
        pt_v = pt_block.attn1.to_v(pt_norm_scaled)

    compare_arrays("Q (before norm)", np.array(mlx_q), pt_q.numpy())
    compare_arrays("K (before norm)", np.array(mlx_k), pt_k.numpy())
    compare_arrays("V", np.array(mlx_v), pt_v.numpy())

    # --- Compare Q, K after QK norm ---
    print("\n  --- Q, K after QK RMSNorm ---")

    mlx_q_norm = mlx_block.attn1.q_norm(mlx_q)
    mlx_k_norm = mlx_block.attn1.k_norm(mlx_k)
    mx.eval(mlx_q_norm, mlx_k_norm)

    with torch.no_grad():
        pt_q_norm = pt_block.attn1.q_norm(pt_q)
        pt_k_norm = pt_block.attn1.k_norm(pt_k)

    compare_arrays("Q (after norm)", np.array(mlx_q_norm), pt_q_norm.numpy())
    compare_arrays("K (after norm)", np.array(mlx_k_norm), pt_k_norm.numpy())

    # --- Compare Q, K after RoPE ---
    print("\n  --- Q, K after RoPE ---")

    from LTX_2_MLX.model.transformer.rope import apply_rotary_emb as mlx_apply_rope
    from ltx_core.model.transformer.rope import apply_rotary_emb as pt_apply_rope

    mlx_q_rope = mlx_apply_rope(mlx_q_norm, mlx_rope, LTXRopeType.SPLIT)
    mlx_k_rope = mlx_apply_rope(mlx_k_norm, mlx_rope, LTXRopeType.SPLIT)
    mx.eval(mlx_q_rope, mlx_k_rope)

    with torch.no_grad():
        pt_q_rope = pt_apply_rope(pt_q_norm, pt_rope, PTRopeType.SPLIT)
        pt_k_rope = pt_apply_rope(pt_k_norm, pt_rope, PTRopeType.SPLIT)

    compare_arrays("Q (after RoPE)", np.array(mlx_q_rope), pt_q_rope.numpy())
    compare_arrays("K (after RoPE)", np.array(mlx_k_rope), pt_k_rope.numpy())

    # --- Compare attention output ---
    print("\n  --- Self-attention output ---")

    # Compute attention manually to avoid double RoPE application
    # Reshape for multi-head attention: (B, T, H*D) -> (B, H, T, D)
    heads = 32
    dim_head = 128
    b, t_q, _ = mlx_q_rope.shape

    mlx_q_mh = mlx_q_rope.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)
    mlx_k_mh = mlx_k_rope.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)
    mlx_v_mh = mlx_v.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scale = 1.0 / (dim_head ** 0.5)
    mlx_scores = mx.matmul(mlx_q_mh, mlx_k_mh.transpose(0, 1, 3, 2)) * scale
    mlx_weights = mx.softmax(mlx_scores, axis=-1)
    mlx_attn = mx.matmul(mlx_weights, mlx_v_mh)
    mlx_attn = mlx_attn.transpose(0, 2, 1, 3).reshape(b, t_q, heads * dim_head)
    mlx_attn_out = mlx_block.attn1.to_out(mlx_attn)
    mx.eval(mlx_attn_out)

    # PyTorch attention
    pt_q_mh = pt_q_rope.view(b, t_q, heads, dim_head).transpose(1, 2)
    pt_k_mh = pt_k_rope.view(b, t_q, heads, dim_head).transpose(1, 2)
    pt_v_mh = pt_v.view(b, t_q, heads, dim_head).transpose(1, 2)

    with torch.no_grad():
        pt_attn_raw = torch.nn.functional.scaled_dot_product_attention(
            pt_q_mh, pt_k_mh, pt_v_mh, dropout_p=0.0, is_causal=False
        )
        pt_attn = pt_attn_raw.transpose(1, 2).reshape(b, t_q, heads * dim_head)
        pt_attn_out = pt_block.attn1.to_out(pt_attn)

    compare_arrays("attn1 output", np.array(mlx_attn_out), pt_attn_out.numpy())

    # --- Compare gated attention output ---
    print("\n  --- Gated attention residual ---")

    mlx_x_after_attn1 = mlx_x + mlx_attn_out * mx.array(mlx_gate)
    mx.eval(mlx_x_after_attn1)

    with torch.no_grad():
        pt_x_after_attn1 = pt_x + pt_attn_out * torch.tensor(pt_gate)

    compare_arrays("x after attn1", np.array(mlx_x_after_attn1), pt_x_after_attn1.numpy())

    # --- Compare cross-attention ---
    print("\n  --- Cross-attention ---")

    mlx_cross_out = mlx_block.attn2(
        mlx_rms_norm(mlx_x_after_attn1, eps=1e-6),
        context=mlx_context,
        mask=None,
    )
    mx.eval(mlx_cross_out)

    with torch.no_grad():
        pt_cross_out = pt_block.attn2(
            pt_rms_norm(pt_x_after_attn1, eps=1e-6),
            context=pt_context,
            mask=None,
        )

    compare_arrays("attn2 (cross) output", np.array(mlx_cross_out), pt_cross_out.numpy())

    mlx_x_after_attn2 = mlx_x_after_attn1 + mlx_cross_out
    mx.eval(mlx_x_after_attn2)

    with torch.no_grad():
        pt_x_after_attn2 = pt_x_after_attn1 + pt_cross_out

    compare_arrays("x after attn2", np.array(mlx_x_after_attn2), pt_x_after_attn2.numpy())

    # --- Compare FFN ---
    print("\n  --- Feed-forward network ---")

    # Get FFN AdaLN values
    mlx_ff_ada = mlx_table_slice[None, None, :, :] + mlx_ts[:, :, 3:6, :]
    mx.eval(mlx_ff_ada)
    # Wait, need to use indices 3:6
    mlx_ff_table = mx.array(mlx_table[3:6])
    mlx_ff_ada = mlx_ff_table[None, None, :, :] + mlx_ts[:, :, 3:6, :]
    mx.eval(mlx_ff_ada)
    mlx_ff_shift = mx.array(mlx_ff_ada[:, :, 0, :])
    mlx_ff_scale = mx.array(mlx_ff_ada[:, :, 1, :])
    mlx_ff_gate = mx.array(mlx_ff_ada[:, :, 2, :])
    mx.eval(mlx_ff_shift, mlx_ff_scale, mlx_ff_gate)

    with torch.no_grad():
        pt_ff_table = pt_block.scale_shift_table[3:6].unsqueeze(0).unsqueeze(0)
        pt_ff_ada = pt_ff_table + pt_ts[:, :, 3:6, :]
        pt_ff_shift = pt_ff_ada[:, :, 0, :]
        pt_ff_scale = pt_ff_ada[:, :, 1, :]
        pt_ff_gate = pt_ff_ada[:, :, 2, :]

    mlx_ff_scaled = mlx_rms_norm(mlx_x_after_attn2, eps=1e-6) * (1 + mlx_ff_scale) + mlx_ff_shift
    mx.eval(mlx_ff_scaled)

    with torch.no_grad():
        pt_ff_scaled = pt_rms_norm(pt_x_after_attn2, eps=1e-6) * (1 + pt_ff_scale) + pt_ff_shift

    compare_arrays("FFN input (scaled)", np.array(mlx_ff_scaled), pt_ff_scaled.numpy())

    mlx_ff_out = mlx_block.ff(mlx_ff_scaled)
    mx.eval(mlx_ff_out)

    with torch.no_grad():
        pt_ff_out = pt_block.ff(pt_ff_scaled)

    compare_arrays("FFN output", np.array(mlx_ff_out), pt_ff_out.numpy())

    mlx_x_after_ff = mlx_x_after_attn2 + mlx_ff_out * mlx_ff_gate
    mx.eval(mlx_x_after_ff)

    with torch.no_grad():
        pt_x_after_ff = pt_x_after_attn2 + pt_ff_out * pt_ff_gate

    compare_arrays("x after FFN (block output)", np.array(mlx_x_after_ff), pt_x_after_ff.numpy())

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary: Find first divergence point")
    print("=" * 60)
    print("If all above match but full transformer differs,")
    print("the issue is in later blocks or final projection.")


if __name__ == "__main__":
    main()
