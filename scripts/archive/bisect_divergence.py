#!/usr/bin/env python3
"""Bisect where MLX diverges from PyTorch in the transformer.

This script compares intermediate outputs at each stage to find
the exact point of divergence.
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
    corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]

    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "✓" if close else "✗"

    print(f"  {name}: {status} max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, corr={corr:.4f}")
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
    }


def patchify_latent(latent):
    """Patchify latent from (B, C, F, H, W) to (B, F*H*W, C)."""
    B, C, F, H, W = latent.shape
    return latent.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)


def compare_patchify_proj(inputs, mlx_model, pt_model):
    """Compare patchify projection (input embedding)."""
    print("\n" + "=" * 60)
    print("Stage 1: Patchify Projection")
    print("=" * 60)

    latent_patchified = patchify_latent(inputs['latent'])

    # MLX
    mlx_input = mx.array(latent_patchified)
    mlx_output = mlx_model.patchify_proj(mlx_input)
    mx.eval(mlx_output)
    mlx_output_np = np.array(mlx_output)

    # PyTorch
    pt_input = torch.from_numpy(latent_patchified)
    with torch.no_grad():
        pt_output = pt_model.patchify_proj(pt_input)
    pt_output_np = pt_output.numpy()

    return compare_arrays("patchify_proj", mlx_output_np, pt_output_np)


def compare_caption_projection(inputs, mlx_model, pt_model):
    """Compare caption projection."""
    print("\n" + "=" * 60)
    print("Stage 2: Caption Projection")
    print("=" * 60)

    context = inputs['context']

    # MLX
    mlx_input = mx.array(context)
    mlx_output = mlx_model.caption_projection(mlx_input)
    mx.eval(mlx_output)
    mlx_output_np = np.array(mlx_output)

    # PyTorch
    pt_input = torch.from_numpy(context)
    with torch.no_grad():
        pt_output = pt_model.caption_projection(pt_input)
    pt_output_np = pt_output.numpy()

    return compare_arrays("caption_projection", mlx_output_np, pt_output_np)


def compare_adaln_embedding(inputs, mlx_model, pt_model):
    """Compare AdaLN timestep embedding."""
    print("\n" + "=" * 60)
    print("Stage 3: AdaLN Timestep Embedding")
    print("=" * 60)

    timestep = inputs['timestep']
    timestep_scale = 1000.0  # Default for LTX-2

    # MLX
    mlx_timestep = mx.array(timestep * timestep_scale)
    mlx_timestep_emb, mlx_embedded = mlx_model.adaln_single(mlx_timestep)
    mx.eval(mlx_timestep_emb, mlx_embedded)
    mlx_timestep_np = np.array(mlx_timestep_emb)
    mlx_embedded_np = np.array(mlx_embedded)

    # PyTorch
    pt_timestep = torch.tensor(timestep * timestep_scale)
    with torch.no_grad():
        pt_timestep_emb, pt_embedded = pt_model.adaln_single(pt_timestep.flatten(), hidden_dtype=torch.float32)
    pt_timestep_np = pt_timestep_emb.numpy()
    pt_embedded_np = pt_embedded.numpy()

    match1 = compare_arrays("adaln_timestep", mlx_timestep_np, pt_timestep_np)
    match2 = compare_arrays("adaln_embedded", mlx_embedded_np, pt_embedded_np)
    return match1 and match2


def compare_scale_shift_table(mlx_model, pt_model):
    """Compare scale_shift_table values."""
    print("\n" + "=" * 60)
    print("Stage 4: Scale-Shift Table")
    print("=" * 60)

    mlx_table = np.array(mlx_model.scale_shift_table)
    pt_table = pt_model.scale_shift_table.detach().numpy()

    return compare_arrays("scale_shift_table", mlx_table, pt_table)


def compare_proj_out(mlx_model, pt_model):
    """Compare proj_out weights."""
    print("\n" + "=" * 60)
    print("Stage 5: Proj Out Weights")
    print("=" * 60)

    mlx_weight = np.array(mlx_model.proj_out.weight)
    pt_weight = pt_model.proj_out.weight.detach().numpy()

    print(f"  MLX proj_out weight shape: {mlx_weight.shape}")
    print(f"  PT proj_out weight shape: {pt_weight.shape}")

    # MLX stores weights as (out, in), PyTorch as (out, in)
    return compare_arrays("proj_out.weight", mlx_weight, pt_weight)


def compare_first_block_weights(mlx_model, pt_model):
    """Compare first transformer block weights."""
    print("\n" + "=" * 60)
    print("Stage 6: First Transformer Block Weights")
    print("=" * 60)

    mlx_block = mlx_model.transformer_blocks[0]
    pt_block = pt_model.transformer_blocks[0]

    all_match = True

    # Self-attention Q, K, V (for VideoOnly model, attrs are directly on block)
    for name in ['to_q', 'to_k', 'to_v']:
        mlx_w = np.array(getattr(mlx_block.attn1, name).weight)
        pt_w = getattr(pt_block.attn1, name).weight.detach().numpy()
        if not compare_arrays(f"attn1.{name}.weight", mlx_w, pt_w):
            all_match = False

    # Cross-attention Q, K, V
    for name in ['to_q', 'to_k', 'to_v']:
        mlx_w = np.array(getattr(mlx_block.attn2, name).weight)
        pt_w = getattr(pt_block.attn2, name).weight.detach().numpy()
        if not compare_arrays(f"attn2.{name}.weight", mlx_w, pt_w):
            all_match = False

    # Scale-shift table for block
    mlx_sst = np.array(mlx_block.scale_shift_table)
    pt_sst = pt_block.scale_shift_table.detach().numpy()
    if not compare_arrays("block0.scale_shift_table", mlx_sst, pt_sst):
        all_match = False

    return all_match


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 60)
    print("Bisecting MLX vs PyTorch Divergence")
    print("=" * 60)

    if not Path(weights_path).exists():
        print(f"Weights not found: {weights_path}")
        return

    # Create test inputs
    inputs = create_test_inputs()

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder

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

    # Compare each stage
    results = {}

    results['patchify_proj'] = compare_patchify_proj(inputs, mlx_model, pt_model)
    results['caption_projection'] = compare_caption_projection(inputs, mlx_model, pt_model)
    results['adaln'] = compare_adaln_embedding(inputs, mlx_model, pt_model)
    results['scale_shift_table'] = compare_scale_shift_table(mlx_model, pt_model)
    results['proj_out'] = compare_proj_out(mlx_model, pt_model)
    results['block0_weights'] = compare_first_block_weights(mlx_model, pt_model)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for stage, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {stage}: {status}")

    # Find first failure
    for stage, passed in results.items():
        if not passed:
            print(f"\n>>> First divergence found at: {stage}")
            break
    else:
        print("\n>>> All stages match - divergence must be in forward pass logic")


if __name__ == "__main__":
    main()
