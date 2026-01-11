#!/usr/bin/env python3
"""Compare MLX and PyTorch LTX-2 outputs to find divergence.

This script requires the PyTorch LTX-2 package to be installed.
Clone from: https://github.com/Lightricks/LTX-2
Install with: cd LTX-2 && uv sync --frozen

Run this to identify where MLX diverges from PyTorch.
"""

import sys
from pathlib import Path

# Mock triton before any imports (triton is only used for CUDA FP8 LoRA fusion)
# This is safe because we're on CPU and not using FP8 weights
import types
mock_triton = types.ModuleType('triton')
mock_triton.cdiv = lambda a, b: (a + b - 1) // b
mock_triton.jit = lambda fn: fn  # No-op decorator

mock_triton_language = types.ModuleType('triton.language')
mock_triton_language.constexpr = int  # Dummy type
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
import mlx.core as mx


def check_pytorch_available():
    """Check if PyTorch LTX packages are available."""
    try:
        import torch
        from ltx_core.model.transformer import LTXModel
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def create_test_inputs(batch_size=1, latent_channels=128, frames=3, height=8, width=12,
                       context_dim=3840, context_len=256):
    """Create identical test inputs for MLX and PyTorch comparison.

    Creates unpatchified latent (B, C, F, H, W) that will be patchified by each model.
    """
    np.random.seed(42)

    # Create unpatchified latent
    latent = np.random.randn(batch_size, latent_channels, frames, height, width).astype(np.float32) * 0.1

    # Context embedding
    context = np.random.randn(batch_size, context_len, context_dim).astype(np.float32) * 0.1

    # Timestep (sigma)
    timestep = np.array([1.0], dtype=np.float32)  # sigma=1.0

    # Position indices - (B, 3, num_patches) format for PyTorch
    # num_patches = frames * height * width
    t_coords = np.arange(frames)
    h_coords = np.arange(height)
    w_coords = np.arange(width)

    # Create meshgrid with 'ij' indexing (frames, height, width)
    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")

    positions = np.stack([
        t_grid.flatten(),
        h_grid.flatten(),
        w_grid.flatten(),
    ], axis=0)[None].astype(np.float32)  # (1, 3, num_patches)

    return {
        'latent': latent,
        'context': context,
        'timestep': timestep,
        'positions': positions,
        'frames': frames,
        'height': height,
        'width': width,
    }


def patchify_latent(latent, patch_size=1):
    """Patchify latent from (B, C, F, H, W) to (B, num_patches, C*patch_size^3).

    For patch_size=1, this is just a reshape to (B, F*H*W, C).
    """
    B, C, F, H, W = latent.shape
    # Reshape to (B, F*H*W, C)
    return latent.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)


def run_mlx_forward(weights_path, inputs):
    """Run MLX forward pass and return output."""
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality
    from LTX_2_MLX.loader import load_transformer_weights

    print("Creating MLX model...")
    model = LTXModel(
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

    print("Loading MLX weights...")
    load_transformer_weights(model, weights_path)

    # Patchify latent
    latent_patchified = patchify_latent(inputs['latent'])

    # Create MLX arrays
    x = mx.array(latent_patchified)
    context = mx.array(inputs['context'])
    timestep = mx.array(inputs['timestep'])
    positions = mx.array(inputs['positions'])

    # MLX expects positions with bounds: (B, 3, T, 2) where last dim is [start, end]
    positions_with_bounds = mx.stack([positions, positions + 1], axis=-1)

    # Apply temporal scaling (fps=24)
    fps = 24.0
    temporal = positions_with_bounds[:, 0:1, :, :] / fps
    spatial = positions_with_bounds[:, 1:, :, :]
    positions_full = mx.concatenate([temporal, spatial], axis=1)

    # Attention mask (all ones = attend to everything)
    context_mask = mx.ones((1, context.shape[1]))

    modality = Modality(
        latent=x,
        context=context,
        context_mask=context_mask,
        timesteps=timestep,
        positions=positions_full,
        enabled=True,
    )

    print("Running MLX forward...")
    output = model(modality)
    mx.eval(output)

    return np.array(output)


def run_pytorch_forward(weights_path, inputs):
    """Run PyTorch forward pass and return output."""
    import torch
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import LTXVideoOnlyModelConfigurator
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    print("Creating PyTorch model...")

    # Build model using the official loader
    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )

    # Use CPU for comparison (no CUDA on Mac)
    device = torch.device("cpu")
    model = builder.build(device=device, dtype=torch.float32)
    model.eval()

    print(f"PyTorch model loaded on {device}")

    # Patchify latent
    latent_patchified = patchify_latent(inputs['latent'])

    # Create PyTorch tensors
    latent = torch.from_numpy(latent_patchified).to(device)
    context = torch.from_numpy(inputs['context']).to(device)

    # PyTorch expects positions as (B, 3, T, 2) where last dim is [start, end]
    # For use_middle_indices_grid=True, it computes (start + end) / 2
    positions_3d = torch.from_numpy(inputs['positions']).to(device)  # (1, 3, T)
    positions = torch.stack([positions_3d, positions_3d + 1], dim=-1)  # (1, 3, T, 2)

    # PyTorch expects timesteps as (B, T) where T is number of tokens
    # For uniform sigma across all tokens, broadcast
    num_tokens = latent.shape[1]
    timesteps = torch.full((1, num_tokens), inputs['timestep'][0], device=device)

    # Attention mask
    context_mask = torch.ones((1, context.shape[1]), device=device)

    # Create Modality
    modality = Modality(
        latent=latent,
        context=context,
        context_mask=context_mask,
        timesteps=timesteps,
        positions=positions,
        enabled=True,
    )

    # Create empty perturbation config (no perturbations)
    perturbations = BatchedPerturbationConfig(perturbations=[])

    print("Running PyTorch forward...")
    with torch.no_grad():
        video_output, audio_output = model(video=modality, audio=None, perturbations=perturbations)

    return video_output.cpu().numpy()


def compare_values(name, mlx_val, pt_val, rtol=1e-3, atol=1e-4):
    """Compare MLX and PyTorch values."""
    if mlx_val.shape != pt_val.shape:
        print(f"  {name}: SHAPE MISMATCH - MLX {mlx_val.shape} vs PT {pt_val.shape}")
        return False

    abs_diff = np.abs(mlx_val - pt_val)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    # Avoid division by zero
    pt_abs = np.abs(pt_val)
    relative_diff = np.where(pt_abs > 1e-8, abs_diff / pt_abs, 0)
    max_rel_diff = relative_diff.max()

    close = np.allclose(mlx_val, pt_val, rtol=rtol, atol=atol)

    status = "✓ MATCH" if close else "✗ DIFFER"
    print(f"  {name}: {status}")
    print(f"    Max abs diff: {max_diff:.6f}")
    print(f"    Mean abs diff: {mean_diff:.6f}")
    print(f"    Max rel diff: {max_rel_diff:.6f}")

    # Show sample values
    print(f"    MLX first 5: {mlx_val.flatten()[:5]}")
    print(f"    PT first 5:  {pt_val.flatten()[:5]}")

    # Correlation
    corr = np.corrcoef(mlx_val.flatten(), pt_val.flatten())[0, 1]
    print(f"    Correlation: {corr:.6f}")

    return close


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 60)
    print("MLX vs PyTorch LTX-2 Comparison")
    print("=" * 60)

    if not Path(weights_path).exists():
        print(f"Weights not found: {weights_path}")
        return

    # Create test inputs
    print("\nCreating test inputs...")
    inputs = create_test_inputs()
    print(f"  latent shape: {inputs['latent'].shape}")
    print(f"  context shape: {inputs['context'].shape}")
    print(f"  timestep: {inputs['timestep']}")
    print(f"  positions shape: {inputs['positions'].shape}")

    # Check if PyTorch LTX is available
    print("\n" + "=" * 60)
    print("Checking PyTorch availability...")
    print("=" * 60)

    if not check_pytorch_available():
        print("PyTorch LTX-2 not installed or not in path.")
        print("Make sure LTX-2-PyTorch is cloned to the parent directory.")
        print("Clone: git clone https://github.com/Lightricks/LTX-2.git ../LTX-2-PyTorch")
        print("Install: cd ../LTX-2-PyTorch && uv sync --frozen")

        # Run MLX only and save for later comparison
        print("\n" + "=" * 60)
        print("Running MLX only...")
        print("=" * 60)
        mlx_output = run_mlx_forward(weights_path, inputs)
        print(f"\nMLX output shape: {mlx_output.shape}")
        print(f"MLX output stats: mean={mlx_output.mean():.6f}, std={mlx_output.std():.6f}")
        print(f"MLX output range: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]")

        # Save for external comparison
        output_path = "gens/mlx_transformer_output.npz"
        Path("gens").mkdir(exist_ok=True)
        np.savez(output_path,
                 output=mlx_output,
                 **inputs)
        print(f"\nSaved MLX output to {output_path}")
        return

    # Run both and compare
    print("\n" + "=" * 60)
    print("Running PyTorch forward pass...")
    print("=" * 60)
    try:
        pt_output = run_pytorch_forward(weights_path, inputs)
        print(f"\nPyTorch output shape: {pt_output.shape}")
        print(f"PyTorch output stats: mean={pt_output.mean():.6f}, std={pt_output.std():.6f}")
        print(f"PyTorch output range: [{pt_output.min():.6f}, {pt_output.max():.6f}]")
    except Exception as e:
        print(f"PyTorch forward failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Running MLX forward pass...")
    print("=" * 60)
    mlx_output = run_mlx_forward(weights_path, inputs)
    print(f"\nMLX output shape: {mlx_output.shape}")
    print(f"MLX output stats: mean={mlx_output.mean():.6f}, std={mlx_output.std():.6f}")
    print(f"MLX output range: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]")

    # Compare outputs
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)

    match = compare_values("Transformer Output", mlx_output, pt_output, rtol=0.01, atol=0.01)

    if match:
        print("\n✓ MLX and PyTorch outputs MATCH!")
        print("  The issue is likely in the denoising loop or VAE, not the transformer.")
    else:
        print("\n✗ MLX and PyTorch outputs DIFFER!")
        print("  The transformer implementation has divergence.")
        print("  Next step: Bisect to find which layer/component diverges.")

        # Save both for detailed analysis
        output_path = "gens/comparison_outputs.npz"
        np.savez(output_path,
                 mlx_output=mlx_output,
                 pt_output=pt_output,
                 **inputs)
        print(f"\nSaved comparison outputs to {output_path}")


if __name__ == "__main__":
    main()
