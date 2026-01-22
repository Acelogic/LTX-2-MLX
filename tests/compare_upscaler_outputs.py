"""
Visual comparison of PyTorch vs MLX Spatial Upscaler outputs.
Saves side-by-side images of the tensor outputs for direct comparison.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-MLX')
sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-Pytorch/packages/ltx-core/src')

import torch
import mlx.core as mx
from safetensors import safe_open


def load_pytorch_upsampler():
    """Load the PyTorch upsampler with weights."""
    from ltx_core.model.upsampler.model import LatentUpsampler

    weights_path = "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"

    model = LatentUpsampler(
        in_channels=128,
        mid_channels=1024,
        num_blocks_per_stage=4,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=True,
    )

    state_dict = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_mlx_upsampler():
    """Load the MLX upsampler with weights."""
    from LTX_2_MLX.model.upscaler.spatial import SpatialUpscaler, load_spatial_upscaler_weights

    weights_path = "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"

    model = SpatialUpscaler(
        in_channels=128,
        mid_channels=1024,
        num_blocks_per_stage=4,
        num_groups=32,
    )
    load_spatial_upscaler_weights(model, weights_path)

    return model


def visualize_comparison():
    """Create visual comparison of PyTorch and MLX outputs."""
    print("Loading models...")
    pt_model = load_pytorch_upsampler()
    mlx_model = load_mlx_upsampler()

    # Create test input (realistic latent)
    np.random.seed(42)
    test_np = np.random.randn(1, 128, 7, 16, 22).astype(np.float32) * 0.5

    pt_input = torch.from_numpy(test_np)
    mlx_input = mx.array(test_np)

    print(f"Input shape: {test_np.shape}")

    # Run both models
    print("Running PyTorch model...")
    with torch.no_grad():
        pt_output = pt_model(pt_input)

    print("Running MLX model...")
    mlx_output = mlx_model(mlx_input)
    mx.eval(mlx_output)

    # Convert to numpy
    pt_np = pt_output.numpy()
    mlx_np = np.array(mlx_output)
    diff_np = np.abs(pt_np - mlx_np)

    print(f"Output shape: {pt_np.shape}")
    print(f"\nPyTorch output stats:")
    print(f"  min={pt_np.min():.6f}, max={pt_np.max():.6f}")
    print(f"  mean={pt_np.mean():.6f}, std={pt_np.std():.6f}")
    print(f"\nMLX output stats:")
    print(f"  min={mlx_np.min():.6f}, max={mlx_np.max():.6f}")
    print(f"  mean={mlx_np.mean():.6f}, std={mlx_np.std():.6f}")
    print(f"\nDifference stats:")
    print(f"  max_diff={diff_np.max():.8f}")
    print(f"  mean_diff={diff_np.mean():.8f}")
    print(f"  std_diff={diff_np.std():.8f}")

    # Correlation
    correlation = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"\nCorrelation: {correlation:.10f}")

    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('PyTorch vs MLX Spatial Upscaler Output Comparison', fontsize=14)

    # Select middle frame and a few channels for visualization
    frame_idx = 3  # Middle frame
    channels_to_show = [0, 32, 64, 96]

    for col, ch in enumerate(channels_to_show):
        # PyTorch output
        pt_slice = pt_np[0, ch, frame_idx]
        ax = axes[0, col]
        im = ax.imshow(pt_slice, cmap='viridis')
        ax.set_title(f'PyTorch ch={ch}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # MLX output
        mlx_slice = mlx_np[0, ch, frame_idx]
        ax = axes[1, col]
        im = ax.imshow(mlx_slice, cmap='viridis')
        ax.set_title(f'MLX ch={ch}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Difference (scaled up for visibility)
        diff_slice = diff_np[0, ch, frame_idx]
        ax = axes[2, col]
        im = ax.imshow(diff_slice, cmap='hot')
        ax.set_title(f'Diff ch={ch}\nmax={diff_slice.max():.2e}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    axes[0, 0].set_ylabel('PyTorch', fontsize=12)
    axes[1, 0].set_ylabel('MLX', fontsize=12)
    axes[2, 0].set_ylabel('Difference', fontsize=12)

    plt.tight_layout()
    plt.savefig('upscaler_comparison.png', dpi=150)
    print(f"\nSaved visualization to upscaler_comparison.png")

    # Also create a scatter plot of PyTorch vs MLX values
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot (subsample for speed)
    subsample = 10000
    indices = np.random.choice(pt_np.size, subsample, replace=False)
    pt_flat = pt_np.flatten()[indices]
    mlx_flat = mlx_np.flatten()[indices]

    axes2[0].scatter(pt_flat, mlx_flat, alpha=0.3, s=1)
    axes2[0].plot([pt_flat.min(), pt_flat.max()], [pt_flat.min(), pt_flat.max()], 'r--', linewidth=2)
    axes2[0].set_xlabel('PyTorch Output')
    axes2[0].set_ylabel('MLX Output')
    axes2[0].set_title(f'Output Correlation: {correlation:.10f}')
    axes2[0].grid(True, alpha=0.3)

    # Histogram of differences
    axes2[1].hist(diff_np.flatten(), bins=100, alpha=0.7, color='blue')
    axes2[1].axvline(diff_np.mean(), color='red', linestyle='--', label=f'Mean: {diff_np.mean():.2e}')
    axes2[1].axvline(diff_np.max(), color='orange', linestyle='--', label=f'Max: {diff_np.max():.2e}')
    axes2[1].set_xlabel('Absolute Difference')
    axes2[1].set_ylabel('Count')
    axes2[1].set_title('Distribution of Differences')
    axes2[1].legend()
    axes2[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('upscaler_correlation.png', dpi=150)
    print(f"Saved correlation plot to upscaler_correlation.png")

    # Print sample values side by side
    print("\n" + "=" * 70)
    print("SAMPLE VALUES (first 20 values, flattened)")
    print("=" * 70)
    print(f"{'Index':<8} {'PyTorch':<15} {'MLX':<15} {'Diff':<15}")
    print("-" * 53)
    for i in range(20):
        pt_val = pt_np.flatten()[i]
        mlx_val = mlx_np.flatten()[i]
        diff_val = abs(pt_val - mlx_val)
        print(f"{i:<8} {pt_val:<15.8f} {mlx_val:<15.8f} {diff_val:<15.10f}")

    return pt_np, mlx_np


if __name__ == "__main__":
    pt_output, mlx_output = visualize_comparison()
