#!/usr/bin/env python3
"""
Validate MLX VAE decoder operations against PyTorch reference.

Compares individual operations (conv3d, depth-to-space, pixel_norm) to identify discrepancies.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    Conv3dSimple,
    _pixel_norm,
)


def compare_tensors(name: str, mlx_arr: mx.array, pt_arr: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-3):
    """Compare MLX and PyTorch tensors."""
    mlx_np = np.array(mlx_arr)
    pt_np = pt_arr.detach().cpu().numpy()

    # Check shapes match
    if mlx_np.shape != pt_np.shape:
        print(f"\n{name}: SHAPE MISMATCH - MLX={mlx_np.shape}, PT={pt_np.shape}")
        return False

    # Difference metrics
    abs_diff = np.abs(mlx_np - pt_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    # Check if close
    is_close = np.allclose(mlx_np, pt_np, rtol=rtol, atol=atol)

    status = "✓" if is_close else "✗"
    print(f"{status} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"mlx_mean={mlx_np.mean():.4f}, pt_mean={pt_np.mean():.4f}")

    return is_close


def validate_conv3d(weights_path: str, seed: int = 42):
    """Validate 3D convolution operation."""
    print("\n" + "=" * 60)
    print("Validating Conv3D Operation")
    print("=" * 60)

    # Load a conv weight from the checkpoint
    with safe_open(weights_path, framework="pt") as f:
        pt_weight = f.get_tensor("vae.decoder.conv_in.conv.weight").float()
        pt_bias = f.get_tensor("vae.decoder.conv_in.conv.bias").float()

    print(f"Weight shape: {pt_weight.shape}")  # (1024, 128, 3, 3, 3)

    # Create test input
    np.random.seed(seed)
    input_np = np.random.randn(1, 128, 5, 8, 8).astype(np.float32)

    # MLX conv
    mlx_conv = Conv3dSimple(128, 1024, kernel_size=3)
    mlx_conv.weight = mx.array(pt_weight.numpy())
    mlx_conv.bias = mx.array(pt_bias.numpy())

    mlx_input = mx.array(input_np)
    mlx_output = mlx_conv(mlx_input, causal=True)
    mx.eval(mlx_output)

    # PyTorch conv with causal padding
    pt_input = torch.from_numpy(input_np)
    # Causal temporal padding: replicate first frame
    pt_input_padded = torch.cat([
        pt_input[:, :, :1].repeat(1, 1, 2, 1, 1),  # 2 frames of padding
        pt_input
    ], dim=2)
    # Spatial padding
    pt_input_padded = F.pad(pt_input_padded, (1, 1, 1, 1, 0, 0))  # H, W padding

    pt_conv = torch.nn.Conv3d(128, 1024, kernel_size=3, padding=0)
    pt_conv.weight.data = pt_weight
    pt_conv.bias.data = pt_bias

    with torch.no_grad():
        pt_output = pt_conv(pt_input_padded)

    result = compare_tensors("conv3d_output", mlx_output, pt_output)
    return result


def validate_pixel_norm(seed: int = 42):
    """Validate pixel normalization."""
    print("\n" + "=" * 60)
    print("Validating Pixel Norm")
    print("=" * 60)

    np.random.seed(seed)
    input_np = np.random.randn(1, 128, 5, 8, 8).astype(np.float32)

    # MLX pixel norm
    mlx_input = mx.array(input_np)
    mlx_output = _pixel_norm(mlx_input)
    mx.eval(mlx_output)

    # PyTorch pixel norm
    pt_input = torch.from_numpy(input_np)
    variance = torch.mean(pt_input ** 2, dim=1, keepdim=True)
    pt_output = pt_input * torch.rsqrt(variance + 1e-6)

    result = compare_tensors("pixel_norm", mlx_output, pt_output)
    return result


def validate_depth_to_space(seed: int = 42):
    """Validate depth-to-space (pixel shuffle) operation."""
    print("\n" + "=" * 60)
    print("Validating Depth-to-Space")
    print("=" * 60)

    np.random.seed(seed)
    # Input with channels divisible by 8 (factor 2x2x2)
    input_np = np.random.randn(1, 512, 3, 8, 8).astype(np.float32)  # 512 = 64 * 8

    # MLX depth-to-space
    from LTX_2_MLX.model.video_vae.simple_decoder import DepthToSpaceUpsample3d
    mlx_d2s = DepthToSpaceUpsample3d(1024, factor=(2, 2, 2), residual=False)

    mlx_input = mx.array(input_np)
    # Just test the _depth_to_space method
    mlx_output = mlx_d2s._depth_to_space(mlx_input, c_out=64)
    mx.eval(mlx_output)

    # PyTorch depth-to-space
    pt_input = torch.from_numpy(input_np)
    b, c, t, h, w = pt_input.shape
    ft, fh, fw = 2, 2, 2
    c_out = 64

    # Reshape and permute
    pt_out = pt_input.reshape(b, c_out, ft, fh, fw, t, h, w)
    pt_out = pt_out.permute(0, 1, 5, 2, 6, 3, 7, 4)  # (B, C, T, ft, H, fh, W, fw)
    pt_out = pt_out.reshape(b, c_out, t * ft, h * fh, w * fw)

    result = compare_tensors("depth_to_space", mlx_output, pt_out)
    return result


def validate_denormalization(weights_path: str, seed: int = 42):
    """Validate latent denormalization using per-channel statistics."""
    print("\n" + "=" * 60)
    print("Validating Denormalization")
    print("=" * 60)

    with safe_open(weights_path, framework="pt") as f:
        mean_of_means = f.get_tensor("vae.per_channel_statistics.mean-of-means").float()
        std_of_means = f.get_tensor("vae.per_channel_statistics.std-of-means").float()

    np.random.seed(seed)
    latent_np = np.random.randn(1, 128, 3, 8, 8).astype(np.float32)

    # MLX denormalization
    mlx_latent = mx.array(latent_np)
    mlx_mean = mx.array(mean_of_means.numpy())
    mlx_std = mx.array(std_of_means.numpy())
    mlx_output = mlx_latent * mlx_std[None, :, None, None, None] + mlx_mean[None, :, None, None, None]
    mx.eval(mlx_output)

    # PyTorch denormalization
    pt_latent = torch.from_numpy(latent_np)
    pt_output = pt_latent * std_of_means.view(1, -1, 1, 1, 1) + mean_of_means.view(1, -1, 1, 1, 1)

    result = compare_tensors("denormalization", mlx_output, pt_output)
    return result


def validate_full_decoder(weights_path: str, seed: int = 42):
    """Run full decoder and report statistics (no PyTorch comparison)."""
    print("\n" + "=" * 60)
    print("Full Decoder Statistics")
    print("=" * 60)

    # Load MLX decoder
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)

    # Create test latent
    np.random.seed(seed)
    latent_np = np.random.randn(1, 128, 3, 8, 8).astype(np.float32)
    mlx_latent = mx.array(latent_np)

    print(f"\nInput latent: mean={latent_np.mean():.4f}, std={latent_np.std():.4f}")

    # Run decoder
    output = decoder(mlx_latent, timestep=0.05, show_progress=False)
    mx.eval(output)

    output_np = np.array(output)
    print(f"Output: shape={output_np.shape}")
    print(f"Output: mean={output_np.mean():.4f}, std={output_np.std():.4f}")
    print(f"Output: range=[{output_np.min():.4f}, {output_np.max():.4f}]")

    # Expected: for video in [-1, 1], mean should be ~0
    brightness = (output_np.mean() + 1) / 2
    print(f"\nBrightness (raw): {brightness:.1%}")
    print(f"Bias needed for 50%: {0.0 - output_np.mean():.4f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate MLX VAE decoder operations")
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to safetensors weights",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MLX VAE Decoder Validation")
    print("=" * 60)

    results = []

    # Validate individual operations
    results.append(("Pixel Norm", validate_pixel_norm(args.seed)))
    results.append(("Depth-to-Space", validate_depth_to_space(args.seed)))
    results.append(("Denormalization", validate_denormalization(args.weights, args.seed)))
    results.append(("Conv3D", validate_conv3d(args.weights, args.seed)))

    # Full decoder stats
    validate_full_decoder(args.weights, args.seed)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All operations match PyTorch reference!")
    else:
        print("\n✗ Some operations differ from PyTorch reference")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
