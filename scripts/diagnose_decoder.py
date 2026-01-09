#!/usr/bin/env python3
"""
Layer-by-layer VAE decoder diagnostics.
Identifies where visual artifacts originate.
"""

import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    _pixel_norm,
)
from LTX_2_MLX.model.video_vae.ops import unpatchify


def analyze_tensor(name: str, x: mx.array, output_dir: str):
    """Analyze and visualize a tensor."""
    x_np = np.array(x)

    stats = {
        "name": name,
        "shape": x_np.shape,
        "dtype": str(x_np.dtype),
        "min": float(x_np.min()),
        "max": float(x_np.max()),
        "mean": float(x_np.mean()),
        "std": float(x_np.std()),
        "has_nan": bool(np.isnan(x_np).any()),
        "has_inf": bool(np.isinf(x_np).any()),
        "nan_count": int(np.isnan(x_np).sum()),
        "inf_count": int(np.isinf(x_np).sum()),
    }

    # Print stats
    print(f"\n{'='*60}")
    print(f"Stage: {name}")
    print(f"  Shape: {stats['shape']}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    if stats['has_nan']:
        print(f"  ⚠️  NaN detected: {stats['nan_count']} values")
    if stats['has_inf']:
        print(f"  ⚠️  Inf detected: {stats['inf_count']} values")

    # Check for grid pattern (high frequency in spatial dims)
    if len(x_np.shape) == 5:  # (B, C, T, H, W)
        # Get first frame, first channel
        frame = x_np[0, 0, 0]  # (H, W)

        # Compute horizontal and vertical gradients
        h_grad = np.abs(np.diff(frame, axis=1)).mean()
        v_grad = np.abs(np.diff(frame, axis=0)).mean()

        # Compute 2-pixel and 4-pixel periodic gradients
        h_grad_2 = np.abs(frame[:, 2:] - frame[:, :-2]).mean() if frame.shape[1] > 2 else 0
        h_grad_4 = np.abs(frame[:, 4:] - frame[:, :-4]).mean() if frame.shape[1] > 4 else 0

        print(f"  Gradient analysis (first frame, ch0):")
        print(f"    Adjacent pixel diff: h={h_grad:.4f}, v={v_grad:.4f}")
        print(f"    2-pixel diff: {h_grad_2:.4f}, 4-pixel diff: {h_grad_4:.4f}")

        # Flag potential grid pattern
        if h_grad > 0 and h_grad_2 / h_grad < 0.7:
            print(f"  🔴 GRID PATTERN DETECTED (2-pixel periodicity)")
        if h_grad > 0 and h_grad_4 / h_grad < 0.5:
            print(f"  🔴 GRID PATTERN DETECTED (4-pixel periodicity)")

    # Save visualization
    if len(x_np.shape) == 5:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"{name}: shape={x_np.shape}", fontsize=14)

        # Top row: first 4 channels of first frame
        for i in range(min(4, x_np.shape[1])):
            ax = axes[0, i]
            im = ax.imshow(x_np[0, i, 0], cmap='viridis')
            ax.set_title(f"Ch {i}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Bottom row: channel statistics
        ax = axes[1, 0]
        channel_means = x_np[0, :, 0].mean(axis=(1, 2))
        ax.bar(range(len(channel_means)), channel_means)
        ax.set_title("Channel means")
        ax.set_xlabel("Channel")

        ax = axes[1, 1]
        channel_stds = x_np[0, :, 0].std(axis=(1, 2))
        ax.bar(range(len(channel_stds)), channel_stds)
        ax.set_title("Channel stds")
        ax.set_xlabel("Channel")

        ax = axes[1, 2]
        ax.hist(x_np.flatten(), bins=50, density=True)
        ax.set_title("Value distribution")
        ax.set_xlabel("Value")

        # FFT to detect periodicity
        ax = axes[1, 3]
        frame = x_np[0, 0, 0]
        fft = np.abs(np.fft.fft2(frame))
        fft_shift = np.fft.fftshift(fft)
        ax.imshow(np.log1p(fft_shift), cmap='hot')
        ax.set_title("FFT (log scale)")
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=100)
        plt.close()

    # Save tensor
    np.save(os.path.join(output_dir, f"{name}.npy"), x_np)

    return stats


def diagnose_decoder(latent_path: str, weights_path: str, output_dir: str = "/tmp/decoder_debug"):
    """Run decoder with layer-by-layer analysis."""

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load latent
    print(f"\nLoading latent from {latent_path}...")
    data = np.load(latent_path)
    latent_np = data['latent']
    latent = mx.array(latent_np)

    # Load decoder
    print(f"Loading decoder from {weights_path}...")
    decoder = SimpleVideoDecoder()
    load_vae_decoder_weights(decoder, weights_path)

    all_stats = []

    # Stage 0: Input
    stats = analyze_tensor("00_input", latent, output_dir)
    all_stats.append(stats)

    x = latent

    # Stage 1: Denormalize
    x = x * decoder.std_of_means[None, :, None, None, None]
    x = x + decoder.mean_of_means[None, :, None, None, None]
    mx.eval(x)
    stats = analyze_tensor("01_denorm", x, output_dir)
    all_stats.append(stats)

    # Stage 2: Conv in
    x = decoder.conv_in(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("02_conv_in", x, output_dir)
    all_stats.append(stats)

    # Stage 3: Up blocks 0 (res blocks)
    x = decoder.up_blocks_0(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("03_up_blocks_0", x, output_dir)
    all_stats.append(stats)

    # Stage 4: Up blocks 1 (upsample)
    x = decoder.up_blocks_1(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("04_up_blocks_1_upsample", x, output_dir)
    all_stats.append(stats)

    # Stage 5: Up blocks 2 (res blocks)
    x = decoder.up_blocks_2(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("05_up_blocks_2", x, output_dir)
    all_stats.append(stats)

    # Stage 6: Up blocks 3 (upsample)
    x = decoder.up_blocks_3(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("06_up_blocks_3_upsample", x, output_dir)
    all_stats.append(stats)

    # Stage 7: Up blocks 4 (res blocks)
    x = decoder.up_blocks_4(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("07_up_blocks_4", x, output_dir)
    all_stats.append(stats)

    # Stage 8: Up blocks 5 (upsample)
    x = decoder.up_blocks_5(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("08_up_blocks_5_upsample", x, output_dir)
    all_stats.append(stats)

    # Stage 9: Up blocks 6 (res blocks)
    x = decoder.up_blocks_6(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("09_up_blocks_6", x, output_dir)
    all_stats.append(stats)

    # Stage 10: Final norm + activation
    # LTX ordering: row 0 = shift, row 1 = scale
    x = _pixel_norm(x)
    shift = decoder.last_scale_shift_table[0][None, :, None, None, None]
    scale = 1 + decoder.last_scale_shift_table[1][None, :, None, None, None]
    x = x * scale + shift
    x = nn.silu(x)
    mx.eval(x)
    stats = analyze_tensor("10_final_norm", x, output_dir)
    all_stats.append(stats)

    # Stage 11: Conv out
    x = decoder.conv_out(x, causal=True)
    mx.eval(x)
    stats = analyze_tensor("11_conv_out", x, output_dir)
    all_stats.append(stats)

    # Stage 12: Unpatchify
    x = unpatchify(x, patch_size_hw=4, patch_size_t=1)
    mx.eval(x)
    stats = analyze_tensor("12_unpatchify", x, output_dir)
    all_stats.append(stats)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    problem_stages = []
    for s in all_stats:
        if s['has_nan'] or s['has_inf']:
            problem_stages.append(f"{s['name']}: NaN/Inf detected")
        if abs(s['max']) > 1000 or abs(s['min']) > 1000:
            problem_stages.append(f"{s['name']}: Extreme values (range [{s['min']:.1f}, {s['max']:.1f}])")

    if problem_stages:
        print("\n🔴 PROBLEMS DETECTED:")
        for p in problem_stages:
            print(f"  - {p}")
    else:
        print("\n✅ No obvious numerical issues detected")

    print(f"\nVisualization images saved to: {output_dir}")
    print(f"Open with: open {output_dir}/*.png")

    return all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose VAE decoder layer by layer")
    parser.add_argument("--latent", type=str, default="/tmp/dog_video_latent.npz",
                        help="Path to latent file")
    parser.add_argument("--weights", type=str,
                        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
                        help="Path to weights")
    parser.add_argument("--output", type=str, default="/tmp/decoder_debug",
                        help="Output directory")

    args = parser.parse_args()

    diagnose_decoder(args.latent, args.weights, args.output)
