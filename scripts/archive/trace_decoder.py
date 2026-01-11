#!/usr/bin/env python3
"""
Trace through the VAE decoder layer by layer to find where spatial structure is lost.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

def main():
    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        load_vae_decoder_weights,
        _pixel_norm,
        get_timestep_embedding,
        unpatchify,
    )

    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("Loading VAE decoder...")
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)
    mx.eval(decoder.parameters())

    # Create structured input latent (spatial gradient)
    frames, height, width = 3, 8, 12
    batch = 1
    channels = 128

    print(f"\nCreating structured latent with spatial gradient...")
    mx.random.seed(42)

    # Base noise
    latent = mx.random.normal((batch, channels, frames, height, width)) * 0.3

    # Add strong spatial gradient
    x_coords = mx.linspace(-2, 2, width)[None, None, None, None, :]
    y_coords = mx.linspace(-2, 2, height)[None, None, None, :, None]
    spatial = x_coords + y_coords  # Range: [-4, 4]
    latent = latent + spatial

    print(f"Input latent: mean={float(mx.mean(latent)):.4f}, std={float(mx.std(latent)):.4f}")
    print(f"              range=[{float(mx.min(latent)):.4f}, {float(mx.max(latent)):.4f}]")

    # Check spatial structure in input
    input_np = np.array(latent[0, 0, 0])  # [H, W]
    print(f"Input spatial var: {input_np.var():.4f}")
    print(f"Input corner values: TL={input_np[0,0]:.2f}, TR={input_np[0,-1]:.2f}, BL={input_np[-1,0]:.2f}, BR={input_np[-1,-1]:.2f}")

    def check_spatial(name, x, show_corners=True):
        """Report spatial statistics at each layer."""
        x_np = np.array(x)
        print(f"\n{name}:")
        print(f"  Shape: {x.shape}")
        print(f"  Mean: {x_np.mean():.4f}, Std: {x_np.std():.4f}")
        print(f"  Range: [{x_np.min():.4f}, {x_np.max():.4f}]")

        # Get first frame
        if x_np.ndim == 5:  # [B, C, F, H, W]
            frame = x_np[0, :, 0, :, :]  # [C, H, W]
        elif x_np.ndim == 4:  # [C, F, H, W]
            frame = x_np[:, 0, :, :]  # [C, H, W]
        else:
            frame = x_np

        # Spatial variance per channel
        if frame.ndim == 3:  # [C, H, W]
            channel_spatial_vars = frame.var(axis=(1, 2))
            print(f"  Spatial var per channel: mean={channel_spatial_vars.mean():.6f}, max={channel_spatial_vars.max():.6f}")

            if show_corners and frame.shape[1] > 1 and frame.shape[2] > 1:
                # Average across channels for corner values
                frame_avg = frame.mean(axis=0)
                print(f"  Corner values (avg): TL={frame_avg[0,0]:.4f}, TR={frame_avg[0,-1]:.4f}, BL={frame_avg[-1,0]:.4f}, BR={frame_avg[-1,-1]:.4f}")
        elif frame.ndim == 2:  # [H, W]
            print(f"  Spatial var: {frame.var():.6f}")
            if show_corners:
                print(f"  Corner values: TL={frame[0,0]:.4f}, TR={frame[0,-1]:.4f}, BL={frame[-1,0]:.4f}, BR={frame[-1,-1]:.4f}")

    # Step through decoder manually
    print("\n" + "="*70)
    print("Tracing through decoder layers...")
    print("="*70)

    timestep = 0.05
    batch_size = latent.shape[0]
    scaled_timestep = mx.array([timestep * decoder.timestep_scale_multiplier] * batch_size)

    check_spatial("Input latent", latent)

    # Denormalize
    x = latent * decoder.std_of_means[None, :, None, None, None]
    x = x + decoder.mean_of_means[None, :, None, None, None]
    mx.eval(x)
    check_spatial("After denormalization", x)

    # Conv in
    x = decoder.conv_in(x, causal=True)
    mx.eval(x)
    check_spatial("After conv_in", x)

    # Up blocks
    x = decoder.up_blocks_0(x, causal=True, timestep=scaled_timestep)
    mx.eval(x)
    check_spatial("After up_blocks_0 (res)", x)

    x = decoder.up_blocks_1(x, causal=True)
    mx.eval(x)
    check_spatial("After up_blocks_1 (upsample)", x)

    x = decoder.up_blocks_2(x, causal=True, timestep=scaled_timestep)
    mx.eval(x)
    check_spatial("After up_blocks_2 (res)", x)

    x = decoder.up_blocks_3(x, causal=True)
    mx.eval(x)
    check_spatial("After up_blocks_3 (upsample)", x)

    x = decoder.up_blocks_4(x, causal=True, timestep=scaled_timestep)
    mx.eval(x)
    check_spatial("After up_blocks_4 (res)", x)

    x = decoder.up_blocks_5(x, causal=True)
    mx.eval(x)
    check_spatial("After up_blocks_5 (upsample)", x)

    x = decoder.up_blocks_6(x, causal=True, timestep=scaled_timestep)
    mx.eval(x)
    check_spatial("After up_blocks_6 (res)", x)

    # Pixel norm
    x_before_norm = x
    x = _pixel_norm(x)
    mx.eval(x)
    check_spatial("After pixel_norm", x)

    # Check if pixel_norm is killing variance
    norm_before = np.array(x_before_norm)
    norm_after = np.array(x)
    print(f"  Pixel norm effect: std before={norm_before.std():.4f}, after={norm_after.std():.4f}")

    # Scale/shift
    if scaled_timestep is not None and decoder.last_time_embedder is not None:
        t_emb = get_timestep_embedding(scaled_timestep, 256)
        time_emb = decoder.last_time_embedder(t_emb)
        time_emb = time_emb.reshape(batch_size, 2, 128)
        ss_table = decoder.last_scale_shift_table[None, :, :] + time_emb
        shift = ss_table[:, 0, :][:, :, None, None, None]
        scale = 1 + ss_table[:, 1, :][:, :, None, None, None]
    else:
        shift = decoder.last_scale_shift_table[0][None, :, None, None, None]
        scale = 1 + decoder.last_scale_shift_table[1][None, :, None, None, None]

    print(f"\nScale/shift parameters:")
    print(f"  Scale: mean={float(mx.mean(scale)):.4f}, range=[{float(mx.min(scale)):.4f}, {float(mx.max(scale)):.4f}]")
    print(f"  Shift: mean={float(mx.mean(shift)):.4f}, range=[{float(mx.min(shift)):.4f}, {float(mx.max(shift)):.4f}]")

    x = x * scale + shift
    mx.eval(x)
    check_spatial("After scale/shift", x)

    # SiLU activation
    x = nn.silu(x)
    mx.eval(x)
    check_spatial("After SiLU", x)

    # Conv out
    x = decoder.conv_out(x, causal=True)
    mx.eval(x)
    check_spatial("After conv_out", x)

    # Unpatchify
    x = unpatchify(x, patch_size_hw=4, patch_size_t=1)
    mx.eval(x)
    check_spatial("After unpatchify (final)", x)

    # Check RGB channels
    output = x
    output_np = np.array(output[0])  # [3, T, H, W]
    print(f"\n" + "="*70)
    print("Final output analysis:")
    print("="*70)
    print(f"Shape: {output.shape}")
    print(f"RGB means: R={output_np[0].mean():.4f}, G={output_np[1].mean():.4f}, B={output_np[2].mean():.4f}")
    print(f"RGB stds:  R={output_np[0].std():.4f}, G={output_np[1].std():.4f}, B={output_np[2].std():.4f}")

    # Check spatial gradient in output
    frame_0 = output_np[:, 0, :, :]  # [3, H, W]
    for i, name in enumerate(['R', 'G', 'B']):
        ch = frame_0[i]
        print(f"{name}: TL={ch[0,0]:.4f}, TR={ch[0,-1]:.4f}, BL={ch[-1,0]:.4f}, BR={ch[-1,-1]:.4f}")

    # Is there any correlation between input gradient and output?
    print(f"\n" + "="*70)
    print("Gradient preservation test:")
    print("="*70)

    # Create expected gradient in output space
    input_grad = np.array(latent[0].mean(axis=0))  # [F, H, W] - mean across channels
    input_grad_frame = input_grad[0]  # [H, W]

    # Resize to match output
    # Input: [H, W], Output: [H*32, W*32] but we sampled at [H*4, W*4] resolution
    from scipy.ndimage import zoom
    output_h, output_w = frame_0.shape[1], frame_0.shape[2]
    scale_h, scale_w = output_h / height, output_w / width
    expected_grad = zoom(input_grad_frame, (scale_h, scale_w), order=1)

    # Compare
    output_mean_frame = frame_0.mean(axis=0)  # [H, W]
    corr = np.corrcoef(expected_grad.flatten(), output_mean_frame.flatten())[0, 1]
    print(f"Correlation between input gradient and output: {corr:.4f}")

    if corr < 0.1:
        print("  -> WARNING: Spatial structure is NOT preserved through decoder!")
    elif corr < 0.5:
        print("  -> Partial correlation - some structure preserved")
    else:
        print("  -> Good correlation - structure is preserved")

if __name__ == "__main__":
    main()
