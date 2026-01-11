"""Debug VAE decoder to find source of grid pattern in output."""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def test_unpatchify_standalone():
    """Test unpatchify function in isolation."""
    print("=" * 60)
    print("Testing unpatchify in isolation")
    print("=" * 60)

    from LTX_2_MLX.model.video_vae.ops import unpatchify, patchify

    # Create a simple gradient input that we can trace through
    # Conv_out outputs (B, 48, T, H, W) -> unpatchify -> (B, 3, T, H*4, W*4)
    batch = 1
    frames = 3
    h_small = 4
    w_small = 6
    channels_packed = 48  # 3 * 16 = 3 * 4 * 4

    # Create input with recognizable pattern
    # Each "pixel" position should have unique value
    x_np = np.zeros((batch, channels_packed, frames, h_small, w_small), dtype=np.float32)

    # Fill with position-dependent values
    for f in range(frames):
        for y in range(h_small):
            for x_pos in range(w_small):
                # Encode position in first 3 channels of packed dimension
                # After unpatchify, position (y,x) should expand to (y*4:y*4+4, x*4:x*4+4)
                base_val = f * 100 + y * 10 + x_pos
                for c in range(channels_packed):
                    x_np[0, c, f, y, x_pos] = float(base_val + c * 0.01)

    x = mx.array(x_np)
    mx.eval(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Input sample at [0, :3, 0, 0, 0]: {x[0, :3, 0, 0, 0]}")
    print(f"Input sample at [0, :3, 0, 1, 1]: {x[0, :3, 0, 1, 1]}")

    # Apply unpatchify
    out = unpatchify(x, patch_size_hw=4, patch_size_t=1)
    mx.eval(out)

    print(f"\nOutput shape: {out.shape}")
    expected_shape = (batch, 3, frames, h_small * 4, w_small * 4)
    print(f"Expected shape: {expected_shape}")

    # Check if unpatchify spread values correctly
    # Position (0,0) in input should map to (0:4, 0:4) in output
    print(f"\nOutput at position [0, 0, 0, 0:4, 0:4]:")
    print(np.array(out[0, 0, 0, :4, :4]))

    print(f"\nOutput at position [0, 0, 0, 4:8, 4:8]:")
    print(np.array(out[0, 0, 0, 4:8, 4:8]))

    # Test roundtrip
    print("\n--- Testing patchify -> unpatchify roundtrip ---")
    # Create a 3-channel image
    img = mx.arange(3 * frames * h_small * 4 * w_small * 4).reshape(
        batch, 3, frames, h_small * 4, w_small * 4
    ).astype(mx.float32)

    # Patchify then unpatchify
    patched = patchify(img, patch_size_hw=4, patch_size_t=1)
    recovered = unpatchify(patched, patch_size_hw=4, patch_size_t=1)
    mx.eval(recovered)

    print(f"Original shape: {img.shape}")
    print(f"Patched shape: {patched.shape}")
    print(f"Recovered shape: {recovered.shape}")

    # Check if roundtrip preserves values
    diff = mx.abs(img - recovered)
    max_diff = float(mx.max(diff))
    print(f"Max diff after roundtrip: {max_diff}")

    if max_diff < 1e-5:
        print("PASS: Roundtrip is correct")
    else:
        print("FAIL: Roundtrip has significant error!")


def test_vae_decoder_with_zero_input(weights_path: str):
    """Test VAE decoder with zero input to see baseline output."""
    print("\n" + "=" * 60)
    print("Testing VAE decoder with zero input")
    print("=" * 60)

    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        load_vae_decoder_weights,
    )

    # Load decoder
    print("\nLoading VAE decoder...")
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)
    mx.eval(decoder.parameters())

    # Check per-channel statistics
    print("\nPer-channel statistics:")
    print(f"  mean_of_means: range [{float(mx.min(decoder.mean_of_means)):.4f}, {float(mx.max(decoder.mean_of_means)):.4f}]")
    print(f"  std_of_means: range [{float(mx.min(decoder.std_of_means)):.4f}, {float(mx.max(decoder.std_of_means)):.4f}]")

    if float(mx.max(decoder.std_of_means)) == 0:
        print("  WARNING: std_of_means is all zeros!")

    # Create zero latent
    batch = 1
    channels = 128
    frames = 3
    height = 4
    width = 6

    latent = mx.zeros((batch, channels, frames, height, width), dtype=mx.float32)

    print(f"\nInput latent shape: {latent.shape}")
    print(f"Input latent: all zeros")

    # Run decoder
    print("\nRunning decoder...")
    output = decoder(latent, timestep=0.05, show_progress=False)
    mx.eval(output)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{float(mx.min(output)):.4f}, {float(mx.max(output)):.4f}]")
    print(f"Output mean: {float(mx.mean(output)):.4f}")
    print(f"Output std: {float(mx.std(output)):.4f}")

    # Check for grid pattern
    output_np = np.array(output[0, 0, 0])  # First frame, first channel
    print(f"\nChecking for grid pattern in first frame...")
    print(f"  Shape: {output_np.shape}")

    # Sample corners of 4x4 blocks
    block_size = 4
    h_blocks = output_np.shape[0] // block_size
    w_blocks = output_np.shape[1] // block_size

    print(f"  Block grid: {h_blocks} x {w_blocks}")

    # Check variance within blocks vs between blocks
    within_block_vars = []
    between_block_vars = []

    for by in range(min(4, h_blocks)):
        for bx in range(min(4, w_blocks)):
            block = output_np[by*4:(by+1)*4, bx*4:(bx+1)*4]
            within_block_vars.append(np.var(block))

    # Check variance between block means
    block_means = []
    for by in range(h_blocks):
        for bx in range(w_blocks):
            block = output_np[by*4:(by+1)*4, bx*4:(bx+1)*4]
            block_means.append(np.mean(block))

    print(f"  Mean within-block variance: {np.mean(within_block_vars):.6f}")
    print(f"  Variance of block means: {np.var(block_means):.6f}")

    if np.mean(within_block_vars) < np.var(block_means) * 0.1:
        print("  WARNING: Possible grid pattern detected - blocks are too uniform internally!")


def test_vae_decoder_with_gradient(weights_path: str):
    """Test VAE decoder with gradient input to check spatial coherence."""
    print("\n" + "=" * 60)
    print("Testing VAE decoder with gradient input")
    print("=" * 60)

    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        load_vae_decoder_weights,
        decode_latent,
    )

    # Load decoder
    print("\nLoading VAE decoder...")
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)
    mx.eval(decoder.parameters())

    # Create gradient latent (should produce gradient in output)
    batch = 1
    channels = 128
    frames = 3
    height = 8
    width = 12

    # Create horizontal gradient
    latent_np = np.zeros((batch, channels, frames, height, width), dtype=np.float32)
    for x_pos in range(width):
        # Set all channels to gradient value
        gradient_val = (x_pos / width) * 2 - 1  # -1 to 1
        latent_np[0, :, :, :, x_pos] = gradient_val

    latent = mx.array(latent_np)
    mx.eval(latent)

    print(f"\nInput latent shape: {latent.shape}")
    print(f"Input is horizontal gradient from -1 to 1")

    # Run decoder via decode_latent
    print("\nRunning decode_latent...")
    video = decode_latent(latent, decoder, timestep=0.05, contrast_boost=None)
    mx.eval(video)

    print(f"\nOutput video shape: {video.shape}")
    print(f"Output dtype: {video.dtype}")

    # Check if gradient is preserved
    output_np = np.array(video)

    # Average across channels to get grayscale
    gray = output_np.mean(axis=-1)  # (T, H, W)

    print(f"\nChecking gradient preservation in first frame...")
    first_frame = gray[0]

    # Check if left side is darker than right side
    left_mean = first_frame[:, :first_frame.shape[1]//2].mean()
    right_mean = first_frame[:, first_frame.shape[1]//2:].mean()

    print(f"  Left half mean: {left_mean:.2f}")
    print(f"  Right half mean: {right_mean:.2f}")

    if right_mean > left_mean:
        print("  PASS: Gradient direction preserved (dark->light)")
    else:
        print("  WARNING: Gradient direction NOT preserved!")

    # Save frame for inspection
    from PIL import Image
    frame = video[0]  # (H, W, 3)
    frame_np = np.array(frame)
    img = Image.fromarray(frame_np.astype(np.uint8))
    img.save("/tmp/vae_gradient_test.png")
    print(f"\n  Saved test frame to /tmp/vae_gradient_test.png")


def test_vae_internal_states(weights_path: str):
    """Trace through VAE decoder and check intermediate states."""
    print("\n" + "=" * 60)
    print("Tracing VAE decoder internal states")
    print("=" * 60)

    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        load_vae_decoder_weights,
        _pixel_norm,
    )
    from LTX_2_MLX.model.video_vae.ops import unpatchify

    # Load decoder
    print("\nLoading VAE decoder...")
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)
    mx.eval(decoder.parameters())

    # Create small random latent
    mx.random.seed(42)
    batch = 1
    channels = 128
    frames = 3
    height = 4
    width = 6

    latent = mx.random.normal((batch, channels, frames, height, width))
    latent = latent * 0.5  # Scale to reasonable range
    mx.eval(latent)

    print(f"\nInput latent: {latent.shape}")
    print(f"  range: [{float(mx.min(latent)):.4f}, {float(mx.max(latent)):.4f}]")
    print(f"  std: {float(mx.std(latent)):.4f}")

    # Manual trace through decoder
    print("\n--- Tracing decoder layers ---")

    # Denormalize
    x = latent * decoder.std_of_means[None, :, None, None, None]
    x = x + decoder.mean_of_means[None, :, None, None, None]
    mx.eval(x)
    print(f"\n1. After denormalize:")
    print(f"   range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"   std: {float(mx.std(x)):.4f}")

    # Conv in
    x = decoder.conv_in(x, causal=True)
    mx.eval(x)
    print(f"\n2. After conv_in: {x.shape}")
    print(f"   range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"   std: {float(mx.std(x)):.4f}")

    # Up blocks
    for i, block_name in enumerate(["up_blocks_0", "up_blocks_1", "up_blocks_2",
                                    "up_blocks_3", "up_blocks_4", "up_blocks_5",
                                    "up_blocks_6"]):
        block = getattr(decoder, block_name)

        if "0" in block_name or "2" in block_name or "4" in block_name or "6" in block_name:
            # Res block group
            x = block(x, causal=True, timestep=None)
        else:
            # Upsample
            x = block(x, causal=True)

        mx.eval(x)
        print(f"\n{i+3}. After {block_name}: {x.shape}")
        print(f"   range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
        print(f"   std: {float(mx.std(x)):.4f}")

    # Final norm and scale/shift
    x = _pixel_norm(x)
    shift = decoder.last_scale_shift_table[0][None, :, None, None, None]
    scale = 1 + decoder.last_scale_shift_table[1][None, :, None, None, None]
    x = x * scale + shift
    mx.eval(x)
    print(f"\n10. After final norm + scale/shift: {x.shape}")
    print(f"   range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"   std: {float(mx.std(x)):.4f}")

    # SiLU
    import mlx.nn as nn
    x = nn.silu(x)
    mx.eval(x)
    print(f"\n11. After SiLU: {x.shape}")
    print(f"   range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"   std: {float(mx.std(x)):.4f}")

    # Conv out
    x_conv_out = decoder.conv_out(x, causal=True)
    mx.eval(x_conv_out)
    print(f"\n12. After conv_out: {x_conv_out.shape}")
    print(f"   range: [{float(mx.min(x_conv_out)):.4f}, {float(mx.max(x_conv_out)):.4f}]")
    print(f"   std: {float(mx.std(x_conv_out)):.4f}")

    # Check conv_out channels
    print(f"\n   Conv_out per-channel stats (first 6 of 48):")
    for c in range(min(6, x_conv_out.shape[1])):
        ch = x_conv_out[0, c]
        print(f"     ch {c}: mean={float(mx.mean(ch)):.4f}, std={float(mx.std(ch)):.4f}")

    # Unpatchify
    x_unpatch = unpatchify(x_conv_out, patch_size_hw=4, patch_size_t=1)
    mx.eval(x_unpatch)
    print(f"\n13. After unpatchify: {x_unpatch.shape}")
    print(f"   range: [{float(mx.min(x_unpatch)):.4f}, {float(mx.max(x_unpatch)):.4f}]")
    print(f"   std: {float(mx.std(x_unpatch)):.4f}")

    # Check spatial variance
    print(f"\n   Per-channel spatial variance:")
    for c in range(3):
        ch = x_unpatch[0, c]
        print(f"     ch {c}: spatial_std={float(mx.std(ch)):.4f}")

    # Check if there's a grid pattern
    out_np = np.array(x_unpatch[0, 0, 0])  # First frame, first channel

    # Compare variance within 4x4 blocks vs across blocks
    h, w = out_np.shape
    block_means = []
    within_vars = []

    for by in range(h // 4):
        for bx in range(w // 4):
            block = out_np[by*4:(by+1)*4, bx*4:(bx+1)*4]
            block_means.append(np.mean(block))
            within_vars.append(np.var(block))

    print(f"\n   Grid pattern analysis:")
    print(f"     Mean within-block variance: {np.mean(within_vars):.6f}")
    print(f"     Variance of block means: {np.var(block_means):.6f}")
    ratio = np.var(block_means) / (np.mean(within_vars) + 1e-10)
    print(f"     Ratio (between/within): {ratio:.2f}")

    if ratio > 10:
        print(f"     WARNING: Strong grid pattern detected!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
    )
    args = parser.parse_args()

    # Test unpatchify standalone
    test_unpatchify_standalone()

    # Test VAE with various inputs
    test_vae_decoder_with_zero_input(args.weights)
    test_vae_decoder_with_gradient(args.weights)
    test_vae_internal_states(args.weights)


if __name__ == "__main__":
    main()
