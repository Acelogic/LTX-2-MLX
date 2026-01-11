#!/usr/bin/env python3
"""
Validate Video VAE Encoder implementation.

Tests:
1. Weight loading
2. Output shape correctness
3. Round-trip encoding/decoding comparison
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def test_encoder_shapes(encoder, verbose: bool = True):
    """Test that encoder produces correct output shapes."""
    test_cases = [
        # (frames, height, width) -> expected (latent_frames, latent_h, latent_w)
        (9, 128, 128, 2, 4, 4),
        (17, 256, 256, 3, 8, 8),
        (25, 256, 384, 4, 8, 12),
        (33, 512, 512, 5, 16, 16),
    ]

    print("\nTesting encoder output shapes...")
    all_passed = True

    for frames, h, w, exp_f, exp_h, exp_w in test_cases:
        # Create test input: (B, C, F, H, W)
        video = mx.random.normal((1, 3, frames, h, w))

        # Encode
        latent = encoder(video, show_progress=False)
        mx.eval(latent)

        # Check shape
        b, c, f, lh, lw = latent.shape
        expected = (1, 128, exp_f, exp_h, exp_w)
        actual = (b, c, f, lh, lw)

        passed = actual == expected
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        if verbose:
            print(f"  Input ({frames}, {h}, {w}) -> Latent {actual} (expected {expected}) [{status}]")

    return all_passed


def test_round_trip(encoder, decoder, verbose: bool = True):
    """Test encode-decode round trip produces reasonable results."""
    print("\nTesting round-trip encode/decode...")

    # Create test input
    video = mx.random.normal((1, 3, 9, 128, 128))

    # Encode
    latent = encoder(video, show_progress=False)
    mx.eval(latent)

    # Decode
    reconstructed = decoder(latent, timestep=None, show_progress=False)
    mx.eval(reconstructed)

    # Check shapes match
    if video.shape != reconstructed.shape:
        print(f"  Shape mismatch: input {video.shape} vs output {reconstructed.shape}")
        return False

    # Compute reconstruction error (should be reasonable but not zero)
    mse = float(mx.mean((video - reconstructed) ** 2))
    max_diff = float(mx.max(mx.abs(video - reconstructed)))

    if verbose:
        print(f"  Input shape: {video.shape}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Output shape: {reconstructed.shape}")
        print(f"  Reconstruction MSE: {mse:.4f}")
        print(f"  Max absolute diff: {max_diff:.4f}")
        print(f"  Latent mean: {float(mx.mean(latent)):.4f}")
        print(f"  Latent std: {float(mx.std(latent)):.4f}")

    # MSE should be reasonable (VAE reconstruction isn't perfect)
    # But it should at least produce output in similar range
    passed = mse < 10.0 and max_diff < 10.0
    print(f"  Round-trip test: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate VAE encoder implementation")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to LTX-2 weights file",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 computation",
    )
    parser.add_argument(
        "--test-round-trip",
        action="store_true",
        help="Also test round-trip encoding/decoding",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        print("Please download LTX-2 weights first.")
        sys.exit(1)

    # Import encoder
    from LTX_2_MLX.model.video_vae import (
        SimpleVideoEncoder,
        load_vae_encoder_weights,
    )

    # Create encoder
    compute_dtype = mx.float16 if args.fp16 else mx.float32
    print(f"Creating encoder with compute_dtype={compute_dtype}...")
    encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)

    # Load weights
    load_vae_encoder_weights(encoder, str(weights_path))
    mx.eval(encoder.parameters())

    # Run shape tests
    shape_test_passed = test_encoder_shapes(encoder)

    # Run round-trip test if requested
    round_trip_passed = True
    if args.test_round_trip:
        from LTX_2_MLX.model.video_vae import (
            SimpleVideoDecoder,
            load_vae_decoder_weights,
        )

        print("\nLoading decoder for round-trip test...")
        decoder = SimpleVideoDecoder(compute_dtype=compute_dtype)
        load_vae_decoder_weights(decoder, str(weights_path))
        mx.eval(decoder.parameters())

        round_trip_passed = test_round_trip(encoder, decoder)

    # Summary
    print("\n" + "=" * 50)
    if shape_test_passed and round_trip_passed:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
