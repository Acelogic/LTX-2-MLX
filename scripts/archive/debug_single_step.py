"""Debug single denoising step to verify model predicts in correct direction."""

import argparse
import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Single Step Denoising Test")
    print("=" * 60)

    from LTX_2_MLX.model.transformer import LTXAVModel, Modality, X0AVModel
    from LTX_2_MLX.loader.weight_converter import load_av_transformer_weights
    from LTX_2_MLX.components import VideoLatentPatchifier
    from LTX_2_MLX.conditioning.tools import VideoLatentTools
    from LTX_2_MLX.types import VideoLatentShape, VideoPixelShape

    # Create model
    print("\n1. Creating model...")
    model = LTXAVModel()
    load_av_transformer_weights(model, args.weights, use_fp8=True, target_dtype="float16")
    mx.eval(model.parameters())

    # Wrap in X0Model
    model = X0AVModel(model)

    # Create synthetic test data
    print("\n2. Creating test data...")
    np.random.seed(42)

    # Small latent for testing - use valid pixel sizes
    pixel_height = 256
    pixel_width = 384
    num_frames = 17  # 8*2 + 1

    pixel_shape = VideoPixelShape(
        batch=1, frames=num_frames, height=pixel_height, width=pixel_width, fps=24.0
    )
    latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)

    batch = latent_shape.batch
    channels = latent_shape.channels
    frames = latent_shape.frames
    height = latent_shape.height
    width = latent_shape.width

    print(f"   Latent shape: ({batch}, {channels}, {frames}, {height}, {width})")

    # Create random initial noise (this is what the denoising loop starts with)
    # The noise should have unit variance for flow matching
    mx.random.seed(42)
    noise = mx.random.normal((batch, channels, frames, height, width), dtype=mx.float16)

    # For testing, we assume "data" would be what a clean VAE latent looks like
    # VAE latents typically have lower variance than pure noise
    # Let's use a scaled version of another random sample
    data = mx.random.normal((batch, channels, frames, height, width), dtype=mx.float16) * 0.3

    # Create noisy sample at sigma level
    sigma = args.sigma
    # Flow matching: x_t = (1-t)*data + t*noise, where t = sigma
    sample = (1 - sigma) * data + sigma * noise

    print(f"   Sigma: {sigma}")
    print(f"   Data std: {float(mx.std(data)):.4f}")
    print(f"   Noise std: {float(mx.std(noise)):.4f}")
    print(f"   Sample std: {float(mx.std(sample)):.4f}")

    # Use VideoLatentTools for patchifying
    print("\n3. Creating latent state...")
    patchifier = VideoLatentPatchifier(patch_size=1)
    video_tools = VideoLatentTools(
        patchifier=patchifier,
        target_shape=latent_shape,
        fps=24.0,
    )

    # Create initial state with our sample as initial latent
    video_state = video_tools.create_initial_state(dtype=mx.float16, initial_latent=sample)
    print(f"   Patchified latent shape: {video_state.latent.shape}")
    print(f"   Positions shape: {video_state.positions.shape}")

    # Create simple text context (zeros for unconditioned)
    context_len = 64
    context_dim = 3840  # Gemma dimension
    context = mx.zeros((batch, context_len, context_dim), dtype=mx.float16)
    context_mask = mx.ones((batch, context_len), dtype=mx.float16)

    # Create per-token timesteps from denoise_mask
    # denoise_mask is (B, T, 1), squeeze last dim for timesteps
    timesteps = video_state.denoise_mask[:, :, 0] * sigma

    # Create modality - positions are already in correct format (B, n_dims, T, 2)
    video_modality = Modality(
        enabled=True,
        latent=video_state.latent,
        timesteps=timesteps,
        positions=video_state.positions,
        context=context,
        context_mask=context_mask,
    )

    # Disabled audio modality - use 128 channels to match AUDIO_IN_CHANNELS
    audio_modality = Modality(
        enabled=False,
        latent=mx.zeros((batch, 0, 128), dtype=mx.float16),
        timesteps=mx.zeros((batch, 0), dtype=mx.float16),
        positions=mx.zeros((batch, 0, 4), dtype=mx.float16),
        context=mx.zeros((batch, 0, 2048), dtype=mx.float16),
        context_mask=mx.zeros((batch, 0), dtype=mx.float16),
    )

    # Run model forward pass
    print("\n4. Running model forward pass...")
    denoised_patchified, _ = model(video_modality, audio_modality)
    mx.eval(denoised_patchified)

    # Unpatchify
    video_state_out = video_state.replace(latent=denoised_patchified)
    video_state_out = video_tools.unpatchify(video_state_out)
    denoised = video_state_out.latent
    mx.eval(denoised)

    print(f"   Denoised shape: {denoised.shape}")
    print(f"   Denoised std: {float(mx.std(denoised)):.4f}")
    print(f"   Denoised range: [{float(mx.min(denoised)):.4f}, {float(mx.max(denoised)):.4f}]")

    # Check if denoised is closer to data than sample
    print("\n5. Analyzing prediction direction...")

    # Compute distances
    sample_to_data = float(mx.mean((sample - data) ** 2))
    denoised_to_data = float(mx.mean((denoised - data) ** 2))
    sample_to_noise = float(mx.mean((sample - noise) ** 2))
    denoised_to_noise = float(mx.mean((denoised - noise) ** 2))

    print(f"   MSE(sample, data):    {sample_to_data:.6f}")
    print(f"   MSE(denoised, data):  {denoised_to_data:.6f}")
    print(f"   MSE(sample, noise):   {sample_to_noise:.6f}")
    print(f"   MSE(denoised, noise): {denoised_to_noise:.6f}")

    if denoised_to_data < sample_to_data:
        print(f"\n   ✓ Denoised is CLOSER to data (good!)")
        print(f"   Improvement: {100 * (1 - denoised_to_data / sample_to_data):.1f}%")
    else:
        print(f"\n   ✗ Denoised is FARTHER from data (bad!)")
        print(f"   Degradation: {100 * (denoised_to_data / sample_to_data - 1):.1f}%")

    # Compute the implicit velocity the model predicted
    # velocity = (sample - denoised) / sigma
    if sigma > 0:
        velocity = (sample - denoised) / sigma
        mx.eval(velocity)
        print(f"\n   Implied velocity std: {float(mx.std(velocity)):.4f}")

        # True velocity = noise - data
        true_velocity = noise - data
        velocity_corr = float(mx.sum(velocity * true_velocity)) / (
            float(mx.sqrt(mx.sum(velocity**2))) * float(mx.sqrt(mx.sum(true_velocity**2))) + 1e-8
        )
        print(f"   Correlation with true velocity: {velocity_corr:.4f}")

        if velocity_corr > 0:
            print(f"   ✓ Velocity points in correct direction")
        else:
            print(f"   ✗ Velocity points in WRONG direction")

    # Check if the model output looks reasonable
    print("\n6. Sanity checks...")

    if np.isnan(np.array(denoised)).any():
        print("   ✗ NaN in output!")
    elif np.isinf(np.array(denoised)).any():
        print("   ✗ Inf in output!")
    else:
        print("   ✓ No NaN/Inf in output")

    denoised_std = float(mx.std(denoised))
    if denoised_std > 10:
        print(f"   ✗ Output std ({denoised_std:.2f}) very high - model may be unstable")
    elif denoised_std < 0.01:
        print(f"   ✗ Output std ({denoised_std:.4f}) very low - model may be dead")
    else:
        print(f"   ✓ Output std ({denoised_std:.4f}) reasonable")


if __name__ == "__main__":
    main()
