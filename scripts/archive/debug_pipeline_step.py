"""Debug full pipeline step by step to find where divergence occurs."""

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
    args = parser.parse_args()

    print("=" * 60)
    print("Full Pipeline Step Debug")
    print("=" * 60)

    from LTX_2_MLX.model.transformer import LTXAVModel, Modality, X0AVModel
    from LTX_2_MLX.loader.weight_converter import load_av_transformer_weights
    from LTX_2_MLX.components import (
        VideoLatentPatchifier,
        EulerDiffusionStep,
        GaussianNoiser,
        DISTILLED_SIGMA_VALUES,
    )
    from LTX_2_MLX.conditioning.tools import VideoLatentTools
    from LTX_2_MLX.types import VideoLatentShape, VideoPixelShape

    # Create model
    print("\n1. Loading model...")
    model = LTXAVModel()
    load_av_transformer_weights(model, args.weights, use_fp8=True, target_dtype="float16")
    mx.eval(model.parameters())
    model = X0AVModel(model)

    # Create small test resolution
    pixel_height = 256
    pixel_width = 384
    num_frames = 17

    pixel_shape = VideoPixelShape(
        batch=1, frames=num_frames, height=pixel_height, width=pixel_width, fps=24.0
    )
    latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)

    print(f"\n2. Creating initial state...")
    print(f"   Pixel shape: {num_frames}x{pixel_height}x{pixel_width}")
    print(f"   Latent shape: {latent_shape.frames}x{latent_shape.height}x{latent_shape.width}")

    # Setup components
    patchifier = VideoLatentPatchifier(patch_size=1)
    video_tools = VideoLatentTools(
        patchifier=patchifier,
        target_shape=latent_shape,
        fps=24.0,
    )
    stepper = EulerDiffusionStep()
    noiser = GaussianNoiser()

    # Create initial state
    mx.random.seed(42)
    video_state = video_tools.create_initial_state(dtype=mx.float16)
    print(f"   Initial latent shape: {video_state.latent.shape}")
    print(f"   Initial latent std: {float(mx.std(video_state.latent)):.4f}")

    # Add noise
    video_state = noiser(video_state, noise_scale=1.0)
    print(f"   After noising - latent std: {float(mx.std(video_state.latent)):.4f}")

    # Create zero text context
    batch = 1
    context_len = 64
    context_dim = 3840
    context = mx.zeros((batch, context_len, context_dim), dtype=mx.float16)
    context_mask = mx.ones((batch, context_len), dtype=mx.float16)

    # Get sigma schedule
    sigmas = mx.array(DISTILLED_SIGMA_VALUES)
    num_steps = len(sigmas) - 1

    print(f"\n3. Running denoising steps (distilled schedule)...")
    print(f"   Sigmas: {DISTILLED_SIGMA_VALUES}")

    for step_idx in range(num_steps):
        sigma = float(sigmas[step_idx])
        sigma_next = float(sigmas[step_idx + 1])

        # Create timesteps from denoise mask
        timesteps = video_state.denoise_mask[:, :, 0] * sigma

        # Create modalities
        video_modality = Modality(
            enabled=True,
            latent=video_state.latent,
            timesteps=timesteps,
            positions=video_state.positions,
            context=context,
            context_mask=context_mask,
        )
        audio_modality = Modality(
            enabled=False,
            latent=mx.zeros((batch, 0, 128), dtype=mx.float16),
            timesteps=mx.zeros((batch, 0), dtype=mx.float16),
            positions=mx.zeros((batch, 0, 4), dtype=mx.float16),
            context=mx.zeros((batch, 0, 2048), dtype=mx.float16),
            context_mask=mx.zeros((batch, 0), dtype=mx.float16),
        )

        # Run model to get denoised prediction
        denoised, _ = model(video_modality, audio_modality)
        mx.eval(denoised)

        # Compute velocity for Euler step
        # EulerDiffusionStep takes unpatchified latents, but we're patchified
        # Let's compute manually: velocity = (sample - denoised) / sigma
        # new_sample = sample + velocity * dt
        velocity = (video_state.latent - denoised) / sigma
        dt = sigma_next - sigma
        new_latent = video_state.latent + velocity * dt
        mx.eval(new_latent)

        # Compute statistics
        latent_std = float(mx.std(new_latent))
        denoised_std = float(mx.std(denoised))
        velocity_std = float(mx.std(velocity))

        # Compute correlation between current latent and denoised
        latent_flat = video_state.latent.flatten()
        denoised_flat = denoised.flatten()
        corr = float(mx.sum(latent_flat * denoised_flat)) / (
            float(mx.sqrt(mx.sum(latent_flat**2))) * float(mx.sqrt(mx.sum(denoised_flat**2))) + 1e-8
        )

        print(f"\n   Step {step_idx}: sigma {sigma:.4f} -> {sigma_next:.4f}")
        print(f"     Input latent std:  {float(mx.std(video_state.latent)):.4f}")
        print(f"     Denoised std:      {denoised_std:.4f}")
        print(f"     Velocity std:      {velocity_std:.4f}")
        print(f"     Output latent std: {latent_std:.4f}")
        print(f"     Input-Denoised correlation: {corr:.4f}")

        # Check for issues
        if latent_std > 5:
            print(f"     ⚠️ WARNING: Latent std very high!")
        if denoised_std < 0.01:
            print(f"     ⚠️ WARNING: Denoised std very low!")
        if velocity_std > 10:
            print(f"     ⚠️ WARNING: Velocity std very high!")

        # Update state
        video_state = video_state.replace(latent=new_latent)

    # Final statistics
    print(f"\n4. Final state...")
    print(f"   Final latent std: {float(mx.std(video_state.latent)):.4f}")
    print(f"   Final latent range: [{float(mx.min(video_state.latent)):.2f}, {float(mx.max(video_state.latent)):.2f}]")

    # Check if it converged or diverged
    final_std = float(mx.std(video_state.latent))
    if final_std < 0.1:
        print(f"\n   ✗ Collapsed (std too low)")
    elif final_std > 2:
        print(f"\n   ✗ Diverged (std too high)")
    else:
        print(f"\n   ✓ Reasonable final std")


if __name__ == "__main__":
    main()
