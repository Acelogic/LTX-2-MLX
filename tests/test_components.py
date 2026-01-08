"""Tests for LTX-2 MLX components."""

import mlx.core as mx
import numpy as np


def test_sigma_schedule():
    """Test sigma schedule generation."""
    from ltx_mlx.components import get_sigma_schedule, LTX2Scheduler

    # Test default schedule
    sigmas = get_sigma_schedule(num_steps=50)
    assert sigmas.shape == (51,), f"Expected (51,), got {sigmas.shape}"
    assert float(sigmas[0]) > 0.9, "First sigma should be close to 1.0"
    assert float(sigmas[-1]) == 0.0, "Last sigma should be 0.0"
    print("  sigma_schedule: OK")

    # Test distilled schedule
    sigmas_distilled = get_sigma_schedule(num_steps=7, distilled=True)
    assert sigmas_distilled.shape == (8,), f"Expected (8,), got {sigmas_distilled.shape}"
    print("  distilled_schedule: OK")


def test_cfg_guider():
    """Test CFG guider."""
    from ltx_mlx.components import CFGGuider

    guider = CFGGuider(scale=7.5)

    # Create test tensors
    cond = mx.ones((1, 100, 128))
    uncond = mx.zeros((1, 100, 128))

    # CFGGuider returns the delta: (scale - 1) * (cond - uncond)
    delta = guider.delta(cond, uncond)
    guided = uncond + delta

    assert guided.shape == cond.shape
    # delta = (7.5 - 1) * (1 - 0) = 6.5
    # guided = uncond + delta = 0 + 6.5 = 6.5
    assert np.allclose(np.array(guided), 6.5, atol=1e-5)
    assert guider.enabled() == True
    print("  cfg_guider: OK")


def test_gaussian_noiser():
    """Test Gaussian noiser."""
    from ltx_mlx.components import GaussianNoiser
    from ltx_mlx.types import LatentState

    noiser = GaussianNoiser()

    # Create test latent state with all required fields
    latent = mx.zeros((1, 128, 4, 15, 22))
    denoise_mask = mx.ones_like(latent)  # Full denoising
    positions = mx.zeros((1, 3, 4 * 15 * 22, 2))  # Position grid placeholder
    clean_latent = mx.zeros_like(latent)  # Initial clean state

    latent_state = LatentState(
        latent=latent,
        denoise_mask=denoise_mask,
        positions=positions,
        clean_latent=clean_latent,
    )

    sigma = 0.5
    noisy_state = noiser(latent_state, noise_scale=sigma)

    assert noisy_state.latent.shape == latent.shape
    # Should have non-zero values after adding noise
    assert float(mx.abs(noisy_state.latent).mean()) > 0
    print("  gaussian_noiser: OK")


def test_euler_step():
    """Test Euler diffusion step."""
    from ltx_mlx.components import EulerDiffusionStep

    euler_step = EulerDiffusionStep()

    # Create test tensors
    sample = mx.ones((1, 128, 4, 15, 22))
    denoised_sample = mx.ones((1, 128, 4, 15, 22)) * 0.9

    # Create sigma schedule
    sigmas = mx.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    step_index = 0

    result = euler_step.step(sample, denoised_sample, sigmas, step_index)
    assert result.shape == sample.shape
    print("  euler_step: OK")


def test_patchifier():
    """Test video latent patchifier."""
    from ltx_mlx.components import VideoLatentPatchifier
    from ltx_mlx.types import VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)

    # Create test latent [B, C, F, H, W]
    latent = mx.random.normal(shape=(1, 128, 4, 15, 22))

    # Patchify: [B, C, F, H, W] -> [B, F*H*W, C]
    patchified = patchifier.patchify(latent)
    expected_tokens = 4 * 15 * 22  # F * H * W
    assert patchified.shape == (1, expected_tokens, 128), f"Got {patchified.shape}"
    print("  patchify: OK")

    # Unpatchify: [B, F*H*W, C] -> [B, C, F, H, W]
    output_shape = VideoLatentShape(batch=1, channels=128, frames=4, height=15, width=22)
    unpatchified = patchifier.unpatchify(patchified, output_shape=output_shape)
    assert unpatchified.shape == latent.shape, f"Got {unpatchified.shape}"
    print("  unpatchify: OK")

    # Roundtrip should preserve values
    diff = float(mx.abs(latent - unpatchified).max())
    assert diff < 1e-5, f"Roundtrip error: {diff}"
    print("  patchify_roundtrip: OK")


def test_rms_norm():
    """Test RMS normalization."""
    from ltx_mlx import rms_norm

    x = mx.random.normal(shape=(2, 10, 64))
    normed = rms_norm(x)

    assert normed.shape == x.shape
    # RMS of normalized tensor should be close to 1
    rms = float(mx.sqrt(mx.mean(normed * normed, axis=-1)).mean())
    assert 0.9 < rms < 1.1, f"RMS should be ~1.0, got {rms}"
    print("  rms_norm: OK")


def test_to_velocity_denoised():
    """Test velocity/denoised conversions."""
    from ltx_mlx import to_velocity, to_denoised

    x = mx.ones((1, 128, 4, 15, 22))
    noise = mx.random.normal(shape=x.shape)
    sigma = 0.5

    # Create noisy sample
    noisy = x + sigma * noise

    # Test to_velocity: convert (sample, denoised) -> velocity
    # velocity = (sample - denoised) / sigma
    velocity = to_velocity(noisy, sigma, x)  # (noisy sample, sigma, denoised prediction)
    assert velocity.shape == x.shape
    print("  to_velocity: OK")

    # Test to_denoised: convert (sample, velocity) -> denoised
    denoised = to_denoised(noisy, velocity, sigma)
    assert denoised.shape == x.shape
    print("  to_denoised: OK")


def run_component_tests():
    """Run all component tests."""
    print("\n=== Component Tests ===\n")

    print("Testing schedulers...")
    test_sigma_schedule()

    print("\nTesting guiders...")
    test_cfg_guider()

    print("\nTesting noisers...")
    test_gaussian_noiser()

    print("\nTesting diffusion steps...")
    test_euler_step()

    print("\nTesting patchifiers...")
    test_patchifier()

    print("\nTesting utils...")
    test_rms_norm()
    test_to_velocity_denoised()

    print("\n=== All Component Tests Passed! ===\n")


if __name__ == "__main__":
    run_component_tests()
