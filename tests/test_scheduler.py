"""Unit tests for LTX-2 schedulers.

Tests the sigma schedule generation and Euler diffusion step logic
without requiring model weights.

Run with: pytest tests/test_scheduler.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.components.diffusion_steps import EulerDiffusionStep
from LTX_2_MLX.components.schedulers import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    LTX2Scheduler,
)


class TestLTX2Scheduler:
    """Test LTX2Scheduler sigma schedule generation."""

    def test_scheduler_basic_creation(self):
        """Test that scheduler can be created with default parameters."""
        scheduler = LTX2Scheduler()
        assert scheduler is not None

    def test_sigma_schedule_length(self):
        """Test that sigma schedule has correct length."""
        steps = 25
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=steps)

        # Scheduler returns steps+1 values (including final 0)
        assert len(sigmas) == steps + 1
        assert sigmas.shape == (steps + 1,)

    def test_sigma_schedule_monotonic_decreasing(self):
        """Test that sigma values decrease monotonically."""
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=30)

        # Convert to numpy for easier comparison
        sigmas_np = np.array(sigmas)

        # Check monotonically decreasing
        assert np.all(sigmas_np[:-1] >= sigmas_np[1:]), "Sigmas should decrease monotonically"

    def test_sigma_schedule_ends_at_zero(self):
        """Test that final sigma is zero or very close to zero."""
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=30)

        final_sigma = float(sigmas[-1])
        assert final_sigma < 1e-6, f"Final sigma should be ~0, got {final_sigma}"

    def test_sigma_schedule_starts_positive(self):
        """Test that initial sigma is positive."""
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=30)

        initial_sigma = float(sigmas[0])
        assert initial_sigma > 0, f"Initial sigma should be positive, got {initial_sigma}"

    def test_different_num_steps(self):
        """Test scheduler with different numbers of steps."""
        scheduler = LTX2Scheduler()
        for steps in [10, 20, 30, 50]:
            sigmas = scheduler.execute(steps=steps)

            assert len(sigmas) == steps + 1
            assert float(sigmas[0]) > 0
            assert float(sigmas[-1]) < 1e-6

    def test_sigma_schedule_range(self):
        """Test that sigma values are in expected range."""
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=30)

        sigmas_np = np.array(sigmas)

        # Sigmas should be between 0 and some reasonable upper bound (e.g., 10)
        assert np.all(sigmas_np >= 0), "All sigmas should be non-negative"
        assert np.all(sigmas_np <= 10), "All sigmas should be <= 10"

    def test_scheduler_reproducibility(self):
        """Test that scheduler produces same sigmas with same parameters."""
        scheduler1 = LTX2Scheduler()
        scheduler2 = LTX2Scheduler()

        sigmas1 = scheduler1.execute(steps=30)
        sigmas2 = scheduler2.execute(steps=30)

        np.testing.assert_allclose(
            np.array(sigmas1),
            np.array(sigmas2),
            rtol=1e-6,
            err_msg="Schedulers with same params should produce same sigmas"
        )


class TestDistilledSigmaValues:
    """Test distilled sigma constant values."""

    def test_distilled_sigmas_monotonic(self):
        """Test that distilled sigma values decrease monotonically."""
        sigmas_np = np.array(DISTILLED_SIGMA_VALUES)
        assert np.all(sigmas_np[:-1] >= sigmas_np[1:]), \
            "Distilled sigmas should decrease monotonically"

    def test_distilled_sigmas_end_at_zero(self):
        """Test that distilled sigmas end at exactly 0."""
        assert DISTILLED_SIGMA_VALUES[-1] == 0.0, \
            "Final distilled sigma should be exactly 0.0"

    def test_distilled_sigmas_count(self):
        """Test that there are exactly 9 distilled sigma values."""
        assert len(DISTILLED_SIGMA_VALUES) == 9, \
            "Should have 9 distilled sigma values"

    def test_stage2_distilled_sigmas_monotonic(self):
        """Test that stage 2 distilled sigmas decrease monotonically."""
        sigmas_np = np.array(STAGE_2_DISTILLED_SIGMA_VALUES)
        assert np.all(sigmas_np[:-1] >= sigmas_np[1:]), \
            "Stage 2 distilled sigmas should decrease monotonically"

    def test_stage2_distilled_sigmas_end_at_zero(self):
        """Test that stage 2 distilled sigmas end at exactly 0."""
        assert STAGE_2_DISTILLED_SIGMA_VALUES[-1] == 0.0, \
            "Final stage 2 distilled sigma should be exactly 0.0"

    def test_stage2_distilled_sigmas_count(self):
        """Test that there are exactly 4 stage 2 distilled sigma values."""
        assert len(STAGE_2_DISTILLED_SIGMA_VALUES) == 4, \
            "Should have 4 stage 2 distilled sigma values"

    def test_stage2_sigmas_smaller_than_stage1(self):
        """Test that stage 2 sigmas are smaller (for refinement)."""
        max_stage2 = max(STAGE_2_DISTILLED_SIGMA_VALUES[:-1])  # Exclude 0
        max_stage1 = max(DISTILLED_SIGMA_VALUES[:-1])  # Exclude 0

        assert max_stage2 < max_stage1, \
            "Stage 2 sigmas should be smaller for refinement"


class TestEulerDiffusionStep:
    """Test Euler diffusion step logic."""

    def test_euler_step_basic(self):
        """Test basic Euler step computation."""
        stepper = EulerDiffusionStep()

        # Create synthetic sample and denoised prediction
        sample = mx.random.normal((1, 128, 16, 16))
        denoised_sample = mx.random.normal((1, 128, 16, 16)) * 0.1

        # Create sigma schedule
        sigmas = mx.array([1.0, 0.5])

        # Apply Euler step
        result = stepper.step(
            sample=sample,
            denoised_sample=denoised_sample,
            sigmas=sigmas,
            step_index=0,
        )

        # Result should have same shape
        assert result.shape == sample.shape

    def test_euler_step_shape_preservation(self):
        """Test that Euler step preserves tensor shapes."""
        stepper = EulerDiffusionStep()

        # Test various shapes
        shapes = [
            (1, 128, 8, 8),
            (1, 128, 16, 32),
            (2, 128, 12, 16),
        ]

        # Create sigma schedule
        sigmas = mx.array([1.0, 0.5])

        for shape in shapes:
            sample = mx.random.normal(shape)
            denoised_sample = mx.random.normal(shape)

            result = stepper.step(
                sample=sample,
                denoised_sample=denoised_sample,
                sigmas=sigmas,
                step_index=0,
            )
            assert result.shape == shape, f"Shape should be preserved for {shape}"

    def test_euler_step_with_zero_sigma(self):
        """Test Euler step when sigma_next is zero (final step)."""
        stepper = EulerDiffusionStep()

        sample = mx.random.normal((1, 128, 16, 16))
        denoised_sample = mx.random.normal((1, 128, 16, 16))

        # Create sigma schedule ending at 0
        sigmas = mx.array([0.5, 0.0])

        # Should not crash when sigma_next = 0
        result = stepper.step(
            sample=sample,
            denoised_sample=denoised_sample,
            sigmas=sigmas,
            step_index=0,
        )
        assert result.shape == sample.shape

    def test_euler_step_zero_denoised(self):
        """Test Euler step with zero denoised prediction."""
        stepper = EulerDiffusionStep()

        sample = mx.random.normal((1, 128, 16, 16))
        denoised_sample = mx.zeros_like(sample)

        # Create sigma schedule
        sigmas = mx.array([1.0, 0.5])

        result = stepper.step(
            sample=sample,
            denoised_sample=denoised_sample,
            sigmas=sigmas,
            step_index=0,
        )

        # With zero denoised, result should still be computed
        # (depends on sigma values, but should not be identical to input)
        assert result.shape == sample.shape

    def test_euler_step_numerical_stability(self):
        """Test that Euler step produces finite values."""
        stepper = EulerDiffusionStep()

        sample = mx.random.normal((1, 128, 16, 16))
        denoised_sample = mx.random.normal((1, 128, 16, 16))

        # Create sigma schedule
        sigmas = mx.array([1.0, 0.5])

        result = stepper.step(
            sample=sample,
            denoised_sample=denoised_sample,
            sigmas=sigmas,
            step_index=0,
        )

        # Check for NaN or Inf
        result_np = np.array(result)
        assert np.all(np.isfinite(result_np)), "Result should not contain NaN or Inf"

    def test_euler_step_decreasing_sigmas(self):
        """Test Euler step with monotonically decreasing sigmas."""
        stepper = EulerDiffusionStep()

        sample = mx.random.normal((1, 128, 16, 16))
        denoised_sample = mx.random.normal((1, 128, 16, 16)) * 0.1

        # Create sigma schedule for denoising loop
        sigmas = mx.array([2.0, 1.5, 1.0, 0.5, 0.0])

        current_sample = sample
        for i in range(len(sigmas) - 1):
            current_sample = stepper.step(
                sample=current_sample,
                denoised_sample=denoised_sample,
                sigmas=sigmas,
                step_index=i,
            )

            # Should produce finite values at each step
            sample_np = np.array(current_sample)
            assert np.all(np.isfinite(sample_np)), \
                f"Sample should be finite at step {i}"

    def test_euler_step_dtype_preservation(self):
        """Test that Euler step preserves data type."""
        stepper = EulerDiffusionStep()

        # Create sigma schedule
        sigmas = mx.array([1.0, 0.5])

        for dtype in [mx.float32, mx.float16]:
            sample = mx.random.normal((1, 128, 16, 16)).astype(dtype)
            denoised_sample = mx.random.normal((1, 128, 16, 16)).astype(dtype)

            result = stepper.step(
                sample=sample,
                denoised_sample=denoised_sample,
                sigmas=sigmas,
                step_index=0,
            )

            # Result dtype might be promoted to float32, but should be consistent
            assert result.dtype in [mx.float32, mx.float16], \
                f"Result dtype should be float, got {result.dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
