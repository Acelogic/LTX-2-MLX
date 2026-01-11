"""Diffusion stepping strategies for LTX-2 sampling."""

from typing import Protocol, Union

import mlx.core as mx

from LTX_2_MLX.core_utils import to_velocity


class DiffusionStepProtocol(Protocol):
    """Protocol for diffusion sampling steps."""

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
    ) -> mx.array:
        """Take a single diffusion step from current sigma to next."""
        ...


class EulerDiffusionStep:
    """
    First-order Euler method for diffusion sampling.

    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying:
        sample = sample + velocity * dt

    where dt = sigma_next - sigma (negative, moving toward less noise).
    """

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
    ) -> mx.array:
        """
        Take a single Euler diffusion step.

        Args:
            sample: Current noisy sample.
            denoised_sample: Predicted denoised sample from the model.
            sigmas: Full sigma schedule array.
            step_index: Current step index in the schedule.

        Returns:
            Updated sample at the next sigma level.
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma

        velocity = to_velocity(sample, sigma, denoised_sample)

        # Compute in float32 for numerical stability
        sample_f32 = sample.astype(mx.float32)
        velocity_f32 = velocity.astype(mx.float32)

        result = sample_f32 + velocity_f32 * float(dt)

        return result.astype(sample.dtype)


class HeunDiffusionStep:
    """
    Second-order Heun method for diffusion sampling.

    Uses a two-stage predictor-corrector approach for more accurate stepping,
    at the cost of requiring two model evaluations per step.

    Note: This implementation requires a callback for the model evaluation
    at the predicted point. For simpler use cases, prefer EulerDiffusionStep.
    """

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        denoised_at_predicted: mx.array | None = None,
    ) -> mx.array:
        """
        Take a single Heun diffusion step.

        If denoised_at_predicted is not provided, falls back to Euler step.

        Args:
            sample: Current noisy sample.
            denoised_sample: Predicted denoised sample from the model.
            sigmas: Full sigma schedule array.
            step_index: Current step index in the schedule.
            denoised_at_predicted: Optional denoised sample evaluated at the
                predicted point (for the corrector step).

        Returns:
            Updated sample at the next sigma level.
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma

        velocity = to_velocity(sample, sigma, denoised_sample)

        # Predictor step (Euler)
        sample_f32 = sample.astype(mx.float32)
        velocity_f32 = velocity.astype(mx.float32)
        predicted = sample_f32 + velocity_f32 * float(dt)

        # If no corrector evaluation provided, return Euler result
        if denoised_at_predicted is None:
            return predicted.astype(sample.dtype)

        # Corrector step: average the velocities
        velocity_at_predicted = to_velocity(
            predicted.astype(sample.dtype), sigma_next, denoised_at_predicted
        )
        velocity_avg = 0.5 * (velocity_f32 + velocity_at_predicted.astype(mx.float32))

        result = sample_f32 + velocity_avg * float(dt)

        return result.astype(sample.dtype)
