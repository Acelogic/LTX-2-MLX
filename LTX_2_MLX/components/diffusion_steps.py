"""Diffusion stepping strategies for LTX-2 sampling."""

import math
from typing import Optional, Protocol, Tuple, Union

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


class EulerAncestralDiffusionStep:
    """
    Euler ancestral sampler — adds noise injection at each step.

    Matches ComfyUI's euler_ancestral_cfg_pp sampler behavior.
    At each step:
      1. Compute velocity from denoised prediction
      2. Step to sigma_down (deterministic part)
      3. Add noise scaled by sigma_up (stochastic part)

    The noise injection helps with audio quality and diversity.
    """

    @staticmethod
    def _get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0):
        """Compute sigma_up and sigma_down for ancestral sampling."""
        if sigma_to == 0.0:
            return 0.0, 0.0
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_up, sigma_down

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
    ) -> mx.array:
        """
        Take an ancestral Euler step with noise injection.

        Args:
            sample: Current noisy sample.
            denoised_sample: Predicted denoised sample.
            sigmas: Full sigma schedule.
            step_index: Current step index.

        Returns:
            Updated sample with injected noise.
        """
        sigma = float(sigmas[step_index])
        sigma_next = float(sigmas[step_index + 1])

        sigma_up, sigma_down = self._get_ancestral_step(sigma, sigma_next)

        velocity = to_velocity(sample, mx.array(sigma), denoised_sample)

        # Deterministic step to sigma_down
        dt = sigma_down - sigma
        sample_f32 = sample.astype(mx.float32)
        velocity_f32 = velocity.astype(mx.float32)
        result = sample_f32 + velocity_f32 * dt

        # Stochastic noise injection
        if sigma_up > 0:
            noise = mx.random.normal(shape=result.shape, dtype=mx.float32)
            result = result + noise * sigma_up

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


class Res2sDiffusionStep:
    """
    Second-order diffusion step for res_2s sampling with SDE noise injection.

    Used by the res_2s denoising loop. Advances the sample from the current
    sigma to the next by mixing a deterministic update (from the denoised
    prediction) with injected noise, producing variance-preserving transitions.
    """

    @staticmethod
    def get_sde_coeff(
        sigma_next: float,
        sigma_up: Optional[float] = None,
        sigma_down: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute SDE coefficients (alpha_ratio, sigma_down, sigma_up) for the step.

        Given either sigma_down or sigma_up, returns the mixing coefficients
        for variance-preserving noise injection.

        Args:
            sigma_next: Target sigma value.
            sigma_up: Optional noise injection scale.
            sigma_down: Optional deterministic target sigma.
            sigma_max: Optional maximum sigma for normalization.

        Returns:
            Tuple of (alpha_ratio, sigma_down, sigma_up).
        """
        if sigma_down is not None:
            alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
            val = sigma_next**2 - sigma_down**2 * alpha_ratio**2
            sigma_up = max(val, 0.0) ** 0.5
        elif sigma_up is not None:
            # Clamp to avoid sqrt(negative)
            sigma_up = min(sigma_up, sigma_next * 0.9999)
            sigmax = sigma_max if sigma_max is not None else 1.0
            sigma_signal = sigmax - sigma_next
            sigma_residual = max(sigma_next**2 - sigma_up**2, 0.0) ** 0.5
            alpha_ratio = sigma_signal + sigma_residual
            sigma_down = sigma_residual / alpha_ratio if alpha_ratio != 0 else sigma_next
        else:
            alpha_ratio = 1.0
            sigma_down = sigma_next
            sigma_up = 0.0

        # Handle NaN/zero
        if math.isnan(sigma_up):
            sigma_up = 0.0
        if math.isnan(sigma_down):
            sigma_down = sigma_next
        if math.isnan(alpha_ratio):
            alpha_ratio = 1.0

        return alpha_ratio, sigma_down, sigma_up

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Advance one step with SDE noise injection.

        Args:
            sample: Current noisy sample.
            denoised_sample: Predicted denoised sample.
            sigmas: Full sigma schedule.
            step_index: Current step index.
            noise: Random noise for SDE injection.

        Returns:
            Updated sample at the next sigma level.
        """
        sigma = float(sigmas[step_index])
        sigma_next = float(sigmas[step_index + 1])

        alpha_ratio, sigma_down, sigma_up = self.get_sde_coeff(
            sigma_next, sigma_up=sigma_next * 0.5
        )

        output_dtype = denoised_sample.dtype

        if sigma_up == 0.0 or sigma_next == 0.0:
            return denoised_sample

        # Extract epsilon prediction
        sample_f32 = sample.astype(mx.float32)
        denoised_f32 = denoised_sample.astype(mx.float32)

        eps_next = (sample_f32 - denoised_f32) / (sigma - sigma_next)
        denoised_next = sample_f32 - sigma * eps_next

        # Mix deterministic and stochastic components
        x_noised = alpha_ratio * (denoised_next + sigma_down * eps_next)
        if noise is not None:
            x_noised = x_noised + sigma_up * noise.astype(mx.float32)

        return x_noised.astype(output_dtype)
