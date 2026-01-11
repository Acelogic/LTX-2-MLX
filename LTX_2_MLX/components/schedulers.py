"""Sigma schedule generators for LTX-2 diffusion sampling."""

import math
from functools import lru_cache
from typing import Optional, Protocol

import mlx.core as mx
import numpy as np

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class SchedulerProtocol(Protocol):
    """Protocol for sigma schedulers."""

    def execute(self, steps: int, **kwargs) -> mx.array:
        """Generate a sigma schedule with the given number of steps."""
        ...


class LTX2Scheduler:
    """
    Default scheduler for LTX-2 diffusion sampling.

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.
    """

    def execute(
        self,
        steps: int,
        latent: Optional[mx.array] = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> mx.array:
        """
        Generate sigma schedule.

        Args:
            steps: Number of denoising steps.
            latent: Optional latent tensor to compute token count from shape.
            max_shift: Maximum shift value for large token counts.
            base_shift: Base shift value for small token counts.
            stretch: Whether to stretch sigmas to match terminal value.
            terminal: Target terminal sigma value.

        Returns:
            Sigma schedule as an MLX array of shape (steps + 1,).
        """
        # Compute token count from latent shape or use default
        if latent is not None:
            tokens = math.prod(latent.shape[2:])
        else:
            tokens = MAX_SHIFT_ANCHOR

        # Linear spacing from 1.0 to 0.0
        sigmas = mx.linspace(1.0, 0.0, steps + 1)

        # Compute shift based on token count (linear interpolation)
        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        # Apply sigmoid-like transformation
        power = 1
        exp_shift = math.exp(sigma_shift)

        # Avoid division by zero for sigmas == 0
        # sigmas_transformed = exp_shift / (exp_shift + (1/sigmas - 1)^power)
        sigmas_transformed = mx.where(
            sigmas != 0,
            exp_shift / (exp_shift + mx.power(1.0 / sigmas - 1.0, power)),
            mx.zeros_like(sigmas),
        )

        sigmas = sigmas_transformed

        # Stretch sigmas so that final non-zero value matches terminal
        if stretch and steps > 0:
            # The last sigma is always 0 after transformation, so last non-zero is at index steps-1
            one_minus_sigmas = 1.0 - sigmas

            # Get the last non-zero sigma's (1 - sigma) value (second to last element)
            last_one_minus = float(one_minus_sigmas[steps - 1])

            # Scale factor to stretch to terminal
            scale_factor = last_one_minus / (1.0 - terminal)

            # Apply stretching: new_sigma = 1 - (1 - sigma) / scale_factor
            stretched = 1.0 - (one_minus_sigmas / scale_factor)

            # Only apply to non-zero positions (all except the last element which is 0)
            non_zero_mask = sigmas != 0
            sigmas = mx.where(non_zero_mask, stretched, sigmas)

        return sigmas.astype(mx.float32)


class LinearQuadraticScheduler:
    """
    Scheduler with linear steps followed by quadratic steps.

    Produces a sigma schedule that transitions linearly up to a threshold,
    then follows a quadratic curve for the remaining steps.
    """

    def execute(
        self,
        steps: int,
        threshold_noise: float = 0.025,
        linear_steps: Optional[int] = None,
        **_kwargs,
    ) -> mx.array:
        """
        Generate sigma schedule with linear-quadratic transition.

        Args:
            steps: Number of denoising steps.
            threshold_noise: Noise threshold for transition.
            linear_steps: Number of linear steps (defaults to steps // 2).

        Returns:
            Sigma schedule as an MLX array.
        """
        if steps == 1:
            return mx.array([1.0, 0.0], dtype=mx.float32)

        if linear_steps is None:
            linear_steps = steps // 2

        # Linear part
        linear_sigma_schedule = [
            i * threshold_noise / linear_steps for i in range(linear_steps)
        ]

        # Quadratic part
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_sigma_schedule = []

        if quadratic_steps > 0:
            quadratic_coef = threshold_noise_step_diff / (
                linear_steps * quadratic_steps**2
            )
            linear_coef = (
                threshold_noise / linear_steps
                - 2 * threshold_noise_step_diff / (quadratic_steps**2)
            )
            const = quadratic_coef * (linear_steps**2)
            quadratic_sigma_schedule = [
                quadratic_coef * (i**2) + linear_coef * i + const
                for i in range(linear_steps, steps)
            ]

        # Combine and transform
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]

        return mx.array(sigma_schedule, dtype=mx.float32)


class BetaScheduler:
    """
    Scheduler using a beta distribution to sample timesteps.

    Based on: https://arxiv.org/abs/2407.12173
    """

    shift = 2.37
    timesteps_length = 10000

    def execute(
        self, steps: int, alpha: float = 0.6, beta: float = 0.6, **_kwargs
    ) -> mx.array:
        """
        Execute the beta scheduler.

        Args:
            steps: The number of steps to execute the scheduler for.
            alpha: The alpha parameter for the beta distribution.
            beta: The beta parameter for the beta distribution.

        Note:
            The number of steps within `sigmas` theoretically might be less
            than `steps+1`, because of the deduplication of identical timesteps.

        Returns:
            A tensor of sigmas.
        """
        try:
            import scipy.stats
        except ImportError:
            raise ImportError(
                "BetaScheduler requires scipy. Install with: pip install scipy"
            )

        model_sampling_sigmas = _precalculate_model_sampling_sigmas(
            self.shift, self.timesteps_length
        )
        total_timesteps = len(model_sampling_sigmas) - 1

        # Use numpy for beta distribution sampling
        ts = 1 - np.linspace(0, 1, steps, endpoint=False)
        ts = np.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps).tolist()

        # Deduplicate while preserving order
        ts = list(dict.fromkeys(ts))

        sigmas = [float(model_sampling_sigmas[int(t)]) for t in ts] + [0.0]

        return mx.array(sigmas, dtype=mx.float32)


@lru_cache(maxsize=5)
def _precalculate_model_sampling_sigmas(
    shift: float, timesteps_length: int
) -> np.ndarray:
    """Precalculate model sampling sigmas with caching."""
    timesteps = np.arange(1, timesteps_length + 1) / timesteps_length
    return np.array([flux_time_shift(shift, 1.0, t) for t in timesteps])


def flux_time_shift(mu: float, sigma: float, t: float) -> float:
    """Compute flux time shift transformation."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


# Predefined sigma values for distilled models
# Official distilled sigma schedule from LTX-2 ComfyUI (9 values for 8 steps)
DISTILLED_SIGMA_VALUES = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

STAGE_2_DISTILLED_SIGMA_VALUES = [
    0.3,
    0.15,
    0.05,
    0.0,
]


def get_sigma_schedule(
    num_steps: int,
    distilled: bool = False,
    latent: Optional[mx.array] = None,
) -> mx.array:
    """
    Get sigma schedule for diffusion sampling.

    Convenience function to get the appropriate sigma schedule.

    Args:
        num_steps: Number of denoising steps.
        distilled: If True, use predefined distilled sigma values.
        latent: Optional latent for token-count-dependent scheduling.

    Returns:
        Sigma schedule as MLX array.
    """
    if distilled:
        return mx.array(DISTILLED_SIGMA_VALUES, dtype=mx.float32)

    scheduler = LTX2Scheduler()
    return scheduler.execute(steps=num_steps, latent=latent)
