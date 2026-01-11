"""Utility functions for LTX-2 MLX."""

from typing import Any, Union

import mlx.core as mx


def rms_norm(
    x: mx.array, weight: mx.array | None = None, eps: float = 1e-6
) -> mx.array:
    """
    Root-mean-square (RMS) normalize `x` over its last dimension.

    Uses optimized mx.fast.rms_norm Metal kernel for efficiency.

    Args:
        x: Input tensor to normalize.
        weight: Optional learnable scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        RMS normalized tensor.
    """
    return mx.fast.rms_norm(x, weight, eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:
    """Check that a config value matches the expected value."""
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: mx.array,
    sigma: Union[float, mx.array],
    denoised_sample: mx.array,
) -> mx.array:
    """
    Convert the sample and its denoised version to velocity.

    Args:
        sample: The noisy sample.
        sigma: The noise level (sigma).
        denoised_sample: The predicted denoised sample.

    Returns:
        Velocity prediction.
    """
    # Convert sigma to scalar if it's an array
    if isinstance(sigma, mx.array):
        sigma = float(sigma.item())

    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")

    # Compute in float32 for stability
    sample_f32 = sample.astype(mx.float32)
    denoised_f32 = denoised_sample.astype(mx.float32)

    velocity = (sample_f32 - denoised_f32) / sigma

    return velocity.astype(sample.dtype)


def to_denoised(
    sample: mx.array,
    velocity: mx.array,
    sigma: Union[float, mx.array],
) -> mx.array:
    """
    Convert the sample and its denoising velocity to denoised sample.

    Args:
        sample: The noisy sample.
        velocity: The velocity prediction.
        sigma: The noise level (sigma).

    Returns:
        Denoised sample prediction.
    """
    # Convert sigma to scalar if needed for computation
    if isinstance(sigma, mx.array):
        sigma_val = sigma.astype(mx.float32)
    else:
        sigma_val = sigma

    # Compute in float32 for stability
    sample_f32 = sample.astype(mx.float32)
    velocity_f32 = velocity.astype(mx.float32)

    denoised = sample_f32 - velocity_f32 * sigma_val

    return denoised.astype(sample.dtype)
