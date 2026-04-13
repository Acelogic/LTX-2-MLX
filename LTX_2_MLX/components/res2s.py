"""Res2s (second-order) Runge-Kutta coefficients for LTX-2 MLX.

Implements the phi exponential integrator basis functions and RK coefficient
computation used by the Res2sDiffusionStep sampler.
"""

import math


def phi(j: int, neg_h: float) -> float:
    """
    Compute phi_j(z) where z = -h (negative step size in log-space).

    phi_1(z) = (e^z - 1) / z
    phi_2(z) = (e^z - 1 - z) / z^2
    phi_j(z) = (e^z - sum_{k=0}^{j-1} z^k/k!) / z^j

    These functions appear when solving: dx/dt = A*x + g(x,t)

    Args:
        j: Order of the phi function.
        neg_h: Negative step size in log-space.

    Returns:
        The phi_j value.
    """
    if abs(neg_h) < 1e-10:
        # Taylor series for small h: phi_j(0) = 1/j!
        return 1.0 / math.factorial(j)

    # Remainder sum: sum_{k=0}^{j-1} z^k/k!
    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))

    # phi_j(z) = (e^z - remainder) / z^j
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(
    h: float,
    phi_cache: dict,
    c2: float = 0.5,
) -> tuple:
    """
    Compute res_2s Runge-Kutta coefficients for a given step size.

    Args:
        h: Step size in log-space = log(sigma / sigma_next).
        phi_cache: Dictionary to cache phi function results.
        c2: Substep position (default 0.5 = midpoint).

    Returns:
        Tuple of (a21, b1, b2):
            a21: Coefficient for computing intermediate x.
            b1, b2: Coefficients for final combination.
    """
    def get_phi(j: int, neg_h: float) -> float:
        cache_key = (j, neg_h)
        if cache_key in phi_cache:
            return phi_cache[cache_key]
        result = phi(j, neg_h)
        phi_cache[cache_key] = result
        return result

    # Substep coefficient: a21 = c2 * phi_1(-h*c2)
    neg_h_c2 = -h * c2
    phi_1_c2 = get_phi(1, neg_h_c2)
    a21 = c2 * phi_1_c2

    # Final combination weights
    # b2 = phi_2(-h) / c2
    neg_h_full = -h
    phi_2_full = get_phi(2, neg_h_full)
    b2 = phi_2_full / c2

    # b1 = phi_1(-h) - b2
    phi_1_full = get_phi(1, neg_h_full)
    b1 = phi_1_full - b2

    return a21, b1, b2
