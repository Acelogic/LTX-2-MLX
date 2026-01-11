"""Guidance strategies for LTX-2 diffusion sampling."""

from dataclasses import dataclass, field
from typing import Optional, Protocol

import mlx.core as mx


class GuiderProtocol(Protocol):
    """Protocol for guidance strategies."""

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        """Compute the guidance delta between conditioned and unconditioned samples."""
        ...

    def guide(self, cond: mx.array, uncond: mx.array) -> mx.array:
        """Apply guidance to get the final output: cond + delta(cond, uncond)."""
        ...

    def enabled(self) -> bool:
        """Return True if guidance is active."""
        ...


@dataclass(frozen=True)
class CFGGuider:
    """
    Classifier-free guidance (CFG) guider.

    Computes the guidance delta as (scale - 1) * (cond - uncond), steering the
    denoising process toward the conditioned prediction.

    Attributes:
        scale: Guidance strength. 1.0 means no guidance, higher values increase
            adherence to the conditioning.
    """

    scale: float

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        return (self.scale - 1) * (cond - uncond)

    def guide(self, cond: mx.array, uncond: mx.array) -> mx.array:
        return cond + self.delta(cond, uncond)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class CFGStarRescalingGuider:
    """
    Calculates the CFG delta between conditioned and unconditioned samples.

    To minimize offset in the denoising direction and move mostly along the
    conditioning axis within the distribution, the unconditioned sample is
    rescaled in accordance with the norm of the conditioned sample.

    Attributes:
        scale: Global guidance strength. A value of 1.0 corresponds to no extra
            guidance beyond the base model prediction. Values > 1.0 increase
            the influence of the conditioned sample relative to the
            unconditioned one.
    """

    scale: float

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        rescaled_neg = projection_coef(cond, uncond) * uncond
        return (self.scale - 1) * (cond - rescaled_neg)

    def guide(self, cond: mx.array, uncond: mx.array) -> mx.array:
        return cond + self.delta(cond, uncond)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class STGGuider:
    """
    Calculates the STG delta between conditioned and perturbed denoised samples.

    Perturbed samples are the result of the denoising process with perturbations,
    e.g. attentions acting as passthrough for certain layers and modalities.

    Attributes:
        scale: Global strength of the STG guidance. A value of 0.0 disables the
            guidance. Larger values increase the correction applied in the
            direction of (pos_denoised - perturbed_denoised).
    """

    scale: float

    def delta(self, pos_denoised: mx.array, perturbed_denoised: mx.array) -> mx.array:
        return self.scale * (pos_denoised - perturbed_denoised)

    def guide(self, pos_denoised: mx.array, perturbed_denoised: mx.array) -> mx.array:
        return pos_denoised + self.delta(pos_denoised, perturbed_denoised)

    def enabled(self) -> bool:
        return self.scale != 0.0


@dataclass(frozen=True)
class LtxAPGGuider:
    """
    Calculates the APG (adaptive projected guidance) delta.

    To minimize offset in the denoising direction and move mostly along the
    conditioning axis within the distribution, the (cond - uncond) delta is
    decomposed into components parallel and orthogonal to the conditioned
    sample. The `eta` parameter weights the parallel component, while `scale`
    is applied to the orthogonal component.

    Attributes:
        scale: Strength applied to the component of the guidance that is orthogonal
            to the conditioned sample.
        eta: Weight of the component of the guidance that is parallel to the
            conditioned sample.
        norm_threshold: Minimum L2 norm of the guidance delta below which the
            guidance can be reduced or ignored.
    """

    scale: float
    eta: float = 1.0
    norm_threshold: float = 0.0

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        guidance = cond - uncond

        if self.norm_threshold > 0:
            ones = mx.ones_like(guidance)
            # Compute L2 norm over last 3 dimensions
            guidance_norm = mx.sqrt(
                mx.sum(guidance * guidance, axis=[-1, -2, -3], keepdims=True)
            )
            scale_factor = mx.minimum(ones, self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor

        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth

        return g_apg * (self.scale - 1)

    def guide(self, cond: mx.array, uncond: mx.array) -> mx.array:
        return cond + self.delta(cond, uncond)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=False)
class LegacyStatefulAPGGuider:
    """
    Calculates the APG delta with momentum accumulation.

    Similar to LtxAPGGuider but maintains a running average of guidance
    for smoother transitions.

    Attributes:
        scale: Strength applied to the orthogonal component.
        eta: Weight of the parallel component.
        norm_threshold: Minimum L2 norm threshold.
        momentum: Exponential moving-average coefficient for accumulating guidance.
    """

    scale: float
    eta: float
    norm_threshold: float = 5.0
    momentum: float = 0.0
    running_avg: Optional[mx.array] = field(default=None, repr=False)

    def delta(self, cond: mx.array, uncond: mx.array) -> mx.array:
        guidance = cond - uncond

        if self.momentum != 0:
            if self.running_avg is None:
                self.running_avg = guidance
            else:
                self.running_avg = self.momentum * self.running_avg + guidance
            guidance = self.running_avg

        if self.norm_threshold > 0:
            ones = mx.ones_like(guidance)
            guidance_norm = mx.sqrt(
                mx.sum(guidance * guidance, axis=[-1, -2, -3], keepdims=True)
            )
            scale_factor = mx.minimum(ones, self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor

        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth

        return g_apg * self.scale

    def guide(self, cond: mx.array, uncond: mx.array) -> mx.array:
        return cond + self.delta(cond, uncond)

    def enabled(self) -> bool:
        return self.scale != 0.0


def projection_coef(to_project: mx.array, project_onto: mx.array) -> mx.array:
    """
    Compute the projection coefficient of to_project onto project_onto.

    Args:
        to_project: Tensor to project.
        project_onto: Tensor to project onto.

    Returns:
        Scalar coefficient for the projection.
    """
    batch_size = to_project.shape[0]
    positive_flat = to_project.reshape(batch_size, -1)
    negative_flat = project_onto.reshape(batch_size, -1)
    dot_product = mx.sum(positive_flat * negative_flat, axis=1, keepdims=True)
    squared_norm = mx.sum(negative_flat * negative_flat, axis=1, keepdims=True) + 1e-8
    return dot_product / squared_norm
