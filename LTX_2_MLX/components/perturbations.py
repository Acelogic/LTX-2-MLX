"""Attention perturbations for STG (Spatio-Temporal Guidance) in LTX-2 MLX.

Perturbations allow fine-grained control over attention during inference by
selectively skipping certain attention operations in specific transformer blocks.
This is used for STG guidance to compute perturbed predictions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import mlx.core as mx


class PerturbationType(Enum):
    """Types of attention perturbations for STG."""

    SKIP_A2V_CROSS_ATTN = "skip_a2v_cross_attn"  # Skip audio-to-video cross attention
    SKIP_V2A_CROSS_ATTN = "skip_v2a_cross_attn"  # Skip video-to-audio cross attention
    SKIP_VIDEO_SELF_ATTN = "skip_video_self_attn"  # Skip video self attention
    SKIP_AUDIO_SELF_ATTN = "skip_audio_self_attn"  # Skip audio self attention


@dataclass(frozen=True)
class Perturbation:
    """
    A single perturbation specifying which attention type to skip and in which blocks.

    Attributes:
        type: The type of attention to skip.
        blocks: List of block indices where this perturbation applies.
                None means all blocks.
    """

    type: PerturbationType
    blocks: Optional[List[int]] = None  # None means all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        """
        Check if this perturbation applies to a specific type and block.

        Args:
            perturbation_type: The attention type to check.
            block: The block index to check.

        Returns:
            True if this perturbation skips the given attention type in the given block.
        """
        if self.type != perturbation_type:
            return False

        if self.blocks is None:
            return True

        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """
    Configuration holding a list of perturbations for a single sample.

    Attributes:
        perturbations: List of perturbations to apply, or None/empty for no perturbations.
    """

    perturbations: Optional[List[Perturbation]] = None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        """
        Check if any perturbation applies to a specific type and block.

        Args:
            perturbation_type: The attention type to check.
            block: The block index to check.

        Returns:
            True if any perturbation skips the given attention type in the given block.
        """
        if self.perturbations is None:
            return False

        return any(
            perturbation.is_perturbed(perturbation_type, block)
            for perturbation in self.perturbations
        )

    @staticmethod
    def empty() -> "PerturbationConfig":
        """Create an empty perturbation config (no perturbations)."""
        return PerturbationConfig(perturbations=[])


@dataclass(frozen=True)
class BatchedPerturbationConfig:
    """
    Perturbation configurations for a batch of samples.

    Provides utilities for generating attention masks based on perturbations.

    Attributes:
        perturbations: List of PerturbationConfig, one per batch sample.
    """

    perturbations: List[PerturbationConfig]

    def mask(
        self,
        perturbation_type: PerturbationType,
        block: int,
        dtype: mx.Dtype = mx.float32,
    ) -> mx.array:
        """
        Generate a mask tensor for the given perturbation type and block.

        Args:
            perturbation_type: The attention type to create a mask for.
            block: The block index.
            dtype: Data type for the mask tensor.

        Returns:
            Mask tensor of shape (batch_size,) where 1 = keep attention, 0 = skip attention.
        """
        mask_values = []
        for perturbation in self.perturbations:
            if perturbation.is_perturbed(perturbation_type, block):
                mask_values.append(0.0)
            else:
                mask_values.append(1.0)

        return mx.array(mask_values, dtype=dtype)

    def mask_like(
        self,
        perturbation_type: PerturbationType,
        block: int,
        values: mx.array,
    ) -> mx.array:
        """
        Generate a mask tensor broadcastable to the given values tensor.

        Args:
            perturbation_type: The attention type to create a mask for.
            block: The block index.
            values: Tensor to match shape for broadcasting.

        Returns:
            Mask tensor broadcastable to values shape.
        """
        mask = self.mask(perturbation_type, block, values.dtype)
        # Reshape for broadcasting: (batch,) -> (batch, 1, 1, ...)
        for _ in range(len(values.shape) - 1):
            mask = mask[:, None]
        return mask

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        """
        Check if any sample in the batch has this perturbation.

        Args:
            perturbation_type: The attention type to check.
            block: The block index.

        Returns:
            True if any sample has this perturbation.
        """
        return any(
            perturbation.is_perturbed(perturbation_type, block)
            for perturbation in self.perturbations
        )

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        """
        Check if all samples in the batch have this perturbation.

        Args:
            perturbation_type: The attention type to check.
            block: The block index.

        Returns:
            True if all samples have this perturbation.
        """
        return all(
            perturbation.is_perturbed(perturbation_type, block)
            for perturbation in self.perturbations
        )

    @staticmethod
    def empty(batch_size: int) -> "BatchedPerturbationConfig":
        """
        Create an empty batched perturbation config.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            BatchedPerturbationConfig with no perturbations.
        """
        return BatchedPerturbationConfig(
            perturbations=[PerturbationConfig.empty() for _ in range(batch_size)]
        )


def create_stg_perturbation(
    skip_video_self_attn: bool = True,
    blocks: Optional[List[int]] = None,
) -> PerturbationConfig:
    """
    Create a typical STG perturbation config.

    STG (Spatio-Temporal Guidance) typically skips video self-attention
    to compute a perturbed prediction for guidance.

    Args:
        skip_video_self_attn: Whether to skip video self attention.
        blocks: Optional list of block indices. None means all blocks.

    Returns:
        PerturbationConfig for STG.
    """
    perturbations = []
    if skip_video_self_attn:
        perturbations.append(
            Perturbation(
                type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                blocks=blocks,
            )
        )
    return PerturbationConfig(perturbations=perturbations)


def create_batched_stg_config(
    batch_size: int,
    skip_video_self_attn: bool = True,
    blocks: Optional[List[int]] = None,
) -> BatchedPerturbationConfig:
    """
    Create a batched STG perturbation config where all samples have the same perturbation.

    Args:
        batch_size: Number of samples in the batch.
        skip_video_self_attn: Whether to skip video self attention.
        blocks: Optional list of block indices. None means all blocks.

    Returns:
        BatchedPerturbationConfig for STG.
    """
    config = create_stg_perturbation(skip_video_self_attn, blocks)
    return BatchedPerturbationConfig(perturbations=[config] * batch_size)
