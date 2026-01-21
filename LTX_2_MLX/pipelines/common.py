"""Common utilities shared across LTX-2 MLX pipelines.

This module contains helper functions that are used by multiple pipeline
implementations to avoid code duplication.
"""

import os
from dataclasses import dataclass
from typing import List

import mlx.core as mx
import numpy as np
from PIL import Image

from ..conditioning.item import ConditioningItem
from ..conditioning.latent import VideoConditionByLatentIndex
from ..conditioning.tools import VideoLatentTools
from ..model.transformer import Modality
from ..model.video_vae.simple_encoder import SimpleVideoEncoder
from ..types import LatentState


@dataclass
class ImageCondition:
    """An image condition for replacing latent at a specific frame."""

    image_path: str
    frame_index: int
    strength: float = 0.95


def load_image_tensor(
    image_path: str,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Load an image and prepare for VAE encoding.

    Args:
        image_path: Path to the image file
        height: Target height in pixels
        width: Target width in pixels
        dtype: Output data type

    Returns:
        Image tensor of shape (1, C, 1, H, W) normalized to [-1, 1]

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image format is unsupported or loading fails
    """
    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {e}")

    # Validate format
    if img.mode not in ['RGB', 'RGBA', 'L']:
        raise ValueError(
            f"Unsupported image format: {img.mode}. "
            f"Supported formats: RGB, RGBA, L"
        )

    # Convert to RGB and resize
    img = img.convert("RGB")
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Convert to numpy and normalize to [-1, 1]
    img_np = np.array(img).astype(np.float32) / 127.5 - 1.0

    # Convert to MLX array
    img_mx = mx.array(img_np)
    img_mx = mx.transpose(img_mx, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img_mx = img_mx[None, :, None, :, :]  # (1, C, 1, H, W)

    return img_mx.astype(dtype)


def create_image_conditionings(
    images: List[ImageCondition],
    video_encoder: SimpleVideoEncoder,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> List[ConditioningItem]:
    """Create conditionings that replace latent at specific frame indices.

    Args:
        images: List of image conditioning configurations
        video_encoder: VAE encoder for encoding images
        height: Target height in pixels
        width: Target width in pixels
        dtype: Computation data type

    Returns:
        List of conditioning items

    Raises:
        FileNotFoundError: If any image file doesn't exist
        ValueError: If any image format is unsupported
    """
    conditionings = []

    for img_cond in images:
        # Load and validate image (raises FileNotFoundError/ValueError)
        image_tensor = load_image_tensor(img_cond.image_path, height, width, dtype)

        # Encode to latent space
        encoded_latent = video_encoder(image_tensor)
        mx.eval(encoded_latent)

        # Create conditioning item
        conditioning = VideoConditionByLatentIndex(
            latent=encoded_latent,
            strength=img_cond.strength,
            latent_idx=img_cond.frame_index,
        )
        conditionings.append(conditioning)

    return conditionings


def apply_conditionings(
    latent_state: LatentState,
    conditionings: List[ConditioningItem],
    video_tools: VideoLatentTools,
) -> LatentState:
    """Apply all conditionings to the latent state.

    Args:
        latent_state: Current latent state
        conditionings: List of conditioning items to apply
        video_tools: Tools for latent manipulation

    Returns:
        Updated latent state with conditionings applied
    """
    for conditioning in conditionings:
        latent_state = conditioning.apply_to(latent_state, video_tools)
    return latent_state


def post_process_latent(
    denoised: mx.array,
    denoise_mask: mx.array,
    clean_latent: mx.array,
) -> mx.array:
    """Blend denoised output with clean state based on mask.

    Args:
        denoised: Denoised latent tensor of shape (B, T, C)
        denoise_mask: Mask indicating which regions were denoised (1 = denoised, 0 = clean),
                      shape (B, T) or (B, T, C)
        clean_latent: Original clean latent tensor of shape (B, T, C)

    Returns:
        Blended latent tensor
    """
    # Expand mask for broadcasting if needed: (B, T) -> (B, T, 1)
    if denoise_mask.ndim == 2 and denoised.ndim == 3:
        denoise_mask = mx.expand_dims(denoise_mask, axis=-1)
    return (denoised * denoise_mask + clean_latent * (1 - denoise_mask)).astype(
        denoised.dtype
    )


def timesteps_from_mask(denoise_mask: mx.array, sigma: float) -> mx.array:
    """Compute timesteps from denoise mask and sigma.

    Args:
        denoise_mask: Mask indicating which regions to denoise
        sigma: Current noise level (sigma value from scheduler)

    Returns:
        Timestep tensor scaled by mask
    """
    return denoise_mask * sigma


def modality_from_state(
    state: LatentState,
    context: mx.array,
    sigma: float,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state.

    Args:
        state: Latent state containing latent, denoise_mask, and positions
        context: Text context embeddings
        sigma: Current noise level (sigma value from scheduler)
        enabled: Whether this modality is enabled

    Returns:
        Modality object ready for transformer input
    """
    # PyTorch always uses context_mask=None in modality_from_latent_state
    return Modality(
        enabled=enabled,
        latent=state.latent,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
    )


def audio_modality_from_state(
    state: LatentState,
    context: mx.array,
    sigma: float,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from an audio latent state.

    Args:
        state: Audio latent state containing latent, denoise_mask, and positions
        context: Text context embeddings (audio encoding)
        sigma: Current noise level (sigma value from scheduler)
        enabled: Whether this modality is enabled

    Returns:
        Modality object ready for transformer input
    """
    # Audio uses the same Modality structure as video
    # PyTorch always uses context_mask=None in modality_from_latent_state
    return Modality(
        enabled=enabled,
        latent=state.latent,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
    )
