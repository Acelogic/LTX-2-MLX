"""Single-stage text/image-to-video generation pipeline for LTX-2 MLX.

This pipeline provides standard CFG-based video generation in a single pass:
  - Uses LTX2Scheduler for sigma schedule
  - Classifier-free guidance with positive/negative prompts
  - Optional image conditioning via latent replacement

This is the most common pipeline for high-quality video generation.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from ..components import (
    CFGGuider,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    VideoLatentPatchifier,
)
from ..conditioning.item import ConditioningItem
from ..conditioning.latent import VideoConditionByLatentIndex
from ..conditioning.tools import VideoLatentTools
from ..model.transformer import LTXModel, Modality, X0Model
from ..model.video_vae.simple_decoder import SimpleVideoDecoder, decode_latent
from ..model.video_vae.simple_encoder import SimpleVideoEncoder
from ..model.video_vae.tiling import TilingConfig, decode_tiled
from ..types import (
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
)


@dataclass
class OneStageCFGConfig:
    """Configuration for single-stage CFG pipeline."""

    # Video dimensions
    height: int = 480
    width: int = 704
    num_frames: int = 97  # Must be 8k + 1

    # Generation parameters
    seed: int = 42
    fps: float = 24.0
    num_inference_steps: int = 30

    # CFG parameters
    cfg_scale: float = 3.0

    # Tiling for VAE decoding
    tiling_config: Optional[TilingConfig] = None

    # Compute settings
    dtype: mx.Dtype = mx.float32

    def __post_init__(self):
        if self.num_frames % 8 != 1:
            raise ValueError(
                f"num_frames must be 8*k + 1, got {self.num_frames}. "
                f"Valid values: 1, 9, 17, 25, 33, ..., 121"
            )
        # For single-stage, resolution must be divisible by 32
        if self.height % 32 != 0 or self.width % 32 != 0:
            raise ValueError(
                f"Resolution ({self.height}x{self.width}) "
                f"must be divisible by 32 for single-stage pipeline."
            )


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
    """Load an image and prepare for VAE encoding."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
    img_mx = mx.array(img_np)
    img_mx = mx.transpose(img_mx, (2, 0, 1))  # (C, H, W)
    img_mx = img_mx[None, :, None, :, :]  # (1, C, 1, H, W)
    return img_mx.astype(dtype)


def create_image_conditionings(
    images: List[ImageCondition],
    video_encoder: SimpleVideoEncoder,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> List[ConditioningItem]:
    """Create conditionings that replace latent at specific frame indices."""
    conditionings = []

    for img_cond in images:
        image_tensor = load_image_tensor(img_cond.image_path, height, width, dtype)
        encoded_latent = video_encoder(image_tensor)
        mx.eval(encoded_latent)

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
    """Apply all conditionings to the latent state."""
    for conditioning in conditionings:
        latent_state = conditioning.apply_to(latent_state, video_tools)
    return latent_state


def post_process_latent(
    denoised: mx.array,
    denoise_mask: mx.array,
    clean_latent: mx.array,
) -> mx.array:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean_latent * (1 - denoise_mask)).astype(
        denoised.dtype
    )


def timesteps_from_mask(denoise_mask: mx.array, sigma: float) -> mx.array:
    """Compute timesteps from denoise mask and sigma."""
    return denoise_mask * sigma


def modality_from_state(
    state: LatentState,
    context: mx.array,
    context_mask: mx.array,
    sigma: float,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state."""
    return Modality(
        enabled=enabled,
        latent=state.latent,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=context_mask,
    )


class OneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.

    This pipeline generates video at target resolution in a single diffusion pass
    with classifier-free guidance (CFG). Supports optional image conditioning.

    Features:
    - Uses LTX2Scheduler for sigma schedule
    - CFG with positive/negative prompts for quality
    - Optional image conditioning via latent replacement
    """

    def __init__(
        self,
        transformer: LTXModel,
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
    ):
        """
        Initialize the single-stage pipeline.

        Args:
            transformer: LTX transformer model.
            video_encoder: VAE encoder for encoding images.
            video_decoder: VAE decoder for decoding latents to video.
        """
        # Wrap transformer in X0Model if needed
        # LTXModel outputs velocity, but denoising expects denoised (X0) predictions
        if isinstance(transformer, X0Model):
            self.transformer = transformer
        else:
            self.transformer = X0Model(transformer)
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.diffusion_step = EulerDiffusionStep()
        self.scheduler = LTX2Scheduler()

    def _create_video_tools(
        self,
        target_shape: VideoLatentShape,
        fps: float,
    ) -> VideoLatentTools:
        """Create video latent tools for the target shape."""
        return VideoLatentTools(
            patchifier=self.patchifier,
            target_shape=target_shape,
            fps=fps,
        )

    def _denoise_loop_cfg(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        positive_context: mx.array,
        positive_mask: mx.array,
        negative_context: mx.array,
        negative_mask: mx.array,
        guider: CFGGuider,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> LatentState:
        """
        Run the denoising loop with CFG guidance.

        Args:
            video_state: Initial noisy video latent state.
            sigmas: Sigma schedule.
            positive_context: Positive text context.
            positive_mask: Positive text attention mask.
            negative_context: Negative text context.
            negative_mask: Negative text attention mask.
            guider: CFG guider instance.
            stepper: Diffusion stepper.
            callback: Optional callback(step, total_steps).

        Returns:
            Denoised latent state.
        """
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Run positive (conditioned) prediction
            pos_modality = modality_from_state(
                video_state, positive_context, positive_mask, sigma
            )
            pos_denoised = self.transformer(pos_modality)

            # Run negative (unconditioned) prediction for CFG
            if guider.enabled():
                neg_modality = modality_from_state(
                    video_state, negative_context, negative_mask, sigma
                )
                neg_denoised = self.transformer(neg_modality)

                # Apply CFG guidance
                denoised = guider.guide(pos_denoised, neg_denoised)
            else:
                denoised = pos_denoised

            # Post-process with denoise mask
            denoised = post_process_latent(
                denoised, video_state.denoise_mask, video_state.clean_latent
            )

            # Euler step
            new_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )

            video_state = video_state.replace(latent=new_latent)
            mx.eval(video_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        return video_state

    def __call__(
        self,
        positive_encoding: mx.array,
        positive_mask: mx.array,
        negative_encoding: mx.array,
        negative_mask: mx.array,
        config: OneStageCFGConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> mx.array:
        """
        Generate video using single-stage CFG pipeline.

        Args:
            positive_encoding: Encoded positive prompt [B, T, D].
            positive_mask: Positive text attention mask [B, T].
            negative_encoding: Encoded negative prompt [B, T, D].
            negative_mask: Negative text attention mask [B, T].
            config: Pipeline configuration.
            images: Optional list of image conditions.
            callback: Optional callback(step, total_steps).

        Returns:
            Generated video tensor [F, H, W, C] in pixel space (0-255).
        """
        images = images or []

        # Set seed
        mx.random.seed(config.seed)

        # Create components
        noiser = GaussianNoiser()
        stepper = self.diffusion_step
        guider = CFGGuider(scale=config.cfg_scale)

        # Create output shape
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.fps,
        )
        latent_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=128
        )

        # Create video tools
        video_tools = self._create_video_tools(latent_shape, config.fps)

        # Create image conditionings
        conditionings = create_image_conditionings(
            images,
            self.video_encoder,
            config.height,
            config.width,
            config.dtype,
        )

        # Create initial state
        video_state = video_tools.create_initial_state(dtype=config.dtype)

        # Apply conditionings
        video_state = apply_conditionings(video_state, conditionings, video_tools)

        # Get sigma schedule
        sigmas = self.scheduler.execute(
            steps=config.num_inference_steps,
            latent=video_state.latent,
        )

        # Add noise
        video_state = noiser(video_state, noise_scale=1.0)

        # Run CFG denoising loop
        video_state = self._denoise_loop_cfg(
            video_state=video_state,
            sigmas=sigmas,
            positive_context=positive_encoding,
            positive_mask=positive_mask,
            negative_context=negative_encoding,
            negative_mask=negative_mask,
            guider=guider,
            stepper=stepper,
            callback=callback,
        )

        # Clear conditioning and unpatchify
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        final_latent = video_state.latent

        # Decode to video
        if config.tiling_config:
            video = decode_tiled(final_latent, self.video_decoder, config.tiling_config)
        else:
            video = decode_latent(final_latent, self.video_decoder)

        return video


def create_one_stage_pipeline(
    transformer: LTXModel,
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
) -> OneStagePipeline:
    """
    Create a single-stage CFG pipeline.

    Args:
        transformer: LTX transformer model.
        video_encoder: VAE encoder.
        video_decoder: VAE decoder.

    Returns:
        Configured OneStagePipeline.
    """
    return OneStagePipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
    )
