"""Distilled two-stage video generation pipeline for LTX-2 MLX.

This pipeline provides fast video generation using a distilled model:
  Stage 1: Generate at half resolution (7 steps with DISTILLED_SIGMA_VALUES)
  Stage 2: Upsample 2x and refine (3 steps with STAGE_2_DISTILLED_SIGMA_VALUES)

No CFG (negative prompts) required - uses simple denoising for speed.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from ..components import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    EulerDiffusionStep,
    GaussianNoiser,
    VideoLatentPatchifier,
)
from ..conditioning.item import ConditioningItem
from ..conditioning.latent import VideoConditionByLatentIndex
from ..conditioning.tools import VideoLatentTools
from ..model.transformer import LTXModel, Modality, X0Model
from ..model.video_vae.simple_decoder import SimpleVideoDecoder, decode_latent
from ..model.video_vae.simple_encoder import SimpleVideoEncoder
from ..model.video_vae.tiling import TilingConfig, decode_tiled
from ..model.upscaler import SpatialUpscaler
from ..types import (
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
)


@dataclass
class DistilledConfig:
    """Configuration for distilled pipeline."""

    # Video dimensions (output - full resolution)
    height: int = 480
    width: int = 704
    num_frames: int = 97  # Must be 8k + 1

    # Generation parameters
    seed: int = 42
    fps: float = 24.0

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
        # For two-stage, resolution must be divisible by 64
        if self.height % 64 != 0 or self.width % 64 != 0:
            raise ValueError(
                f"Resolution ({self.height}x{self.width}) "
                f"must be divisible by 64 for two-stage pipeline."
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


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.

    This pipeline is optimized for speed using a distilled model:
    - Stage 1: Generate at half resolution (7 steps)
    - Stage 2: Upsample 2x and refine (3 steps)

    No CFG is required - uses simple denoising without negative prompts.
    This makes it faster than pipelines requiring CFG.
    """

    def __init__(
        self,
        transformer: LTXModel,
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        spatial_upscaler: SpatialUpscaler,
    ):
        """
        Initialize the distilled pipeline.

        Args:
            transformer: LTX transformer model (distilled weights).
                         Will be wrapped in X0Model if not already wrapped.
            video_encoder: VAE encoder for encoding images.
            video_decoder: VAE decoder for decoding latents to video.
            spatial_upscaler: 2x spatial upscaler.
        """
        # Wrap transformer in X0Model if needed
        # LTXModel outputs velocity, but denoising expects denoised (X0) predictions
        if isinstance(transformer, X0Model):
            self.transformer = transformer
        else:
            self.transformer = X0Model(transformer)
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.spatial_upscaler = spatial_upscaler
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.diffusion_step = EulerDiffusionStep()

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

    def _denoise_loop(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        context: mx.array,
        context_mask: mx.array,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> LatentState:
        """
        Run the denoising loop (simple denoising, no CFG).

        Args:
            video_state: Initial noisy video latent state.
            sigmas: Sigma schedule.
            context: Text context.
            context_mask: Text attention mask.
            stepper: Diffusion stepper.
            callback: Optional callback(step, total_steps).

        Returns:
            Denoised latent state.
        """
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Create modality (simple denoising - only positive context)
            modality = modality_from_state(
                video_state, context, context_mask, sigma
            )

            # Run model - direct denoising without CFG
            denoised = self.transformer(modality)

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
        text_encoding: mx.array,
        text_mask: mx.array,
        config: DistilledConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> mx.array:
        """
        Generate video using distilled two-stage pipeline.

        Args:
            text_encoding: Encoded text prompt [B, T, D].
            text_mask: Text attention mask [B, T].
            config: Pipeline configuration.
            images: Optional list of image conditions.
            callback: Optional callback(stage, step, total_steps).

        Returns:
            Generated video tensor [F, H, W, C] in pixel space (0-255).
        """
        images = images or []

        # Set seed
        mx.random.seed(config.seed)

        # Create noiser and stepper
        noiser = GaussianNoiser()
        stepper = self.diffusion_step

        # ====== STAGE 1: Half resolution generation ======
        stage_1_height = config.height // 2
        stage_1_width = config.width // 2

        # Create stage 1 output shape
        stage_1_pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=stage_1_height,
            width=stage_1_width,
            fps=config.fps,
        )
        stage_1_latent_shape = VideoLatentShape.from_pixel_shape(
            stage_1_pixel_shape, latent_channels=128
        )

        # Create video tools
        video_tools = self._create_video_tools(stage_1_latent_shape, config.fps)

        # Create conditionings at stage 1 resolution
        stage_1_conditionings = create_image_conditionings(
            images,
            self.video_encoder,
            stage_1_height,
            stage_1_width,
            config.dtype,
        )

        # Create initial state
        video_state = video_tools.create_initial_state(dtype=config.dtype)

        # Apply conditionings
        video_state = apply_conditionings(video_state, stage_1_conditionings, video_tools)

        # Get stage 1 sigmas (distilled - 7 steps)
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        # Add noise
        video_state = noiser(video_state, noise_scale=1.0)

        # Stage 1 callback wrapper
        def stage_1_callback(step: int, total: int):
            if callback:
                callback("stage1", step, total)

        # Run stage 1 denoising
        video_state = self._denoise_loop(
            video_state=video_state,
            sigmas=sigmas,
            context=text_encoding,
            context_mask=text_mask,
            stepper=stepper,
            callback=stage_1_callback,
        )

        # Clear conditioning and unpatchify
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        stage_1_latent = video_state.latent

        # ====== STAGE 2: Upsample and refine ======
        # Upsample the latent 2x
        upscaled_latent = self.spatial_upscaler(stage_1_latent)
        mx.eval(upscaled_latent)

        # Create stage 2 output shape (full resolution)
        stage_2_pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.fps,
        )
        stage_2_latent_shape = VideoLatentShape.from_pixel_shape(
            stage_2_pixel_shape, latent_channels=128
        )

        # Create video tools for stage 2
        video_tools_2 = self._create_video_tools(stage_2_latent_shape, config.fps)

        # Create conditionings at full resolution
        stage_2_conditionings = create_image_conditionings(
            images,
            self.video_encoder,
            config.height,
            config.width,
            config.dtype,
        )

        # Create initial state from upscaled latent
        video_state_2 = video_tools_2.create_initial_state(
            dtype=config.dtype, initial_latent=upscaled_latent
        )

        # Apply conditionings
        video_state_2 = apply_conditionings(
            video_state_2, stage_2_conditionings, video_tools_2
        )

        # Get stage 2 (distilled refinement) sigmas - 3 steps
        distilled_sigmas = mx.array(STAGE_2_DISTILLED_SIGMA_VALUES)

        # Add noise at lower scale for refinement
        video_state_2 = noiser(video_state_2, noise_scale=float(distilled_sigmas[0]))

        # Stage 2 callback wrapper
        def stage_2_callback(step: int, total: int):
            if callback:
                callback("stage2", step, total)

        # Run stage 2 denoising
        video_state_2 = self._denoise_loop(
            video_state=video_state_2,
            sigmas=distilled_sigmas,
            context=text_encoding,
            context_mask=text_mask,
            stepper=stepper,
            callback=stage_2_callback,
        )

        # Clear conditioning and unpatchify
        video_state_2 = video_tools_2.clear_conditioning(video_state_2)
        video_state_2 = video_tools_2.unpatchify(video_state_2)

        final_latent = video_state_2.latent

        # Decode to video
        if config.tiling_config:
            video = decode_tiled(final_latent, self.video_decoder, config.tiling_config)
        else:
            video = decode_latent(final_latent, self.video_decoder)

        return video


def create_distilled_pipeline(
    transformer: LTXModel,
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
    spatial_upscaler: SpatialUpscaler,
) -> DistilledPipeline:
    """
    Create a distilled pipeline.

    Args:
        transformer: LTX transformer model (distilled weights).
        video_encoder: VAE encoder.
        video_decoder: VAE decoder.
        spatial_upscaler: 2x spatial upscaler.

    Returns:
        Configured DistilledPipeline.
    """
    return DistilledPipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        spatial_upscaler=spatial_upscaler,
    )
