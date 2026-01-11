"""Keyframe interpolation pipeline for LTX-2 MLX.

This pipeline generates videos by interpolating between keyframe images.
Uses a two-stage approach:
  Stage 1: Generate at half resolution with CFG
  Stage 2: Upsample 2x and refine with distilled sigmas
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from ..components import (
    STAGE_2_DISTILLED_SIGMA_VALUES,
    CFGGuider,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    VideoLatentPatchifier,
)
from ..conditioning.item import ConditioningItem
from ..conditioning.keyframe import VideoConditionByKeyframeIndex
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
class KeyframeInterpolationConfig:
    """Configuration for keyframe interpolation pipeline."""

    # Video dimensions (output)
    height: int = 480
    width: int = 704
    num_frames: int = 97  # Must be 8k + 1

    # Generation parameters
    num_inference_steps: int = 30
    cfg_scale: float = 7.5
    seed: int = 42
    fps: float = 24.0

    # Two-stage parameters
    use_two_stage: bool = True
    stage_2_steps: int = 3  # Refinement steps

    # Tiling
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
        if self.use_two_stage:
            if self.height % 64 != 0 or self.width % 64 != 0:
                raise ValueError(
                    f"For two-stage pipeline, resolution ({self.height}x{self.width}) "
                    f"must be divisible by 64."
                )


@dataclass
class Keyframe:
    """A keyframe for interpolation."""

    image_path: str  # Path to the image file
    frame_index: int  # Target frame index in the output video
    strength: float = 0.95  # Conditioning strength (higher = more faithful)


def load_image_as_tensor(
    image_path: str,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Load an image and prepare it for VAE encoding.

    Args:
        image_path: Path to the image file.
        height: Target height.
        width: Target width.
        dtype: Data type for output tensor.

    Returns:
        Image tensor of shape (1, 3, 1, H, W) for VAE encoding.
    """
    # Load image with PIL
    img = Image.open(image_path).convert("RGB")

    # Resize to target dimensions
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [-1, 1]
    img_np = np.array(img).astype(np.float32) / 127.5 - 1.0

    # Convert to MLX: (H, W, C) -> (1, C, 1, H, W)
    img_mx = mx.array(img_np)
    img_mx = mx.transpose(img_mx, (2, 0, 1))  # (C, H, W)
    img_mx = img_mx[None, :, None, :, :]  # (1, C, 1, H, W)

    return img_mx.astype(dtype)


def create_keyframe_conditionings(
    keyframes: List[Keyframe],
    video_encoder: SimpleVideoEncoder,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> List[ConditioningItem]:
    """
    Create conditioning items from keyframes.

    Args:
        keyframes: List of keyframes with image paths and frame indices.
        video_encoder: VAE encoder for encoding images to latents.
        height: Target height for images.
        width: Target width for images.
        dtype: Data type for tensors.

    Returns:
        List of VideoConditionByKeyframeIndex conditioning items.
    """
    conditionings = []

    for keyframe in keyframes:
        # Load and prepare image
        image_tensor = load_image_as_tensor(
            keyframe.image_path, height, width, dtype
        )

        # Encode with VAE
        encoded_latent = video_encoder(image_tensor)
        mx.eval(encoded_latent)

        # Create conditioning
        conditioning = VideoConditionByKeyframeIndex(
            keyframes=encoded_latent,
            frame_idx=keyframe.frame_index,
            strength=keyframe.strength,
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


class KeyframeInterpolationPipeline:
    """
    Two-stage keyframe interpolation pipeline.

    Stage 1: Generate at half resolution with CFG
    Stage 2: Upsample 2x and refine with distilled sigmas

    The pipeline uses VideoConditionByKeyframeIndex to append keyframe
    guidance tokens to the latent sequence.
    """

    def __init__(
        self,
        transformer: LTXModel,
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        spatial_upscaler: Optional[SpatialUpscaler] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            transformer: LTX transformer model.
            video_encoder: VAE encoder for encoding keyframe images.
            video_decoder: VAE decoder for decoding latents to video.
            spatial_upscaler: Optional 2x spatial upscaler for stage 2.
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
        context_pos: mx.array,
        context_neg: mx.array,
        context_mask_pos: mx.array,
        context_mask_neg: mx.array,
        cfg_guider: CFGGuider,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> LatentState:
        """
        Run the denoising loop with CFG.

        Args:
            video_state: Initial noisy video latent state.
            sigmas: Sigma schedule.
            context_pos: Positive (conditional) text context.
            context_neg: Negative (unconditional) text context.
            context_mask_pos: Positive text attention mask.
            context_mask_neg: Negative text attention mask.
            cfg_guider: CFG guider for guidance.
            stepper: Diffusion stepper.
            callback: Optional callback(step, total_steps).

        Returns:
            Denoised latent state.
        """
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Create positive modality
            pos_modality = modality_from_state(
                video_state, context_pos, context_mask_pos, sigma
            )

            # Run positive pass
            denoised_pos = self.transformer(pos_modality)

            # Run negative pass for CFG
            if cfg_guider.enabled():
                neg_modality = modality_from_state(
                    video_state, context_neg, context_mask_neg, sigma
                )
                denoised_neg = self.transformer(neg_modality)

                # Apply CFG
                denoised = denoised_pos + cfg_guider.delta(denoised_pos, denoised_neg)
            else:
                denoised = denoised_pos

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
        keyframes: List[Keyframe],
        config: KeyframeInterpolationConfig,
        negative_text_encoding: Optional[mx.array] = None,
        negative_text_mask: Optional[mx.array] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> mx.array:
        """
        Generate video by interpolating between keyframes.

        Args:
            text_encoding: Encoded text prompt [B, T, D].
            text_mask: Text attention mask [B, T].
            keyframes: List of keyframes with images and target frame indices.
            config: Pipeline configuration.
            negative_text_encoding: Optional negative prompt encoding for CFG.
            negative_text_mask: Optional negative prompt mask.
            callback: Optional callback(stage, step, total_steps).

        Returns:
            Generated video tensor [F, H, W, C] in pixel space (0-255).
        """
        # Set seed
        mx.random.seed(config.seed)

        # Create null encoding if not provided
        if negative_text_encoding is None:
            negative_text_encoding = mx.zeros_like(text_encoding)
        if negative_text_mask is None:
            negative_text_mask = mx.zeros_like(text_mask)

        # Create CFG guider and noiser
        cfg_guider = CFGGuider(config.cfg_scale)
        noiser = GaussianNoiser()
        stepper = self.diffusion_step

        # ====== STAGE 1: Half resolution generation ======
        stage_1_height = config.height // 2 if config.use_two_stage else config.height
        stage_1_width = config.width // 2 if config.use_two_stage else config.width

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
        stage_1_conditionings = create_keyframe_conditionings(
            keyframes,
            self.video_encoder,
            stage_1_height,
            stage_1_width,
            config.dtype,
        )

        # Create initial state
        video_state = video_tools.create_initial_state(dtype=config.dtype)

        # Apply conditionings
        video_state = apply_conditionings(video_state, stage_1_conditionings, video_tools)

        # Get stage 1 sigmas
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(config.num_inference_steps).astype(mx.float32)

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
            context_pos=text_encoding,
            context_neg=negative_text_encoding,
            context_mask_pos=text_mask,
            context_mask_neg=negative_text_mask,
            cfg_guider=cfg_guider,
            stepper=stepper,
            callback=stage_1_callback,
        )

        # Clear conditioning and unpatchify
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        stage_1_latent = video_state.latent

        if not config.use_two_stage:
            # Single-stage: decode directly
            if config.tiling_config:
                video = decode_tiled(
                    stage_1_latent, self.video_decoder, config.tiling_config
                )
            else:
                video = decode_latent(stage_1_latent, self.video_decoder)
            return video

        # ====== STAGE 2: Upsample and refine ======
        if self.spatial_upscaler is None:
            raise ValueError(
                "Two-stage pipeline requires spatial_upscaler to be provided"
            )

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
        stage_2_conditionings = create_keyframe_conditionings(
            keyframes,
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

        # Get stage 2 (distilled) sigmas
        distilled_sigmas = mx.array(
            STAGE_2_DISTILLED_SIGMA_VALUES[: config.stage_2_steps + 1]
        )

        # Add noise at lower scale for refinement
        video_state_2 = noiser(video_state_2, noise_scale=float(distilled_sigmas[0]))

        # Stage 2 callback wrapper
        def stage_2_callback(step: int, total: int):
            if callback:
                callback("stage2", step, total)

        # Run stage 2 denoising (no CFG for refinement)
        video_state_2 = self._denoise_loop(
            video_state=video_state_2,
            sigmas=distilled_sigmas,
            context_pos=text_encoding,
            context_neg=text_encoding,  # No negative for stage 2
            context_mask_pos=text_mask,
            context_mask_neg=text_mask,
            cfg_guider=CFGGuider(1.0),  # Disable CFG
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


def create_keyframe_pipeline(
    transformer: LTXModel,
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
    spatial_upscaler: Optional[SpatialUpscaler] = None,
) -> KeyframeInterpolationPipeline:
    """
    Create a keyframe interpolation pipeline.

    Args:
        transformer: LTX transformer model.
        video_encoder: VAE encoder.
        video_decoder: VAE decoder.
        spatial_upscaler: Optional 2x spatial upscaler.

    Returns:
        Configured KeyframeInterpolationPipeline.
    """
    return KeyframeInterpolationPipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        spatial_upscaler=spatial_upscaler,
    )
