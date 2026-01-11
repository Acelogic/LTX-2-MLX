"""Two-stage text/image-to-video generation pipeline for LTX-2 MLX.

This pipeline provides high-quality video generation using a two-stage approach:
  Stage 1: Generate at half resolution with CFG using LTX2Scheduler
  Stage 2: Upsample 2x and refine using distilled LoRA (no CFG, fast)

This combines the quality of CFG guidance with the speed of distilled refinement.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from ..components import (
    CFGGuider,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    VideoLatentPatchifier,
)
from ..conditioning.item import ConditioningItem
from ..conditioning.latent import VideoConditionByLatentIndex
from ..conditioning.tools import VideoLatentTools
from ..loader import LoRAConfig, fuse_lora_into_weights
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
class TwoStageCFGConfig:
    """Configuration for two-stage CFG pipeline."""

    # Video dimensions (output - full resolution)
    height: int = 480
    width: int = 704
    num_frames: int = 97  # Must be 8k + 1

    # Generation parameters
    seed: int = 42
    fps: float = 24.0
    num_inference_steps: int = 30

    # CFG parameters (for stage 1)
    cfg_scale: float = 3.0

    # LoRA config for stage 2 (distilled refinement)
    distilled_lora_config: Optional[LoRAConfig] = None

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


class TwoStagePipeline:
    """
    Two-stage text/image-to-video generation pipeline.

    This pipeline generates video using a two-stage approach:
    - Stage 1: Generate at half resolution with CFG guidance
    - Stage 2: Upsample 2x and refine using distilled LoRA (fast, no CFG)

    Features:
    - Stage 1 uses LTX2Scheduler with CFG for quality
    - Stage 2 uses distilled sigma values with optional LoRA refinement
    - Supports image conditioning in both stages
    """

    def __init__(
        self,
        transformer: LTXModel,
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        spatial_upscaler: SpatialUpscaler,
    ):
        """
        Initialize the two-stage pipeline.

        Args:
            transformer: LTX transformer model.
            video_encoder: VAE encoder for encoding images.
            video_decoder: VAE decoder for decoding latents to video.
            spatial_upscaler: 2x spatial upscaler for stage 2.
        """
        # Store raw velocity model for LoRA operations
        if isinstance(transformer, X0Model):
            self._velocity_model = transformer.velocity_model
            self.transformer = transformer
        else:
            self._velocity_model = transformer
            # Wrap in X0Model for denoising (velocity -> denoised conversion)
            self.transformer = X0Model(transformer)
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.spatial_upscaler = spatial_upscaler
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.diffusion_step = EulerDiffusionStep()
        self.scheduler = LTX2Scheduler()

        # Store original weights for LoRA switching (flat parameters)
        self._original_weights: Optional[List] = None

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
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> LatentState:
        """Run the denoising loop with CFG guidance (Stage 1)."""
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
                callback("stage1", step_idx + 1, num_steps)

        return video_state

    def _denoise_loop_simple(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        context: mx.array,
        context_mask: mx.array,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> LatentState:
        """Run simple denoising loop without CFG (Stage 2)."""
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Simple denoising - only positive context
            modality = modality_from_state(video_state, context, context_mask, sigma)
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
                callback("stage2", step_idx + 1, num_steps)

        return video_state

    def __call__(
        self,
        positive_encoding: mx.array,
        positive_mask: mx.array,
        negative_encoding: mx.array,
        negative_mask: mx.array,
        config: TwoStageCFGConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> mx.array:
        """
        Generate video using two-stage CFG pipeline.

        Args:
            positive_encoding: Encoded positive prompt [B, T, D].
            positive_mask: Positive text attention mask [B, T].
            negative_encoding: Encoded negative prompt [B, T, D].
            negative_mask: Negative text attention mask [B, T].
            config: Pipeline configuration.
            images: Optional list of image conditions.
            callback: Optional callback(stage, step, total_steps).

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

        # ====== STAGE 1: Half resolution with CFG ======
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

        # Get stage 1 sigmas using LTX2Scheduler
        sigmas = self.scheduler.execute(
            steps=config.num_inference_steps,
            latent=video_state.latent,
        )

        # Add noise
        video_state = noiser(video_state, noise_scale=1.0)

        # Run stage 1 denoising with CFG
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

        stage_1_latent = video_state.latent

        # ====== STAGE 2: Upsample and refine with distilled LoRA ======
        # Upsample the latent 2x
        # CRITICAL: Must un-normalize before upsampling, then re-normalize after
        # This is required by the PyTorch reference implementation to preserve latent distribution
        latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(stage_1_latent)

        # Use bilinear upsampling (spatial upscaler has res block instability)
        b, c, f, h, w = latent_unnorm.shape
        upscaled_unnorm = mx.zeros((b, c, f, h * 2, w * 2), dtype=latent_unnorm.dtype)

        for fi in range(f):
            frame = latent_unnorm[:, :, fi, :, :]  # (B, C, H, W)
            frame_t = frame.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            frame_up = mx.repeat(mx.repeat(frame_t, 2, axis=1), 2, axis=2)  # Nearest neighbor 2x
            frame_out = frame_up.transpose(0, 3, 1, 2)  # (B, H*2, W*2, C) -> (B, C, H*2, W*2)
            upscaled_unnorm[:, :, fi, :, :] = frame_out

        # Re-normalize back to latent space
        upscaled_latent = self.video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
        mx.eval(upscaled_latent)

        # Apply distilled LoRA if provided
        if config.distilled_lora_config is not None:
            from mlx.utils import tree_flatten

            # Store original weights if not already stored (use raw velocity model)
            if self._original_weights is None:
                self._original_weights = list(tree_flatten(self._velocity_model.parameters()))

            # Fuse LoRA weights (takes a list of LoRAConfigs)
            flat_params = dict(tree_flatten(self._velocity_model.parameters()))
            fused_weights = fuse_lora_into_weights(
                flat_params,
                [config.distilled_lora_config],
            )
            self._velocity_model.load_weights(list(fused_weights.items()))
            mx.eval(self._velocity_model.parameters())

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

        # Get stage 2 distilled sigmas
        distilled_sigmas = mx.array(STAGE_2_DISTILLED_SIGMA_VALUES)

        # Add noise at lower scale for refinement
        video_state_2 = noiser(video_state_2, noise_scale=float(distilled_sigmas[0]))

        # Run stage 2 denoising (simple, no CFG)
        video_state_2 = self._denoise_loop_simple(
            video_state=video_state_2,
            sigmas=distilled_sigmas,
            context=positive_encoding,
            context_mask=positive_mask,
            stepper=stepper,
            callback=callback,
        )

        # Restore original weights if LoRA was applied (use raw velocity model)
        if config.distilled_lora_config is not None and self._original_weights is not None:
            self._velocity_model.load_weights(self._original_weights)
            mx.eval(self._velocity_model.parameters())
            self._original_weights = None

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


def create_two_stage_pipeline(
    transformer: LTXModel,
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
    spatial_upscaler: SpatialUpscaler,
) -> TwoStagePipeline:
    """
    Create a two-stage CFG pipeline.

    Args:
        transformer: LTX transformer model.
        video_encoder: VAE encoder.
        video_decoder: VAE decoder.
        spatial_upscaler: 2x spatial upscaler.

    Returns:
        Configured TwoStagePipeline.
    """
    return TwoStagePipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        spatial_upscaler=spatial_upscaler,
    )
