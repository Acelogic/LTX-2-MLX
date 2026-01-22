"""Two-stage text/image-to-video generation pipeline for LTX-2 MLX.

This pipeline provides high-quality video generation using a two-stage approach:
  Stage 1: Generate at half resolution with CFG using LTX2Scheduler
  Stage 2: Upsample 2x and refine using distilled LoRA (no CFG, fast)

This combines the quality of CFG guidance with the speed of distilled refinement.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx

from .common import (
    ImageCondition,
    apply_conditionings,
    create_image_conditionings,
    modality_from_state,
    audio_modality_from_state,
    post_process_latent,
)
from ..components import (
    CFGGuider,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    VideoLatentPatchifier,
)
from ..components.patchifiers import AudioPatchifier
from ..conditioning.tools import VideoLatentTools, AudioLatentTools
from ..loader import LoRAConfig, fuse_lora_into_weights
from ..model.transformer import LTXModel, Modality, X0Model
from ..model.video_vae.simple_decoder import SimpleVideoDecoder, decode_latent
from ..model.video_vae.simple_encoder import SimpleVideoEncoder
from ..model.video_vae.tiling import TilingConfig, decode_tiled
from ..model.upscaler import SpatialUpscaler
from ..model.audio_vae import AudioDecoder, Vocoder
from ..types import (
    AudioLatentShape,
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
)


def rescale_noise_cfg(
    noise_cfg: mx.array,
    noise_cond: mx.array,
    guidance_rescale: float = 0.7,
) -> mx.array:
    """
    Rescale CFG output to prevent variance explosion.

    Based on "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    (https://arxiv.org/abs/2305.08891). This rescales the CFG output to match
    the variance of the conditioned prediction, preventing oversaturation.

    Args:
        noise_cfg: CFG-guided prediction.
        noise_cond: Conditioned prediction (no CFG).
        guidance_rescale: Blend factor (0=no rescale, 1=full rescale).

    Returns:
        Rescaled CFG prediction.
    """
    # Get statistics
    cfg_std = noise_cfg.std()
    cfg_mean = noise_cfg.mean()
    cond_std = noise_cond.std()
    cond_mean = noise_cond.mean()

    # Rescale CFG output to match conditioned prediction's statistics
    noise_pred_rescaled = (noise_cfg - cfg_mean) / (cfg_std + 1e-8) * cond_std + cond_mean

    # Blend between original and rescaled
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


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
    guidance_rescale: float = 0.7  # Rescale CFG to prevent variance explosion (0=off, 0.7=default)

    # LoRA config for stage 2 (distilled refinement)
    distilled_lora_config: Optional[LoRAConfig] = None

    # Tiling for VAE decoding
    tiling_config: Optional[TilingConfig] = None

    # Compute settings
    dtype: mx.Dtype = mx.float32

    # Audio configuration
    audio_enabled: bool = False
    audio_vae_channels: int = 8
    audio_mel_bins: int = 16
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_downsample_factor: int = 4
    audio_output_sample_rate: int = 24000

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
        audio_decoder: Optional[AudioDecoder] = None,
        vocoder: Optional[Vocoder] = None,
    ):
        """
        Initialize the two-stage pipeline.

        Args:
            transformer: LTX transformer model.
            video_encoder: VAE encoder for encoding images.
            video_decoder: VAE decoder for decoding latents to video.
            spatial_upscaler: 2x spatial upscaler for stage 2.
            audio_decoder: Optional audio VAE decoder for decoding audio latents to mel spectrograms.
            vocoder: Optional vocoder for converting mel spectrograms to waveforms.
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
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)
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

    def _create_audio_tools(
        self,
        target_shape: AudioLatentShape,
    ) -> AudioLatentTools:
        """Create audio latent tools for the target shape."""
        return AudioLatentTools(
            patchifier=self.audio_patchifier,
            target_shape=target_shape,
        )

    def _decode_audio(self, audio_latent: mx.array) -> mx.array:
        """
        Decode audio latent to waveform via VAE decoder + vocoder.

        Args:
            audio_latent: Audio latent tensor [B, C, F, mel_bins].

        Returns:
            Audio waveform tensor [B, channels, samples].
        """
        if self.audio_decoder is None or self.vocoder is None:
            raise ValueError("Audio decoder and vocoder required for audio decoding")

        # Decode latent to mel spectrogram
        mel_spectrogram = self.audio_decoder(audio_latent)
        mx.eval(mel_spectrogram)

        # Convert mel spectrogram to waveform
        waveform = self.vocoder(mel_spectrogram)
        mx.eval(waveform)

        return waveform

    def _denoise_loop_cfg(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        positive_context: mx.array,
        negative_context: mx.array,
        guider: CFGGuider,
        stepper: EulerDiffusionStep,
        guidance_rescale: float = 0.0,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> LatentState:
        """Run the denoising loop with CFG guidance (Stage 1)."""
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Run positive (conditioned) prediction
            pos_modality = modality_from_state(
                video_state, positive_context, sigma
            )
            pos_denoised = self.transformer(pos_modality)

            # Run negative (unconditioned) prediction for CFG
            if guider.enabled():
                neg_modality = modality_from_state(
                    video_state, negative_context, sigma
                )
                neg_denoised = self.transformer(neg_modality)

                # Apply CFG guidance
                denoised = guider.guide(pos_denoised, neg_denoised)

                # Apply guidance rescale to prevent variance explosion
                if guidance_rescale > 0:
                    denoised = rescale_noise_cfg(denoised, pos_denoised, guidance_rescale)
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

            if callback:
                callback("stage1", step_idx + 1, num_steps)

        return video_state

    def _denoise_loop_cfg_av(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: mx.array,
        positive_video_context: mx.array,
        negative_video_context: mx.array,
        positive_audio_context: mx.array,
        negative_audio_context: mx.array,
        guider: CFGGuider,
        stepper: EulerDiffusionStep,
        guidance_rescale: float = 0.0,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[LatentState, LatentState]:
        """
        Run joint audio-video denoising loop with CFG guidance (Stage 1).

        Args:
            video_state: Initial noisy video latent state.
            audio_state: Initial noisy audio latent state.
            sigmas: Sigma schedule.
            positive_video_context: Positive text context for video.
            negative_video_context: Negative text context for video.
            positive_audio_context: Positive text context for audio.
            negative_audio_context: Negative text context for audio.
            guider: CFG guider instance.
            stepper: Diffusion stepper.
            guidance_rescale: Rescale factor to prevent variance explosion.
            callback: Optional callback(stage, step, total_steps).

        Returns:
            Tuple of (denoised_video_state, denoised_audio_state).
        """
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Create positive (conditioned) modalities
            pos_video_modality = modality_from_state(
                video_state, positive_video_context, sigma
            )
            pos_audio_modality = audio_modality_from_state(
                audio_state, positive_audio_context, sigma
            )

            # Run joint forward pass (conditioned)
            pos_video_denoised, pos_audio_denoised = self.transformer(
                pos_video_modality, pos_audio_modality
            )

            # Run negative (unconditioned) prediction for CFG
            if guider.enabled():
                neg_video_modality = modality_from_state(
                    video_state, negative_video_context, sigma
                )
                neg_audio_modality = audio_modality_from_state(
                    audio_state, negative_audio_context, sigma
                )

                neg_video_denoised, neg_audio_denoised = self.transformer(
                    neg_video_modality, neg_audio_modality
                )

                # Apply CFG guidance to both modalities
                video_denoised = guider.guide(pos_video_denoised, neg_video_denoised)
                audio_denoised = guider.guide(pos_audio_denoised, neg_audio_denoised)

                # Apply guidance rescale to prevent variance explosion
                if guidance_rescale > 0:
                    video_denoised = rescale_noise_cfg(video_denoised, pos_video_denoised, guidance_rescale)
                    audio_denoised = rescale_noise_cfg(audio_denoised, pos_audio_denoised, guidance_rescale)
            else:
                video_denoised = pos_video_denoised
                audio_denoised = pos_audio_denoised

            # Post-process with denoise mask
            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised = post_process_latent(
                audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
            )

            # Euler step for both modalities
            new_video_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=video_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )
            new_audio_latent = stepper.step(
                sample=audio_state.latent,
                denoised_sample=audio_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )

            video_state = video_state.replace(latent=new_video_latent)
            audio_state = audio_state.replace(latent=new_audio_latent)

            mx.eval(video_state.latent)
            mx.eval(audio_state.latent)

            if callback:
                callback("stage1", step_idx + 1, num_steps)

        return video_state, audio_state

    def _denoise_loop_simple(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        context: mx.array,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> LatentState:
        """Run simple denoising loop without CFG (Stage 2)."""
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Simple denoising - only positive context
            modality = modality_from_state(video_state, context, sigma)
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

            if callback:
                callback("stage2", step_idx + 1, num_steps)

        return video_state

    def _denoise_loop_simple_av(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: mx.array,
        video_context: mx.array,
        audio_context: mx.array,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[LatentState, LatentState]:
        """Run simple joint audio-video denoising loop without CFG (Stage 2)."""
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Simple denoising - only positive context
            video_modality = modality_from_state(video_state, video_context, sigma)
            audio_modality = audio_modality_from_state(audio_state, audio_context, sigma)
            video_denoised, audio_denoised = self.transformer(video_modality, audio_modality)

            # Post-process with denoise mask
            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised = post_process_latent(
                audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
            )

            # Euler step for both modalities
            new_video_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=video_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )
            new_audio_latent = stepper.step(
                sample=audio_state.latent,
                denoised_sample=audio_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )

            video_state = video_state.replace(latent=new_video_latent)
            audio_state = audio_state.replace(latent=new_audio_latent)

            mx.eval(video_state.latent)
            mx.eval(audio_state.latent)

            if callback:
                callback("stage2", step_idx + 1, num_steps)

        return video_state, audio_state

    def __call__(
        self,
        positive_encoding: mx.array,
        negative_encoding: mx.array,
        config: TwoStageCFGConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
        positive_audio_encoding: Optional[mx.array] = None,
        negative_audio_encoding: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Generate video (and optionally audio) using two-stage CFG pipeline.

        Args:
            positive_encoding: Encoded positive prompt for video [B, T, D].
            negative_encoding: Encoded negative prompt for video [B, T, D].
            config: Pipeline configuration.
            images: Optional list of image conditions.
            callback: Optional callback(stage, step, total_steps).
            positive_audio_encoding: Encoded positive prompt for audio [B, T, D].
                Required when config.audio_enabled is True.
            negative_audio_encoding: Encoded negative prompt for audio [B, T, D].
                Required when config.audio_enabled is True.

        Returns:
            Tuple of (video, audio) where:
                - video: Generated video tensor [F, H, W, C] in pixel space (0-255).
                - audio: Audio waveform [B, channels, samples] at output_sample_rate,
                         or None if audio_enabled is False.
        """
        images = images or []

        # Validate audio parameters
        if config.audio_enabled:
            if positive_audio_encoding is None or negative_audio_encoding is None:
                raise ValueError(
                    "Audio encoding required when audio_enabled is True. "
                    "Provide positive_audio_encoding and negative_audio_encoding."
                )
            if self.audio_decoder is None or self.vocoder is None:
                raise ValueError(
                    "Audio decoder and vocoder required when audio_enabled is True."
                )

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

        # Create initial video state
        video_state = video_tools.create_initial_state(dtype=config.dtype)

        # Apply conditionings
        video_state = apply_conditionings(video_state, stage_1_conditionings, video_tools)

        # Get stage 1 sigmas using LTX2Scheduler
        sigmas = self.scheduler.execute(steps=config.num_inference_steps)

        # Add noise to video
        video_state = noiser(video_state, noise_scale=1.0)

        # Handle audio if enabled
        audio_state = None
        audio_tools = None
        if config.audio_enabled:
            # Create audio latent shape from video duration
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                stage_1_pixel_shape,
                channels=config.audio_vae_channels,
                mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate,
                hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools = self._create_audio_tools(audio_shape)
            audio_state = audio_tools.create_initial_state(dtype=config.dtype)
            audio_state = noiser(audio_state, noise_scale=1.0)

        # Run stage 1 denoising
        if config.audio_enabled and audio_state is not None:
            # Joint audio-video denoising with CFG
            video_state, audio_state = self._denoise_loop_cfg_av(
                video_state=video_state,
                audio_state=audio_state,
                sigmas=sigmas,
                positive_video_context=positive_encoding,
                negative_video_context=negative_encoding,
                positive_audio_context=positive_audio_encoding,
                negative_audio_context=negative_audio_encoding,
                guider=guider,
                stepper=stepper,
                guidance_rescale=config.guidance_rescale,
                callback=callback,
            )
        else:
            # Video-only denoising with CFG
            video_state = self._denoise_loop_cfg(
                video_state=video_state,
                sigmas=sigmas,
                positive_context=positive_encoding,
                negative_context=negative_encoding,
                guider=guider,
                stepper=stepper,
                guidance_rescale=config.guidance_rescale,
                callback=callback,
            )

        # Clear conditioning and unpatchify
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        stage_1_video_latent = video_state.latent
        stage_1_audio_latent = None
        if config.audio_enabled and audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            stage_1_audio_latent = audio_state.latent

        # ====== STAGE 2: Upsample video and refine with distilled LoRA ======
        # Note: Audio doesn't need spatial upscaling, but we still refine it with video
        # CRITICAL: Must un-normalize before upsampling, then re-normalize after
        # This preserves the latent distribution through the upsampling process
        print("  Upsampling latent 2x with spatial upscaler...")
        # Un-normalize before upscaling (upscaler was trained on un-normalized latents)
        latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(stage_1_video_latent)

        # Apply learned spatial 2x upscaler
        upscaled_unnorm = self.spatial_upscaler(latent_unnorm)
        mx.eval(upscaled_unnorm)

        # Re-normalize back to latent space
        upscaled_video_latent = self.video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
        mx.eval(upscaled_video_latent)

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
        stage_2_video_latent_shape = VideoLatentShape.from_pixel_shape(
            stage_2_pixel_shape, latent_channels=128
        )

        # Create video tools for stage 2
        video_tools_2 = self._create_video_tools(stage_2_video_latent_shape, config.fps)

        # Create conditionings at full resolution
        stage_2_conditionings = create_image_conditionings(
            images,
            self.video_encoder,
            config.height,
            config.width,
            config.dtype,
        )

        # Create initial video state from upscaled latent
        video_state_2 = video_tools_2.create_initial_state(
            dtype=config.dtype, initial_latent=upscaled_video_latent
        )

        # Apply conditionings
        video_state_2 = apply_conditionings(
            video_state_2, stage_2_conditionings, video_tools_2
        )

        # Get stage 2 distilled sigmas
        distilled_sigmas = mx.array(STAGE_2_DISTILLED_SIGMA_VALUES)

        # Add noise at lower scale for refinement
        video_state_2 = noiser(video_state_2, noise_scale=float(distilled_sigmas[0]))

        # Handle audio for stage 2
        audio_state_2 = None
        audio_tools_2 = None
        if config.audio_enabled and stage_1_audio_latent is not None:
            # Create audio tools for stage 2 (same shape as stage 1 - no spatial upscaling for audio)
            audio_shape_2 = AudioLatentShape.from_video_pixel_shape(
                stage_2_pixel_shape,
                channels=config.audio_vae_channels,
                mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate,
                hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools_2 = self._create_audio_tools(audio_shape_2)
            audio_state_2 = audio_tools_2.create_initial_state(
                dtype=config.dtype, initial_latent=stage_1_audio_latent
            )
            audio_state_2 = noiser(audio_state_2, noise_scale=float(distilled_sigmas[0]))

        # Run stage 2 denoising (simple, no CFG)
        if config.audio_enabled and audio_state_2 is not None:
            # Joint audio-video refinement
            video_state_2, audio_state_2 = self._denoise_loop_simple_av(
                video_state=video_state_2,
                audio_state=audio_state_2,
                sigmas=distilled_sigmas,
                video_context=positive_encoding,
                audio_context=positive_audio_encoding,
                stepper=stepper,
                callback=callback,
            )
        else:
            # Video-only refinement
            video_state_2 = self._denoise_loop_simple(
                video_state=video_state_2,
                sigmas=distilled_sigmas,
                context=positive_encoding,
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

        final_video_latent = video_state_2.latent

        # Decode to video
        if config.tiling_config:
            video = decode_tiled(final_video_latent, self.video_decoder, config.tiling_config)
        else:
            video = decode_latent(final_video_latent, self.video_decoder)

        # Decode audio if enabled
        audio_waveform = None
        if config.audio_enabled and audio_state_2 is not None and audio_tools_2 is not None:
            audio_state_2 = audio_tools_2.clear_conditioning(audio_state_2)
            audio_state_2 = audio_tools_2.unpatchify(audio_state_2)
            final_audio_latent = audio_state_2.latent
            audio_waveform = self._decode_audio(final_audio_latent)

        return video, audio_waveform


def create_two_stage_pipeline(
    transformer: LTXModel,
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
    spatial_upscaler: SpatialUpscaler,
    audio_decoder: Optional[AudioDecoder] = None,
    vocoder: Optional[Vocoder] = None,
) -> TwoStagePipeline:
    """
    Create a two-stage CFG pipeline.

    Args:
        transformer: LTX transformer model.
        video_encoder: VAE encoder.
        video_decoder: VAE decoder.
        spatial_upscaler: 2x spatial upscaler.
        audio_decoder: Optional audio VAE decoder (required for audio generation).
        vocoder: Optional vocoder (required for audio generation).

    Returns:
        Configured TwoStagePipeline.
    """
    return TwoStagePipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        spatial_upscaler=spatial_upscaler,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
    )
