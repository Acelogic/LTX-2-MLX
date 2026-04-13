"""Distilled two-stage audio-video generation pipeline for LTX-2 MLX.

This pipeline provides fast video generation using a distilled model:
  Stage 1: Generate at half resolution (7 steps with DISTILLED_SIGMA_VALUES)
  Stage 2: Upsample 2x and refine (3 steps with STAGE_2_DISTILLED_SIGMA_VALUES)

No CFG (negative prompts) required - uses simple denoising for speed.
Supports joint audio-video generation via LTXAVModel.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

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
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    EulerDiffusionStep,
    EulerAncestralDiffusionStep,
    GaussianNoiser,
    VideoLatentPatchifier,
)
from ..components.patchifiers import AudioPatchifier
from ..conditioning.tools import VideoLatentTools, AudioLatentTools
from ..model.transformer import LTXModel, LTXAVModel, LTXModelType, X0Model, Modality
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


@dataclass
class DistilledConfig:
    """Configuration for distilled pipeline."""

    # Video dimensions (output - full resolution)
    height: int = 480
    width: int = 704
    num_frames: int = 97  # Must be 8k + 1

    # Generation parameters
    seed: int = 42
    fps: float = 25.0  # Match PyTorch default frame_rate

    # Tiling for VAE decoding
    tiling_config: Optional[TilingConfig] = None

    # Compute settings
    dtype: mx.Dtype = mx.float32

    # Audio configuration
    audio_enabled: bool = False
    use_internal_audio_branch: bool = True
    audio_vae_channels: int = 8
    audio_mel_bins: int = 16
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_downsample_factor: int = 4
    audio_output_sample_rate: int = 24000

    def _get_tiling_config(self) -> Optional[TilingConfig]:
        """Return tiling config, auto-enabling for larger generations."""
        if self.tiling_config is not None:
            return self.tiling_config
        latent_frames = (self.num_frames - 1) // 8 + 1
        latent_pixels = latent_frames * (self.height // 32) * (self.width // 32)
        if latent_pixels > 4000:
            return TilingConfig.default()
        return None

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


class DistilledPipeline:
    """
    Two-stage distilled audio-video generation pipeline.

    This pipeline is optimized for speed using a distilled model:
    - Stage 1: Generate at half resolution (7 steps)
    - Stage 2: Upsample 2x and refine (3 steps)

    No CFG is required - uses simple denoising without negative prompts.
    Supports joint audio-video generation when audio components are provided.
    Works with both LTXModel (video-only) and LTXAVModel (audio-video).
    """

    def __init__(
        self,
        transformer: Union[LTXModel, LTXAVModel],
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        spatial_upscaler: Optional[SpatialUpscaler] = None,
        audio_decoder: Optional[AudioDecoder] = None,
        vocoder: Optional[Vocoder] = None,
    ):
        # Wrap in X0Model if needed
        # X0Model handles both video-only and audio-video transparently
        if isinstance(transformer, X0Model):
            self.transformer = transformer
        else:
            self.transformer = X0Model(transformer)

        # Detect if the underlying model supports audio-video
        inner = self.transformer.velocity_model if hasattr(self.transformer, 'velocity_model') else transformer
        self.is_av_model = getattr(inner, 'model_type', None) == LTXModelType.AudioVideo

        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.spatial_upscaler = spatial_upscaler
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        # Ancestral sampler for stage 1 (matches ComfyUI euler_ancestral_cfg_pp)
        self.diffusion_step_ancestral = EulerAncestralDiffusionStep()
        # Plain Euler for stage 2 (matches ComfyUI euler_cfg_pp)
        self.diffusion_step = EulerDiffusionStep()

    def _create_video_tools(
        self,
        target_shape: VideoLatentShape,
        fps: float,
    ) -> VideoLatentTools:
        return VideoLatentTools(
            patchifier=self.patchifier,
            target_shape=target_shape,
            fps=fps,
        )

    def _create_audio_tools(
        self,
        target_shape: AudioLatentShape,
    ) -> AudioLatentTools:
        return AudioLatentTools(
            patchifier=AudioPatchifier(patch_size=1),
            target_shape=target_shape,
        )

    @staticmethod
    def _channelwise_normalize_audio(latent: mx.array) -> mx.array:
        """Normalize pure audio noise so amplitude is independent of sequence length.

        Fixes the duration-dependent amplitude bug where 10s clips are 5x quieter
        than 5s clips. Raw Gaussian noise has the same per-token variance regardless
        of duration, but something downstream in the transformer integrates over the
        full sequence, causing longer clips to produce weaker latents.

        By normalizing per-feature over the token dimension (axis=1 for shape
        [B, T, D]) we ensure each feature has unit std for any sequence length T.

        Only apply to pure noise (stage 1 initialization), not stage 2 where the
        latent already carries meaningful signal from stage 1.
        """
        # Step 1: global zero-mean, unit-std across all elements
        x = (latent - mx.mean(latent)) / (mx.std(latent) + 1e-8)
        # Step 2: per-feature unit-std over token dim so stats are length-invariant
        # Shape: (B, T, D) → mean/std over T → (B, 1, D)
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.std(x, axis=1, keepdims=True) + 1e-8
        return (x - mean) / std

    def _decode_audio(self, audio_latent: mx.array) -> Optional[mx.array]:
        """Decode audio latent to waveform via Audio VAE + Vocoder."""
        if self.audio_decoder is None or self.vocoder is None:
            return None
        mel = self.audio_decoder(audio_latent)
        mx.eval(mel)
        waveform = self.vocoder(mel)
        mx.eval(waveform)
        return waveform

    def _denoise_loop_av(
        self,
        video_state: LatentState,
        audio_state: Optional[LatentState],
        sigmas: mx.array,
        video_context: mx.array,
        audio_context: Optional[mx.array],
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[LatentState, Optional[LatentState]]:
        """
        Run joint audio-video denoising loop (simple denoising, no CFG).
        Works with both AV and video-only transformers.
        """
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Create video modality
            video_modality = modality_from_state(video_state, video_context, sigma)

            if self.is_av_model:
                if audio_state is not None:
                    audio_modality = audio_modality_from_state(
                        audio_state, audio_context, sigma
                    )
                else:
                    audio_modality = None

                if audio_modality is not None:
                    result = self.transformer(video_modality, audio_modality)
                else:
                    result = self.transformer(video_modality)

                if isinstance(result, tuple):
                    video_denoised, audio_denoised = result
                else:
                    video_denoised = result
                    audio_denoised = None
            else:
                video_denoised = self.transformer(video_modality)
                audio_denoised = None

            # Post-process video
            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            new_video_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=video_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )
            video_state = video_state.replace(latent=new_video_latent)
            mx.eval(video_state.latent)

            # Post-process audio
            if audio_state is not None and audio_denoised is not None:
                audio_denoised = post_process_latent(
                    audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
                )
                new_audio_latent = stepper.step(
                    sample=audio_state.latent,
                    denoised_sample=audio_denoised,
                    sigmas=sigmas,
                    step_index=step_idx,
                )
                audio_state = audio_state.replace(latent=new_audio_latent)
                mx.eval(audio_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        return video_state, audio_state

    def __call__(
        self,
        text_encoding: mx.array,
        text_mask: mx.array,
        config: DistilledConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
        audio_encoding: Optional[mx.array] = None,
    ) -> Union[mx.array, Tuple[mx.array, Optional[mx.array]]]:
        """
        Generate video using distilled two-stage pipeline.

        Args:
            text_encoding: Encoded text prompt [B, T, D] (video context).
            text_mask: Text attention mask [B, T].
            config: Pipeline configuration.
            images: Optional list of image conditions.
            callback: Optional callback(stage, step, total_steps).
            audio_encoding: Optional audio text encoding [B, T, D].

        Returns:
            If audio_enabled: Tuple of (video, audio_waveform).
            Otherwise: video tensor.
            Video is (B, C, T, H, W) float32 in [-1, 1].
        """
        images = images or []

        # Set seed
        mx.random.seed(config.seed)

        # Create noiser and steppers
        noiser = GaussianNoiser()
        stage_1_stepper = self.diffusion_step
        stage_2_stepper = self.diffusion_step

        # ====== STAGE 1: Half resolution generation ======
        stage_1_height = config.height // 2
        stage_1_width = config.width // 2

        # Create stage 1 video shape
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

        video_tools = self._create_video_tools(stage_1_latent_shape, config.fps)

        # Create conditionings at stage 1 resolution
        stage_1_conditionings = create_image_conditionings(
            images, self.video_encoder, stage_1_height, stage_1_width, config.dtype,
        )

        # Create initial video state
        video_state = video_tools.create_initial_state(dtype=config.dtype)
        video_state = apply_conditionings(video_state, stage_1_conditionings, video_tools)

        # Get stage 1 sigmas (distilled - 7 steps)
        sigmas = mx.array(DISTILLED_SIGMA_VALUES)

        # Add noise to video
        video_state = noiser(video_state, noise_scale=1.0)

        internal_audio_active = self.is_av_model and (config.use_internal_audio_branch or config.audio_enabled)

        # Create audio state for AV models when the internal branch is enabled.
        audio_state = None
        audio_tools = None
        if internal_audio_active:
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
            # Normalize audio noise to be length-invariant (fixes duration-dependent
            # amplitude bug: 10s clips 5x quieter than 5s clips without this)
            audio_state = audio_state.replace(
                latent=self._channelwise_normalize_audio(audio_state.latent)
            )

        # Stage 1 callback
        def stage_1_callback(step: int, total: int):
            if callback:
                callback("stage1", step, total)

        # Run stage 1 denoising (ancestral sampler for better audio diversity)
        video_state, audio_state = self._denoise_loop_av(
            video_state=video_state,
            audio_state=audio_state,
            sigmas=sigmas,
            video_context=text_encoding,
            audio_context=audio_encoding,
            stepper=stage_1_stepper,
            callback=stage_1_callback,
        )

        # Clear conditioning and unpatchify video
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        stage_1_latent = video_state.latent

        # Unpatchify audio from stage 1 (needed for stage 2 refinement)
        stage_1_audio_latent = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            stage_1_audio_latent = audio_state.latent

        # ====== STAGE 2: Spatial upscale + refinement ======
        if self.spatial_upscaler is not None:
            print("  Upsampling latent 2x with spatial upscaler...")
            # Un-normalize before upscaling (upscaler was trained on un-normalized latents)
            latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(stage_1_latent)

            # Apply learned spatial 2x upscaler
            upscaled_unnorm = self.spatial_upscaler(latent_unnorm)
            mx.eval(upscaled_unnorm)

            # Re-normalize back to latent space
            upscaled_video_latent = self.video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
            mx.eval(upscaled_video_latent)

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
                images, self.video_encoder, config.height, config.width, config.dtype,
            )

            # Create initial video state from upscaled latent
            video_state_2 = video_tools_2.create_initial_state(
                dtype=config.dtype, initial_latent=upscaled_video_latent
            )
            video_state_2 = apply_conditionings(video_state_2, stage_2_conditionings, video_tools_2)

            # Get stage 2 distilled sigmas (3 refinement steps)
            stage_2_sigmas = mx.array(STAGE_2_DISTILLED_SIGMA_VALUES)

            # Add noise at lower scale for refinement
            video_state_2 = noiser(video_state_2, noise_scale=float(stage_2_sigmas[0]))

            # Handle audio for stage 2 (no spatial upscaling for audio)
            audio_state_2 = None
            if internal_audio_active:
                audio_shape_2 = AudioLatentShape.from_video_pixel_shape(
                    stage_2_pixel_shape,
                    channels=config.audio_vae_channels,
                    mel_bins=config.audio_mel_bins,
                    sample_rate=config.audio_sample_rate,
                    hop_length=config.audio_hop_length,
                    audio_latent_downsample_factor=config.audio_downsample_factor,
                )
                audio_tools_2 = self._create_audio_tools(audio_shape_2)
                if stage_1_audio_latent is not None:
                    audio_state_2 = audio_tools_2.create_initial_state(
                        dtype=config.dtype, initial_latent=stage_1_audio_latent
                    )
                else:
                    audio_state_2 = audio_tools_2.create_initial_state(dtype=config.dtype)
                audio_state_2 = noiser(audio_state_2, noise_scale=float(stage_2_sigmas[0]))

            # Stage 2 callback
            def stage_2_callback(step: int, total: int):
                if callback:
                    callback("stage2", step, total)

            # Run stage 2 denoising (plain Euler, no CFG)
            video_state_2, audio_state_2 = self._denoise_loop_av(
                video_state=video_state_2,
                audio_state=audio_state_2,
                sigmas=stage_2_sigmas,
                video_context=text_encoding,
                audio_context=audio_encoding,
                stepper=stage_2_stepper,
                callback=stage_2_callback,
            )

            # Unpatchify stage 2 results
            video_state_2 = video_tools_2.clear_conditioning(video_state_2)
            video_state_2 = video_tools_2.unpatchify(video_state_2)
            final_video_latent = video_state_2.latent

            # Update audio from stage 2 if available
            if audio_state_2 is not None:
                audio_state_2 = audio_tools_2.clear_conditioning(audio_state_2)
                audio_state_2 = audio_tools_2.unpatchify(audio_state_2)
                stage_1_audio_latent = audio_state_2.latent
        else:
            # No upscaler — output at half resolution
            final_video_latent = stage_1_latent

        # Decode video (auto-tile for large generations)
        effective_tiling = config._get_tiling_config()
        if effective_tiling:
            print(f"  Using tiled VAE decoding (preventing GPU watchdog timeout)")
            video_chunks = list(decode_tiled(final_video_latent, self.video_decoder, effective_tiling))
            video = mx.concatenate(video_chunks, axis=2) if len(video_chunks) > 1 else video_chunks[0]
        else:
            video = decode_latent(final_video_latent, self.video_decoder)

        # Decode audio
        audio_waveform = None
        if stage_1_audio_latent is not None:
            audio_waveform = self._decode_audio(stage_1_audio_latent)

        if config.audio_enabled:
            return video, audio_waveform
        return video


def create_distilled_pipeline(
    transformer: Union[LTXModel, LTXAVModel],
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
    spatial_upscaler: SpatialUpscaler,
    audio_decoder: Optional[AudioDecoder] = None,
    vocoder: Optional[Vocoder] = None,
) -> DistilledPipeline:
    return DistilledPipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        spatial_upscaler=spatial_upscaler,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
    )
