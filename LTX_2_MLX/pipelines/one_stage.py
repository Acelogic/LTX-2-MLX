"""Single-stage text/image-to-video generation pipeline for LTX-2 MLX.

This pipeline provides standard CFG-based video generation in a single pass:
  - Uses LTX2Scheduler for sigma schedule
  - Classifier-free guidance with positive/negative prompts
  - Optional image conditioning via latent replacement
  - Optional audio generation via AudioVideo transformer

This is the most common pipeline for high-quality video generation.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

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
    VideoLatentPatchifier,
)
from ..components.patchifiers import AudioPatchifier
from ..conditioning.tools import VideoLatentTools, AudioLatentTools
from ..model.transformer import LTXModel, LTXAVModel, X0Model, Modality
from ..model.video_vae.simple_decoder import SimpleVideoDecoder, decode_latent
from ..model.video_vae.simple_encoder import SimpleVideoEncoder
from ..model.video_vae.tiling import TilingConfig, decode_tiled
from ..model.audio_vae import AudioDecoder, Vocoder
from ..types import (
    AudioLatentShape,
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
    fps: float = 25.0  # Match PyTorch default frame_rate for audio latent calculations
    num_inference_steps: int = 30

    # CFG parameters
    cfg_scale: float = 3.0

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
        # For single-stage, resolution must be divisible by 32
        if self.height % 32 != 0 or self.width % 32 != 0:
            raise ValueError(
                f"Resolution ({self.height}x{self.width}) "
                f"must be divisible by 32 for single-stage pipeline."
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
    - Optional joint audio-video generation via AudioVideo transformer
    """

    def __init__(
        self,
        transformer: LTXModel,
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        audio_decoder: Optional[AudioDecoder] = None,
        vocoder: Optional[Vocoder] = None,
    ):
        """
        Initialize the single-stage pipeline.

        Args:
            transformer: LTX transformer model (LTXModel for video-only, LTXAVModel for audio+video).
            video_encoder: VAE encoder for encoding images.
            video_decoder: VAE decoder for decoding latents to video.
            audio_decoder: Optional audio VAE decoder for decoding audio latents to mel spectrograms.
            vocoder: Optional vocoder for converting mel spectrograms to waveforms.
        """
        # Wrap transformer in X0Model if needed
        # LTXModel outputs velocity, but denoising expects denoised (X0) predictions
        if isinstance(transformer, X0Model):
            self.transformer = transformer
        else:
            self.transformer = X0Model(transformer)
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)
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

        # Decode latent to mel spectrogram (output is log-mel, which vocoder expects)
        mel_spectrogram = self.audio_decoder(audio_latent)
        mx.eval(mel_spectrogram)

        # Convert mel spectrogram to waveform (vocoder takes log-mel directly)
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
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> LatentState:
        """
        Run the denoising loop with CFG guidance.

        Args:
            video_state: Initial noisy video latent state.
            sigmas: Sigma schedule.
            positive_context: Positive text context.
            negative_context: Negative text context.
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
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[LatentState, LatentState]:
        """
        Run joint audio-video denoising loop with CFG guidance.

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
            callback: Optional callback(step, total_steps).

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
                callback(step_idx + 1, num_steps)

        return video_state, audio_state

    def __call__(
        self,
        positive_encoding: mx.array,
        negative_encoding: mx.array,
        config: OneStageCFGConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[int, int], None]] = None,
        positive_audio_encoding: Optional[mx.array] = None,
        negative_audio_encoding: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Generate video (and optionally audio) using single-stage CFG pipeline.

        Args:
            positive_encoding: Encoded positive prompt for video [B, T, D].
            negative_encoding: Encoded negative prompt for video [B, T, D].
            config: Pipeline configuration.
            images: Optional list of image conditions.
            callback: Optional callback(step, total_steps).
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

        # Create initial video state
        video_state = video_tools.create_initial_state(dtype=config.dtype)

        # Apply conditionings
        video_state = apply_conditionings(video_state, conditionings, video_tools)

        # Get sigma schedule
        sigmas = self.scheduler.execute(steps=config.num_inference_steps)

        # Add noise to video
        video_state = noiser(video_state, noise_scale=1.0)

        # Handle audio if enabled
        audio_state = None
        audio_tools = None
        if config.audio_enabled:
            # Create audio latent shape from video duration
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                pixel_shape,
                channels=config.audio_vae_channels,
                mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate,
                hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools = self._create_audio_tools(audio_shape)
            audio_state = audio_tools.create_initial_state(dtype=config.dtype)
            audio_state = noiser(audio_state, noise_scale=1.0)

        # Run denoising loop
        if config.audio_enabled and audio_state is not None:
            # Joint audio-video denoising
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
                callback=callback,
            )
        else:
            # Video-only denoising
            video_state = self._denoise_loop_cfg(
                video_state=video_state,
                sigmas=sigmas,
                positive_context=positive_encoding,
                negative_context=negative_encoding,
                guider=guider,
                stepper=stepper,
                callback=callback,
            )

        # Clear conditioning and unpatchify video
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        final_video_latent = video_state.latent

        # Decode video
        if config.tiling_config:
            video = decode_tiled(final_video_latent, self.video_decoder, config.tiling_config)
        else:
            video = decode_latent(final_video_latent, self.video_decoder)

        # Decode audio if enabled
        audio_waveform = None
        if config.audio_enabled and audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            final_audio_latent = audio_state.latent
            audio_waveform = self._decode_audio(final_audio_latent)

        return video, audio_waveform


def create_one_stage_pipeline(
    transformer: LTXModel,
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
    audio_decoder: Optional[AudioDecoder] = None,
    vocoder: Optional[Vocoder] = None,
) -> OneStagePipeline:
    """
    Create a single-stage CFG pipeline.

    Args:
        transformer: LTX transformer model (LTXModel or LTXAVModel).
        video_encoder: VAE encoder.
        video_decoder: VAE decoder.
        audio_decoder: Optional audio VAE decoder (required for audio generation).
        vocoder: Optional vocoder (required for audio generation).

    Returns:
        Configured OneStagePipeline.
    """
    return OneStagePipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
    )
