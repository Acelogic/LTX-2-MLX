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
    CFGStarRescalingGuider,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    VideoLatentPatchifier,
)
from ..components.guiders import GuiderProtocol
from ..components.diffusion_steps import HeunDiffusionStep
from ..components.patchifiers import AudioPatchifier
from ..conditioning.tools import VideoLatentTools, AudioLatentTools
from ..model.transformer import LTXModel, LTXAVModel, LTXModelType, X0Model, Modality
from ..components.guiders import STGGuider
from ..components.perturbations import create_batched_stg_config, BatchedPerturbationConfig
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
    fps: float = 24.0  # Restore MLX known-good short-clip baseline
    num_inference_steps: int = 30

    # CFG parameters (matching LTX-2.3 reference defaults)
    cfg_scale: float = 3.0           # Video text guidance
    audio_cfg_scale: float = 7.0     # Audio text guidance (higher for better audio conditioning)
    rescale_scale: float = 0.7       # Restore MLX known-good short-clip baseline

    # Tiling for VAE decoding (enabled by default to prevent Metal watchdog crashes on long videos)
    tiling_config: Optional[TilingConfig] = None

    def _get_tiling_config(self) -> Optional[TilingConfig]:
        """Return tiling config, auto-enabling for larger generations."""
        if self.tiling_config is not None:
            return self.tiling_config
        # Auto-enable tiling for videos with many frames or high resolution
        # to prevent Metal GPU watchdog timeouts
        latent_frames = (self.num_frames - 1) // 8 + 1
        latent_pixels = latent_frames * (self.height // 32) * (self.width // 32)
        if latent_pixels > 4000:  # ~121 frames at 768x512 = 5,760
            return TilingConfig.default()
        return None

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
        inner = self.transformer.velocity_model if hasattr(self.transformer, "velocity_model") else transformer
        self.is_av_model = getattr(inner, "model_type", None) == LTXModelType.AudioVideo
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

    def _apply_cross_attn_scales(self, scale: float, start_block: int):
        """Set cross-attention scale on late transformer blocks."""
        model = self.transformer.model if hasattr(self.transformer, 'model') else self.transformer
        blocks = getattr(model, 'transformer_blocks', [])
        for i, block in enumerate(blocks):
            if i >= start_block:
                block._cross_attn_scale = scale
            else:
                block._cross_attn_scale = None

    def _clear_cross_attn_scales(self):
        """Remove cross-attention scaling from all blocks."""
        model = self.transformer.model if hasattr(self.transformer, 'model') else self.transformer
        blocks = getattr(model, 'transformer_blocks', [])
        for block in blocks:
            block._cross_attn_scale = None

    def _denoise_loop_cfg(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        positive_context: mx.array,
        negative_context: mx.array,
        guider,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> LatentState:
        """
        Run the denoising loop with guidance and optional STG + GE.

        Args:
            video_state: Initial noisy video latent state.
            sigmas: Sigma schedule.
            positive_context: Positive text context.
            negative_context: Negative text context.
            guider: Any guider implementing GuiderProtocol (CFG, CFG*, APG, etc.).
            stepper: Diffusion stepper.
            callback: Optional callback(step, total_steps).
            stg_guider: Optional STG guider for temporal coherence.
            stg_perturbations: Perturbation config for STG (skip video self-attn).
            stg_cutoff: Apply STG for first N fraction of steps (0.0-1.0).
            ge_gamma: GE (Gradient Estimation) velocity correction factor. 0.0 = disabled.

        Returns:
            Denoised latent state.
        """
        num_steps = len(sigmas) - 1
        prev_velocity = None  # GE velocity tracking

        # Apply cross-attention scaling if non-default
        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)
            print(f"  Cross-attn scaling: {cross_attn_scale}x on blocks {cross_attn_start_block}-47")

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

                # Apply guidance (CFG, CFG*, APG, etc.)
                denoised = guider.guide(pos_denoised, neg_denoised)
            else:
                denoised = pos_denoised

            # Apply STG (Spatio-Temporal Guidance) if enabled and within cutoff
            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_denoised = self.transformer(pos_modality, perturbations=stg_perturbations)
                denoised = stg_guider.guide(denoised, perturbed_denoised)
                del perturbed_denoised

            # Apply GE (Gradient Estimation) velocity correction
            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

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

        # Clean up cross-attention scaling
        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

        return video_state

    def _denoise_loop_heun(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        positive_context: mx.array,
        negative_context: mx.array,
        guider,
        stepper: HeunDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> LatentState:
        """
        Run the denoising loop with Heun (predictor-corrector) stepping.

        Heun uses two model evaluations per step:
        1. Euler prediction at current sigma
        2. Model eval at predicted point (next sigma)
        3. Average velocities for corrector step

        This gives higher accuracy per step at 2x compute cost.
        """
        num_steps = len(sigmas) - 1
        prev_velocity = None

        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            sigma_next = float(sigmas[step_idx + 1])

            # ── First evaluation: denoised at current sigma ──
            pos_modality = modality_from_state(
                video_state, positive_context, sigma
            )
            pos_denoised = self.transformer(pos_modality)

            if guider.enabled():
                neg_modality = modality_from_state(
                    video_state, negative_context, sigma
                )
                neg_denoised = self.transformer(neg_modality)
                denoised = guider.guide(pos_denoised, neg_denoised)
            else:
                denoised = pos_denoised

            # STG on first evaluation
            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_denoised = self.transformer(pos_modality, perturbations=stg_perturbations)
                denoised = stg_guider.guide(denoised, perturbed_denoised)
                del perturbed_denoised

            # GE velocity correction
            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

            denoised = post_process_latent(
                denoised, video_state.denoise_mask, video_state.clean_latent
            )

            # ── Euler prediction to get predicted sample ──
            from ..core_utils import to_velocity
            velocity = to_velocity(video_state.latent, sigmas[step_idx], denoised)
            dt = sigma_next - sigma
            predicted = video_state.latent.astype(mx.float32) + velocity.astype(mx.float32) * float(dt)
            predicted = predicted.astype(video_state.latent.dtype)
            mx.eval(predicted)

            # Skip corrector on last step (sigma_next == 0)
            if sigma_next == 0:
                video_state = video_state.replace(latent=denoised)
                mx.eval(video_state.latent)
                if callback:
                    callback(step_idx + 1, num_steps)
                continue

            # ── Second evaluation: denoised at predicted point ──
            predicted_state = video_state.replace(latent=predicted)
            pos_modality_2 = modality_from_state(
                predicted_state, positive_context, sigma_next
            )
            pos_denoised_2 = self.transformer(pos_modality_2)

            if guider.enabled():
                neg_modality_2 = modality_from_state(
                    predicted_state, negative_context, sigma_next
                )
                neg_denoised_2 = self.transformer(neg_modality_2)
                denoised_at_predicted = guider.guide(pos_denoised_2, neg_denoised_2)
            else:
                denoised_at_predicted = pos_denoised_2

            denoised_at_predicted = post_process_latent(
                denoised_at_predicted, video_state.denoise_mask, video_state.clean_latent
            )

            # ── Heun corrector step ──
            new_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=denoised,
                sigmas=sigmas,
                step_index=step_idx,
                denoised_at_predicted=denoised_at_predicted,
            )

            video_state = video_state.replace(latent=new_latent)
            mx.eval(video_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

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
        video_guider,
        audio_guider,
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> Tuple[LatentState, LatentState]:
        """Run joint audio-video denoising loop with separate guidance per modality."""
        num_steps = len(sigmas) - 1
        need_cfg = video_guider.enabled() or audio_guider.enabled()
        prev_velocity = None

        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)
            print(f"  Cross-attn scaling: {cross_attn_scale}x on blocks {cross_attn_start_block}-47")

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            pos_video_modality = modality_from_state(video_state, positive_video_context, sigma)
            pos_audio_modality = audio_modality_from_state(audio_state, positive_audio_context, sigma)
            pos_video_denoised, pos_audio_denoised = self.transformer(
                pos_video_modality, pos_audio_modality
            )

            if need_cfg:
                neg_video_modality = modality_from_state(video_state, negative_video_context, sigma)
                neg_audio_modality = audio_modality_from_state(audio_state, negative_audio_context, sigma)
                neg_video_denoised, neg_audio_denoised = self.transformer(
                    neg_video_modality, neg_audio_modality
                )
                video_denoised = video_guider.guide(pos_video_denoised, neg_video_denoised)
                audio_denoised = audio_guider.guide(pos_audio_denoised, neg_audio_denoised)
            else:
                video_denoised = pos_video_denoised
                audio_denoised = pos_audio_denoised

            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_video_denoised, _ = self.transformer(
                    pos_video_modality, pos_audio_modality, perturbations=stg_perturbations
                )
                video_denoised = stg_guider.guide(video_denoised, perturbed_video_denoised)

            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - video_denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    video_denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised = post_process_latent(
                audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
            )

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

        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

        return video_state, audio_state

    def _denoise_loop_heun_av(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: mx.array,
        positive_video_context: mx.array,
        negative_video_context: mx.array,
        positive_audio_context: mx.array,
        negative_audio_context: mx.array,
        video_guider,
        audio_guider,
        stepper: HeunDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> Tuple[LatentState, LatentState]:
        """Run joint audio-video denoising loop with Heun stepping."""
        from ..core_utils import to_velocity

        num_steps = len(sigmas) - 1
        need_cfg = video_guider.enabled() or audio_guider.enabled()
        prev_velocity = None

        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            sigma_next = float(sigmas[step_idx + 1])

            pos_video_modality = modality_from_state(video_state, positive_video_context, sigma)
            pos_audio_modality = audio_modality_from_state(audio_state, positive_audio_context, sigma)
            pos_video_denoised, pos_audio_denoised = self.transformer(
                pos_video_modality, pos_audio_modality
            )

            if need_cfg:
                neg_video_modality = modality_from_state(video_state, negative_video_context, sigma)
                neg_audio_modality = audio_modality_from_state(audio_state, negative_audio_context, sigma)
                neg_video_denoised, neg_audio_denoised = self.transformer(
                    neg_video_modality, neg_audio_modality
                )
                video_denoised = video_guider.guide(pos_video_denoised, neg_video_denoised)
                audio_denoised = audio_guider.guide(pos_audio_denoised, neg_audio_denoised)
            else:
                video_denoised = pos_video_denoised
                audio_denoised = pos_audio_denoised

            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_video_denoised, _ = self.transformer(
                    pos_video_modality, pos_audio_modality, perturbations=stg_perturbations
                )
                video_denoised = stg_guider.guide(video_denoised, perturbed_video_denoised)

            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - video_denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    video_denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised = post_process_latent(
                audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
            )

            video_velocity = to_velocity(video_state.latent, sigmas[step_idx], video_denoised)
            audio_velocity = to_velocity(audio_state.latent, sigmas[step_idx], audio_denoised)
            dt = sigma_next - sigma
            predicted_video = video_state.latent.astype(mx.float32) + video_velocity.astype(mx.float32) * float(dt)
            predicted_audio = audio_state.latent.astype(mx.float32) + audio_velocity.astype(mx.float32) * float(dt)
            predicted_video = predicted_video.astype(video_state.latent.dtype)
            predicted_audio = predicted_audio.astype(audio_state.latent.dtype)
            mx.eval(predicted_video)
            mx.eval(predicted_audio)

            if sigma_next == 0:
                video_state = video_state.replace(latent=video_denoised)
                audio_state = audio_state.replace(latent=audio_denoised)
                mx.eval(video_state.latent)
                mx.eval(audio_state.latent)
                if callback:
                    callback(step_idx + 1, num_steps)
                continue

            predicted_video_state = video_state.replace(latent=predicted_video)
            predicted_audio_state = audio_state.replace(latent=predicted_audio)
            pos_video_modality_2 = modality_from_state(
                predicted_video_state, positive_video_context, sigma_next
            )
            pos_audio_modality_2 = audio_modality_from_state(
                predicted_audio_state, positive_audio_context, sigma_next
            )
            pos_video_denoised_2, pos_audio_denoised_2 = self.transformer(
                pos_video_modality_2, pos_audio_modality_2
            )

            if need_cfg:
                neg_video_modality_2 = modality_from_state(
                    predicted_video_state, negative_video_context, sigma_next
                )
                neg_audio_modality_2 = audio_modality_from_state(
                    predicted_audio_state, negative_audio_context, sigma_next
                )
                neg_video_denoised_2, neg_audio_denoised_2 = self.transformer(
                    neg_video_modality_2, neg_audio_modality_2
                )
                video_denoised_at_predicted = video_guider.guide(pos_video_denoised_2, neg_video_denoised_2)
                audio_denoised_at_predicted = audio_guider.guide(pos_audio_denoised_2, neg_audio_denoised_2)
            else:
                video_denoised_at_predicted = pos_video_denoised_2
                audio_denoised_at_predicted = pos_audio_denoised_2

            video_denoised_at_predicted = post_process_latent(
                video_denoised_at_predicted, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised_at_predicted = post_process_latent(
                audio_denoised_at_predicted, audio_state.denoise_mask, audio_state.clean_latent
            )

            new_video_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=video_denoised,
                sigmas=sigmas,
                step_index=step_idx,
                denoised_at_predicted=video_denoised_at_predicted,
            )
            new_audio_latent = stepper.step(
                sample=audio_state.latent,
                denoised_sample=audio_denoised,
                sigmas=sigmas,
                step_index=step_idx,
                denoised_at_predicted=audio_denoised_at_predicted,
            )

            video_state = video_state.replace(latent=new_video_latent)
            audio_state = audio_state.replace(latent=new_audio_latent)
            mx.eval(video_state.latent)
            mx.eval(audio_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

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
        stg_scale: float = 0.0,
        stg_blocks: Optional[List[int]] = None,
        stg_cutoff: float = 1.0,
        guider_override=None,
        ge_gamma: float = 0.0,
        sampler: str = "euler",
        temporal_upscaler=None,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
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

        internal_audio_active = self.is_av_model and (config.use_internal_audio_branch or config.audio_enabled)

        # AV checkpoints can use audio context internally even for silent video output.
        if config.audio_enabled or internal_audio_active:
            if positive_audio_encoding is None or negative_audio_encoding is None:
                raise ValueError(
                    "Audio encoding required for AudioVideo generation. "
                    "Provide positive_audio_encoding and negative_audio_encoding."
                )
        if config.audio_enabled:
            if self.audio_decoder is None or self.vocoder is None:
                raise ValueError(
                    "Audio decoder and vocoder required when audio_enabled is True."
                )

        # Set seed
        mx.random.seed(config.seed)

        # Create components
        noiser = GaussianNoiser()
        stepper = self.diffusion_step

        # Create separate guiders for video and audio (LTX-2.3 reference uses different scales)
        if guider_override is not None:
            video_guider = guider_override
        elif config.rescale_scale > 0:
            video_guider = CFGStarRescalingGuider(scale=config.cfg_scale)
        else:
            video_guider = CFGGuider(scale=config.cfg_scale)
        # Audio always uses standard guiders (APG not tested with audio)
        if config.rescale_scale > 0:
            audio_guider = CFGStarRescalingGuider(scale=config.audio_cfg_scale)
        else:
            audio_guider = CFGGuider(scale=config.audio_cfg_scale)
        # Legacy single guider for video-only path
        guider = video_guider

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

        # Keep an internal audio state for AV checkpoints when requested.
        audio_state = None
        audio_tools = None
        if internal_audio_active:
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
        if internal_audio_active and audio_state is not None:
            # Joint audio-video denoising with separate guidance per modality
            _stg_guider = None
            _stg_perturbations = None
            if stg_scale > 0:
                _stg_guider = STGGuider(scale=stg_scale)
                _stg_perturbations = create_batched_stg_config(
                    batch_size=1,
                    skip_video_self_attn=True,
                    blocks=stg_blocks,
                )
                cutoff_pct = int(stg_cutoff * 100)
                print(f"  STG guidance: scale={stg_scale}, blocks={stg_blocks or 'all'}, cutoff={cutoff_pct}%")

            if sampler == "heun":
                heun_stepper = HeunDiffusionStep()
                print(f"  Using Heun sampler (2x model evals per step)")
                video_state, audio_state = self._denoise_loop_heun_av(
                    video_state=video_state,
                    audio_state=audio_state,
                    sigmas=sigmas,
                    positive_video_context=positive_encoding,
                    negative_video_context=negative_encoding,
                    positive_audio_context=positive_audio_encoding,
                    negative_audio_context=negative_audio_encoding,
                    video_guider=video_guider,
                    audio_guider=audio_guider,
                    stepper=heun_stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
                )
            else:
                video_state, audio_state = self._denoise_loop_cfg_av(
                    video_state=video_state,
                    audio_state=audio_state,
                    sigmas=sigmas,
                    positive_video_context=positive_encoding,
                    negative_video_context=negative_encoding,
                    positive_audio_context=positive_audio_encoding,
                    negative_audio_context=negative_audio_encoding,
                    video_guider=video_guider,
                    audio_guider=audio_guider,
                    stepper=stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
                )
        else:
            # Set up STG if enabled
            _stg_guider = None
            _stg_perturbations = None
            if stg_scale > 0:
                _stg_guider = STGGuider(scale=stg_scale)
                _stg_perturbations = create_batched_stg_config(
                    batch_size=1,
                    skip_video_self_attn=True,
                    blocks=stg_blocks,
                )
                cutoff_pct = int(stg_cutoff * 100)
                print(f"  STG guidance: scale={stg_scale}, blocks={stg_blocks or 'all'}, cutoff={cutoff_pct}%")

            # Video-only denoising — select loop based on sampler
            if sampler == "heun":
                heun_stepper = HeunDiffusionStep()
                print(f"  Using Heun sampler (2x model evals per step)")
                video_state = self._denoise_loop_heun(
                    video_state=video_state,
                    sigmas=sigmas,
                    positive_context=positive_encoding,
                    negative_context=negative_encoding,
                    guider=guider,
                    stepper=heun_stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
                )
            else:
                video_state = self._denoise_loop_cfg(
                    video_state=video_state,
                    sigmas=sigmas,
                    positive_context=positive_encoding,
                    negative_context=negative_encoding,
                    guider=guider,
                    stepper=stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
            )

        # Clear conditioning and unpatchify video
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        final_video_latent = video_state.latent

        # Apply temporal upscaler (2x frame interpolation) if provided
        if temporal_upscaler is not None:
            input_frames = final_video_latent.shape[2]
            print(f"  Temporal upscaling: {input_frames} → {input_frames * 2 - 1} latent frames...")
            # Un-normalize latent (upscaler trained on raw latents)
            std = self.video_decoder.std_of_means.reshape(1, -1, 1, 1, 1)
            mean = self.video_decoder.mean_of_means.reshape(1, -1, 1, 1, 1)
            latent_unnorm = final_video_latent * std + mean
            # Upscale
            latent_upscaled = temporal_upscaler(latent_unnorm)
            mx.eval(latent_upscaled)
            # Re-normalize
            final_video_latent = (latent_upscaled - mean) / std
            mx.eval(final_video_latent)
            del latent_unnorm, latent_upscaled
            output_frames = final_video_latent.shape[2]
            print(f"  Temporal upscale complete: {output_frames} latent frames")

        # Decode video (auto-tile for large generations to prevent Metal watchdog timeout)
        effective_tiling = config._get_tiling_config()
        if effective_tiling:
            print(f"  Using tiled VAE decoding (preventing GPU watchdog timeout)")
            # decode_tiled returns an iterator/generator — collect and concatenate chunks
            video_chunks = list(decode_tiled(final_video_latent, self.video_decoder, effective_tiling))
            video = mx.concatenate(video_chunks, axis=2) if len(video_chunks) > 1 else video_chunks[0]
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
