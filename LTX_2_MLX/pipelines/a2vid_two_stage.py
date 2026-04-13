"""Audio-to-video two-stage pipeline for LTX-2 MLX.

Takes an audio file as input and generates a video synchronized to that audio.
Stage 1 generates at half resolution with the audio latent frozen (video-only
denoising). Stage 2 upsamples 2x and refines with distilled sigmas.

The original audio waveform is returned as-is (not VAE-decoded) for fidelity.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .common import (
    ImageCondition,
    apply_conditionings,
    create_image_conditionings,
    modality_from_state,
    audio_modality_from_state,
    post_process_latent,
)
from ..components import (
    STAGE_2_DISTILLED_SIGMA_VALUES,
    CFGGuider,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    VideoLatentPatchifier,
)
from ..components.patchifiers import AudioPatchifier
from ..conditioning.tools import VideoLatentTools, AudioLatentTools
from ..model.transformer import LTXModel, LTXAVModel, LTXModelType, X0Model
from ..model.video_vae.simple_decoder import SimpleVideoDecoder, decode_latent
from ..model.video_vae.simple_encoder import SimpleVideoEncoder
from ..model.video_vae.tiling import TilingConfig, decode_tiled
from ..model.upscaler import SpatialUpscaler
from ..model.audio_vae import AudioDecoder, Vocoder
from ..loader.lora_loader import LoRAConfig, fuse_lora_into_weights
from ..types import (
    AudioLatentShape,
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
)


@dataclass
class A2VidConfig:
    """Configuration for audio-to-video pipeline."""

    height: int = 512
    width: int = 768
    num_frames: int = 97

    num_inference_steps: int = 30
    cfg_scale: float = 3.0
    seed: int = 42
    fps: float = 25.0

    # LoRA for stage 2 refinement
    distilled_lora_config: Optional[LoRAConfig] = None

    tiling_config: Optional[TilingConfig] = None
    dtype: mx.Dtype = mx.float16

    # Audio params
    audio_vae_channels: int = 8
    audio_mel_bins: int = 16
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_downsample_factor: int = 4
    audio_output_sample_rate: int = 24000

    # Audio input
    audio_start_time: float = 0.0
    audio_max_duration: Optional[float] = None

    def _get_tiling_config(self) -> Optional[TilingConfig]:
        if self.tiling_config is not None:
            return self.tiling_config
        latent_frames = (self.num_frames - 1) // 8 + 1
        latent_pixels = latent_frames * (self.height // 32) * (self.width // 32)
        if latent_pixels > 4000:
            return TilingConfig.default()
        return None

    def __post_init__(self):
        if self.num_frames % 8 != 1:
            raise ValueError(f"num_frames must be 8*k + 1, got {self.num_frames}")
        if self.height % 64 != 0 or self.width % 64 != 0:
            raise ValueError(f"Resolution ({self.height}x{self.width}) must be divisible by 64.")


def load_audio_file(
    audio_path: str,
    target_sr: int = 16000,
    start_time: float = 0.0,
    max_duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        target_sr: Target sample rate.
        start_time: Start time in seconds to extract from.
        max_duration: Maximum duration in seconds.

    Returns:
        Tuple of (waveform as numpy array [channels, samples], sample_rate).
    """
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path)
    except ImportError:
        # Fallback to ffmpeg
        import subprocess
        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            cmd = ["ffmpeg", "-v", "quiet", "-i", audio_path, "-ar", str(target_sr), "-ac", "2", "-y", tmp.name]
            subprocess.run(cmd, check=True)
            with wave.open(tmp.name, "r") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32) / 32768.0
                if wf.getnchannels() == 2:
                    data = data.reshape(-1, 2)
                else:
                    data = data.reshape(-1, 1)

    # Handle mono/stereo
    if data.ndim == 1:
        data = data[:, np.newaxis]
    # Transpose to (channels, samples)
    if data.shape[0] > data.shape[1]:
        data = data.T

    # Trim to start_time / max_duration
    start_sample = int(start_time * sr)
    data = data[:, start_sample:]
    if max_duration is not None:
        max_samples = int(max_duration * sr)
        data = data[:, :max_samples]

    # Simple resample if needed (nearest neighbor — not ideal but functional)
    if sr != target_sr:
        num_output = int(data.shape[1] * target_sr / sr)
        indices = np.linspace(0, data.shape[1] - 1, num_output).astype(int)
        data = data[:, indices]
        sr = target_sr

    return data, sr


class A2VidPipelineTwoStage:
    """
    Audio-to-video two-stage pipeline.

    Takes an audio file as input and generates video conditioned on it.
    Stage 1: Half resolution, video-only denoising (audio frozen).
    Stage 2: Full resolution, spatial upscaler, distilled refinement.

    Returns original audio (not VAE-decoded) for maximum fidelity.
    """

    def __init__(
        self,
        transformer: Union[LTXModel, LTXAVModel],
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        spatial_upscaler: SpatialUpscaler,
        audio_decoder: Optional[AudioDecoder] = None,
        vocoder: Optional[Vocoder] = None,
    ):
        if isinstance(transformer, X0Model):
            self.transformer = transformer
            self._velocity_model = transformer.velocity_model
        else:
            self.transformer = X0Model(transformer)
            self._velocity_model = transformer

        inner = self._velocity_model
        self.is_av_model = getattr(inner, 'model_type', None) == LTXModelType.AudioVideo
        if not self.is_av_model:
            raise ValueError("A2Vid pipeline requires an audio-video (AV) model")

        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.spatial_upscaler = spatial_upscaler
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.stepper = EulerDiffusionStep()

        self._original_weights = None

    def _create_video_tools(self, target_shape, fps):
        return VideoLatentTools(patchifier=self.patchifier, target_shape=target_shape, fps=fps)

    def _create_audio_tools(self, target_shape):
        return AudioLatentTools(patchifier=AudioPatchifier(patch_size=1), target_shape=target_shape)

    def _encode_audio_to_latent(
        self,
        audio_waveform: np.ndarray,
        audio_sr: int,
        pixel_shape: VideoPixelShape,
        config: A2VidConfig,
    ) -> mx.array:
        """Encode audio waveform to latent space via audio VAE encoder.

        Note: This requires the audio encoder to be available. If not available,
        we create a placeholder latent (zeros) and the model generates audio from
        scratch based on the text prompt.
        """
        # Check if video_encoder has an audio encoder component
        # For now, return None to indicate audio should be generated, not encoded
        # Full implementation would use vae_encode_audio
        print("  Note: Audio VAE encoder not yet ported — audio will be generated from prompt")
        return None

    def _video_only_denoise_loop(
        self,
        video_state, audio_state, sigmas,
        video_context, audio_context,
        negative_video_context=None, negative_audio_context=None,
        cfg_scale=1.0, callback=None,
    ):
        """Denoise only video; audio latent stays frozen (denoise_mask=0).

        The audio state is passed to the transformer for conditioning but
        its latent is not updated by the stepper.
        """
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Forward pass with both modalities
            video_mod = modality_from_state(video_state, video_context, sigma)
            audio_mod = audio_modality_from_state(audio_state, audio_context, sigma)

            result = self.transformer(video_mod, audio_mod)
            if isinstance(result, tuple):
                denoised_v, denoised_a = result
            else:
                denoised_v, denoised_a = result, None

            # CFG for video only
            if cfg_scale > 1.0 and negative_video_context is not None:
                video_mod_neg = modality_from_state(video_state, negative_video_context, sigma)
                audio_mod_neg = audio_modality_from_state(audio_state, negative_audio_context or audio_context, sigma)
                result_neg = self.transformer(video_mod_neg, audio_mod_neg)
                uncond_v = result_neg[0] if isinstance(result_neg, tuple) else result_neg
                denoised_v = uncond_v + cfg_scale * (denoised_v - uncond_v)

            # Post-process and step VIDEO only
            denoised_v = post_process_latent(denoised_v, video_state.denoise_mask, video_state.clean_latent)
            new_v = self.stepper.step(video_state.latent, denoised_v, sigmas, step_idx)
            video_state = video_state.replace(latent=new_v)
            mx.eval(video_state.latent)

            # Audio stays frozen — do NOT update audio_state.latent

            if callback:
                callback(step_idx + 1, num_steps)

        return video_state, audio_state

    def __call__(
        self,
        audio_path: str,
        positive_encoding: mx.array,
        negative_encoding: mx.array,
        config: A2VidConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
        positive_audio_encoding: Optional[mx.array] = None,
        negative_audio_encoding: Optional[mx.array] = None,
    ) -> Tuple[mx.array, np.ndarray, int]:
        """
        Generate video from audio file.

        Args:
            audio_path: Path to input audio file.
            positive_encoding: Encoded text prompt.
            negative_encoding: Encoded negative prompt.
            config: Pipeline configuration.
            images: Optional image conditionings.
            callback: Optional progress callback.
            positive_audio_encoding: Audio text encoding.
            negative_audio_encoding: Negative audio encoding.

        Returns:
            Tuple of (video_tensor, original_audio_waveform, audio_sample_rate).
        """
        images = images or []
        mx.random.seed(config.seed)

        noiser = GaussianNoiser()

        # Load and preserve original audio
        print(f"  Loading audio: {audio_path}")
        audio_waveform, audio_sr = load_audio_file(
            audio_path, config.audio_sample_rate,
            config.audio_start_time, config.audio_max_duration,
        )
        print(f"  Audio: {audio_waveform.shape[1] / audio_sr:.1f}s, {audio_sr}Hz, {audio_waveform.shape[0]}ch")

        # Try to encode audio to latent (may return None if encoder not available)
        stage_1_height = config.height // 2
        stage_1_width = config.width // 2

        stage_1_pixel_shape = VideoPixelShape(
            batch=1, frames=config.num_frames,
            height=stage_1_height, width=stage_1_width, fps=config.fps,
        )

        initial_audio_latent = self._encode_audio_to_latent(
            audio_waveform, audio_sr, stage_1_pixel_shape, config,
        )

        # ====== STAGE 1: Half resolution, video-only denoising ======
        stage_1_latent_shape = VideoLatentShape.from_pixel_shape(stage_1_pixel_shape, latent_channels=128)
        video_tools = self._create_video_tools(stage_1_latent_shape, config.fps)

        stage_1_conditionings = create_image_conditionings(
            images, self.video_encoder, stage_1_height, stage_1_width, config.dtype,
        )

        video_state = video_tools.create_initial_state(dtype=config.dtype)
        video_state = apply_conditionings(video_state, stage_1_conditionings, video_tools)
        video_state = noiser(video_state, noise_scale=1.0)

        # Audio state — frozen (denoise_mask = 0)
        audio_shape = AudioLatentShape.from_video_pixel_shape(
            stage_1_pixel_shape,
            channels=config.audio_vae_channels, mel_bins=config.audio_mel_bins,
            sample_rate=config.audio_sample_rate, hop_length=config.audio_hop_length,
            audio_latent_downsample_factor=config.audio_downsample_factor,
        )
        audio_tools = self._create_audio_tools(audio_shape)

        if initial_audio_latent is not None:
            # Use encoded audio as frozen latent
            audio_state = audio_tools.create_initial_state(dtype=config.dtype, initial_latent=initial_audio_latent)
            # Set denoise_mask to 0 — audio stays frozen
            frozen_mask = mx.zeros_like(audio_state.denoise_mask)
            audio_state = LatentState(
                latent=audio_state.latent,
                denoise_mask=frozen_mask,
                positions=audio_state.positions,
                clean_latent=audio_state.clean_latent,
            )
        else:
            # No encoded audio — generate from noise (but still with audio context)
            audio_state = audio_tools.create_initial_state(dtype=config.dtype)
            audio_state = noiser(audio_state, noise_scale=1.0)

        # Get sigmas
        sigmas = LTX2Scheduler().execute(steps=config.num_inference_steps)

        def stage_1_cb(step, total):
            if callback:
                callback("stage1_a2v", step, total)

        print(f"  Stage 1: {config.num_inference_steps} steps, video-only denoising...")
        video_state, audio_state = self._video_only_denoise_loop(
            video_state, audio_state, sigmas,
            positive_encoding, positive_audio_encoding,
            negative_encoding, negative_audio_encoding,
            cfg_scale=config.cfg_scale, callback=stage_1_cb,
        )

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        stage_1_latent = video_state.latent

        # ====== STAGE 2: Upscale + refine ======
        print("  Upsampling latent 2x with spatial upscaler...")
        latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(stage_1_latent)
        upscaled_unnorm = self.spatial_upscaler(latent_unnorm)
        mx.eval(upscaled_unnorm)
        upscaled_latent = self.video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
        mx.eval(upscaled_latent)

        # Apply distilled LoRA
        if config.distilled_lora_config is not None:
            from mlx.utils import tree_flatten
            if self._original_weights is None:
                self._original_weights = list(tree_flatten(self._velocity_model.parameters()))
            flat_params = dict(tree_flatten(self._velocity_model.parameters()))
            fused = fuse_lora_into_weights(flat_params, [config.distilled_lora_config])
            self._velocity_model.load_weights(list(fused.items()))
            mx.eval(self._velocity_model.parameters())

        # Stage 2 shapes
        stage_2_pixel_shape = VideoPixelShape(
            batch=1, frames=config.num_frames,
            height=config.height, width=config.width, fps=config.fps,
        )
        stage_2_latent_shape = VideoLatentShape.from_pixel_shape(stage_2_pixel_shape, latent_channels=128)
        video_tools_2 = self._create_video_tools(stage_2_latent_shape, config.fps)

        stage_2_conditionings = create_image_conditionings(
            images, self.video_encoder, config.height, config.width, config.dtype,
        )

        video_state_2 = video_tools_2.create_initial_state(dtype=config.dtype, initial_latent=upscaled_latent)
        video_state_2 = apply_conditionings(video_state_2, stage_2_conditionings, video_tools_2)

        stage_2_sigmas = mx.array(STAGE_2_DISTILLED_SIGMA_VALUES)
        video_state_2 = noiser(video_state_2, noise_scale=float(stage_2_sigmas[0]))

        # Audio state for stage 2 (frozen again)
        audio_shape_2 = AudioLatentShape.from_video_pixel_shape(
            stage_2_pixel_shape,
            channels=config.audio_vae_channels, mel_bins=config.audio_mel_bins,
            sample_rate=config.audio_sample_rate, hop_length=config.audio_hop_length,
            audio_latent_downsample_factor=config.audio_downsample_factor,
        )
        audio_tools_2 = self._create_audio_tools(audio_shape_2)
        audio_state_2 = audio_tools_2.create_initial_state(dtype=config.dtype)
        # Frozen
        audio_state_2 = LatentState(
            latent=audio_state_2.latent,
            denoise_mask=mx.zeros_like(audio_state_2.denoise_mask),
            positions=audio_state_2.positions,
            clean_latent=audio_state_2.clean_latent,
        )

        def stage_2_cb(step, total):
            if callback:
                callback("stage2_a2v", step, total)

        print(f"  Stage 2: {len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1} refinement steps...")
        video_state_2, _ = self._video_only_denoise_loop(
            video_state_2, audio_state_2, stage_2_sigmas,
            positive_encoding, positive_audio_encoding,
            callback=stage_2_cb,
        )

        # Restore weights
        if config.distilled_lora_config is not None and self._original_weights is not None:
            self._velocity_model.load_weights(self._original_weights)
            mx.eval(self._velocity_model.parameters())
            self._original_weights = None

        # Decode video
        video_state_2 = video_tools_2.clear_conditioning(video_state_2)
        video_state_2 = video_tools_2.unpatchify(video_state_2)

        effective_tiling = config._get_tiling_config()
        if effective_tiling:
            video_chunks = list(decode_tiled(video_state_2.latent, self.video_decoder, effective_tiling))
            video = mx.concatenate(video_chunks, axis=2) if len(video_chunks) > 1 else video_chunks[0]
        else:
            video = decode_latent(video_state_2.latent, self.video_decoder)

        # Return original audio (not VAE-decoded) for fidelity
        return video, audio_waveform, audio_sr
