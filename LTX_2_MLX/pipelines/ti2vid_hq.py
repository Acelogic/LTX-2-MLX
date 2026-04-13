"""High-quality two-stage text/image-to-video pipeline using Res2s sampler.

Uses the second-order Res2s (Runge-Kutta) sampler for better quality in fewer
steps compared to Euler. Stage 1 at half resolution with CFG, Stage 2 at full
resolution with spatial upscaler and distilled LoRA.

Default: 15 inference steps (vs 30 for Euler two-stage).
"""

import math
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
    Res2sDiffusionStep,
    VideoLatentPatchifier,
    get_res2s_coefficients,
)
from ..components.patchifiers import AudioPatchifier
from ..conditioning.tools import VideoLatentTools, AudioLatentTools
from ..model.transformer import LTXModel, LTXAVModel, LTXModelType, X0Model, Modality
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
class TI2VidHQConfig:
    """Configuration for HQ two-stage pipeline."""

    height: int = 1088
    width: int = 1920
    num_frames: int = 97

    num_inference_steps: int = 15
    cfg_scale: float = 3.0
    audio_cfg_scale: float = 7.0
    guidance_rescale: float = 0.45
    seed: int = 42
    fps: float = 25.0

    # LoRA for stage 2 refinement
    distilled_lora_config: Optional[LoRAConfig] = None

    tiling_config: Optional[TilingConfig] = None
    dtype: mx.Dtype = mx.float16

    audio_enabled: bool = False
    use_internal_audio_branch: bool = True
    audio_vae_channels: int = 8
    audio_mel_bins: int = 16
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_downsample_factor: int = 4
    audio_output_sample_rate: int = 24000

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
            raise ValueError(
                f"Resolution ({self.height}x{self.width}) must be divisible by 64."
            )


class TI2VidHQPipeline:
    """
    High-quality two-stage pipeline using Res2s second-order sampler.

    Stage 1: Generate at half resolution with CFG + Res2s (15 steps default)
    Stage 2: Upsample 2x and refine with distilled sigmas + optional LoRA
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

        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.spatial_upscaler = spatial_upscaler
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.euler_step = EulerDiffusionStep()
        self.res2s_step = Res2sDiffusionStep()

        self._original_weights = None

    def _create_video_tools(self, target_shape, fps):
        return VideoLatentTools(patchifier=self.patchifier, target_shape=target_shape, fps=fps)

    def _create_audio_tools(self, target_shape):
        return AudioLatentTools(patchifier=AudioPatchifier(patch_size=1), target_shape=target_shape)

    def _decode_audio(self, audio_latent):
        if self.audio_decoder is None or self.vocoder is None:
            return None
        mel = self.audio_decoder(audio_latent)
        mx.eval(mel)
        waveform = self.vocoder(mel)
        mx.eval(waveform)
        return waveform

    def _res2s_denoise_loop(
        self,
        video_state: LatentState,
        audio_state: Optional[LatentState],
        sigmas: mx.array,
        video_context: mx.array,
        audio_context: Optional[mx.array],
        negative_video_context: Optional[mx.array],
        negative_audio_context: Optional[mx.array],
        cfg_scale: float,
        audio_cfg_scale: float,
        callback: Optional[Callable] = None,
    ) -> Tuple[LatentState, Optional[LatentState]]:
        """Run Res2s second-order denoising loop with CFG."""
        num_steps = len(sigmas) - 1

        # Inject minimal sigma to avoid div-by-zero
        sigmas_list = [float(s) for s in sigmas]
        if sigmas_list[-1] == 0.0:
            sigmas_list = sigmas_list[:-1] + [0.0011, 0.0]

        # Compute step sizes in log-space
        hs = []
        for i in range(len(sigmas_list) - 1):
            if sigmas_list[i] > 0 and sigmas_list[i + 1] > 0:
                hs.append(-math.log(sigmas_list[i + 1] / sigmas_list[i]))
            else:
                hs.append(0.0)

        phi_cache = {}
        c2 = 0.5

        for step_idx in range(num_steps):
            sigma = sigmas_list[step_idx]
            sigma_next = sigmas_list[step_idx + 1]

            # ── Stage 1: Evaluate at current point with CFG ──
            denoised_v, denoised_a = self._denoise_with_cfg(
                video_state, audio_state, sigma,
                video_context, audio_context,
                negative_video_context, negative_audio_context,
                cfg_scale,
                audio_cfg_scale,
            )
            denoised_v = post_process_latent(denoised_v, video_state.denoise_mask, video_state.clean_latent)
            if audio_state is not None and denoised_a is not None:
                denoised_a = post_process_latent(denoised_a, audio_state.denoise_mask, audio_state.clean_latent)

            h = hs[step_idx]

            if h == 0.0 or sigma_next <= 0.001:
                # Final step — just use denoised directly
                video_state = video_state.replace(latent=denoised_v)
                if audio_state is not None and denoised_a is not None:
                    audio_state = audio_state.replace(latent=denoised_a)
                break

            # Get RK coefficients
            a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

            # Compute substep sigma (geometric mean for c2=0.5)
            sub_sigma = math.sqrt(sigma * sigma_next)

            # ── Compute midpoint ──
            anchor_v = video_state.latent.astype(mx.float32)
            eps_1_v = denoised_v.astype(mx.float32) - anchor_v
            x_mid_v = anchor_v + h * a21 * eps_1_v
            mx.eval(x_mid_v)

            anchor_a = None
            eps_1_a = None
            x_mid_a = None
            if audio_state is not None and denoised_a is not None:
                anchor_a = audio_state.latent.astype(mx.float32)
                eps_1_a = denoised_a.astype(mx.float32) - anchor_a
                x_mid_a = anchor_a + h * a21 * eps_1_a
                mx.eval(x_mid_a)

            # ── Bong iteration for stability ──
            if h < 0.5 and sigma > 0.03:
                for _ in range(100):
                    anchor_v = x_mid_v - h * a21 * eps_1_v
                    eps_1_v = denoised_v.astype(mx.float32) - anchor_v
                    if anchor_a is not None:
                        anchor_a = x_mid_a - h * a21 * eps_1_a
                        eps_1_a = denoised_a.astype(mx.float32) - anchor_a

            # ── Stage 2: Evaluate at midpoint ──
            mid_video_state = video_state.replace(latent=x_mid_v.astype(video_state.latent.dtype))
            mid_audio_state = None
            if audio_state is not None and x_mid_a is not None:
                mid_audio_state = audio_state.replace(latent=x_mid_a.astype(audio_state.latent.dtype))

            denoised_v2, denoised_a2 = self._denoise_with_cfg(
                mid_video_state, mid_audio_state, sub_sigma,
                video_context, audio_context,
                negative_video_context, negative_audio_context,
                cfg_scale,
                audio_cfg_scale,
            )
            denoised_v2 = post_process_latent(denoised_v2, video_state.denoise_mask, video_state.clean_latent)
            if audio_state is not None and denoised_a2 is not None:
                denoised_a2 = post_process_latent(denoised_a2, audio_state.denoise_mask, audio_state.clean_latent)

            # ── Final combination with RK coefficients ──
            eps_2_v = denoised_v2.astype(mx.float32) - anchor_v
            x_next_v = anchor_v + h * (b1 * eps_1_v + b2 * eps_2_v)

            video_state = video_state.replace(latent=x_next_v.astype(video_state.latent.dtype))
            mx.eval(video_state.latent)

            if audio_state is not None and denoised_a2 is not None and anchor_a is not None:
                eps_2_a = denoised_a2.astype(mx.float32) - anchor_a
                x_next_a = anchor_a + h * (b1 * eps_1_a + b2 * eps_2_a)
                audio_state = audio_state.replace(latent=x_next_a.astype(audio_state.latent.dtype))
                mx.eval(audio_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        return video_state, audio_state

    def _denoise_with_cfg(
        self,
        video_state, audio_state, sigma,
        pos_v_ctx, pos_a_ctx, neg_v_ctx, neg_a_ctx,
        cfg_scale,
        audio_cfg_scale,
    ):
        """Run transformer with CFG guidance."""
        video_mod = modality_from_state(video_state, pos_v_ctx, sigma)

        if self.is_av_model and audio_state is not None:
            # AV model always needs audio context — fall back to video context if None
            effective_a_ctx = pos_a_ctx if pos_a_ctx is not None else pos_v_ctx
            audio_mod = audio_modality_from_state(audio_state, effective_a_ctx, sigma)
            result = self.transformer(video_mod, audio_mod)
            if isinstance(result, tuple):
                cond_v, cond_a = result
            else:
                cond_v, cond_a = result, None
        else:
            cond_v = self.transformer(video_mod)
            cond_a = None

        # Unconditional pass for CFG
        if (cfg_scale > 1.0 or audio_cfg_scale > 1.0) and neg_v_ctx is not None:
            video_mod_neg = modality_from_state(video_state, neg_v_ctx, sigma)

            if self.is_av_model and audio_state is not None:
                effective_neg_a_ctx = neg_a_ctx if neg_a_ctx is not None else neg_v_ctx
                audio_mod_neg = audio_modality_from_state(audio_state, effective_neg_a_ctx, sigma)
                result_neg = self.transformer(video_mod_neg, audio_mod_neg)
                if isinstance(result_neg, tuple):
                    uncond_v, uncond_a = result_neg
                else:
                    uncond_v, uncond_a = result_neg, None
            else:
                uncond_v = self.transformer(video_mod_neg)
                uncond_a = None

            # Apply CFG: cond + scale * (cond - uncond)
            cond_v = uncond_v + cfg_scale * (cond_v - uncond_v)
            if cond_a is not None and uncond_a is not None:
                cond_a = uncond_a + audio_cfg_scale * (cond_a - uncond_a)

        return cond_v, cond_a

    def _simple_denoise_loop(
        self,
        video_state, audio_state, sigmas, video_context, audio_context,
        stepper, callback=None,
    ):
        """Simple denoising loop (no CFG) for stage 2 refinement."""
        num_steps = len(sigmas) - 1
        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            video_mod = modality_from_state(video_state, video_context, sigma)

            if self.is_av_model and audio_state is not None:
                effective_a_ctx = audio_context if audio_context is not None else video_context
                audio_mod = audio_modality_from_state(audio_state, effective_a_ctx, sigma)
                result = self.transformer(video_mod, audio_mod)
                if isinstance(result, tuple):
                    denoised_v, denoised_a = result
                else:
                    denoised_v, denoised_a = result, None
            else:
                denoised_v = self.transformer(video_mod)
                denoised_a = None

            denoised_v = post_process_latent(denoised_v, video_state.denoise_mask, video_state.clean_latent)
            new_v = stepper.step(video_state.latent, denoised_v, sigmas, step_idx)
            video_state = video_state.replace(latent=new_v)
            mx.eval(video_state.latent)

            if audio_state is not None and denoised_a is not None:
                denoised_a = post_process_latent(denoised_a, audio_state.denoise_mask, audio_state.clean_latent)
                new_a = stepper.step(audio_state.latent, denoised_a, sigmas, step_idx)
                audio_state = audio_state.replace(latent=new_a)
                mx.eval(audio_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        return video_state, audio_state

    def __call__(
        self,
        positive_encoding: mx.array,
        negative_encoding: mx.array,
        config: TI2VidHQConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
        positive_audio_encoding: Optional[mx.array] = None,
        negative_audio_encoding: Optional[mx.array] = None,
    ) -> Union[mx.array, Tuple[mx.array, Optional[mx.array]]]:
        """Generate video using HQ two-stage Res2s pipeline."""
        images = images or []
        mx.random.seed(config.seed)

        noiser = GaussianNoiser()

        # ====== STAGE 1: Half resolution with Res2s + CFG ======
        stage_1_height = config.height // 2
        stage_1_width = config.width // 2

        stage_1_pixel_shape = VideoPixelShape(
            batch=1, frames=config.num_frames,
            height=stage_1_height, width=stage_1_width, fps=config.fps,
        )
        stage_1_latent_shape = VideoLatentShape.from_pixel_shape(stage_1_pixel_shape, latent_channels=128)
        video_tools = self._create_video_tools(stage_1_latent_shape, config.fps)

        stage_1_conditionings = create_image_conditionings(
            images, self.video_encoder, stage_1_height, stage_1_width, config.dtype,
        )

        video_state = video_tools.create_initial_state(dtype=config.dtype)
        video_state = apply_conditionings(video_state, stage_1_conditionings, video_tools)

        # Get sigma schedule from LTX2Scheduler
        sigmas = LTX2Scheduler().execute(steps=config.num_inference_steps)

        video_state = noiser(video_state, noise_scale=1.0)

        internal_audio_active = self.is_av_model and (config.use_internal_audio_branch or config.audio_enabled)

        # Audio state
        audio_state = None
        if internal_audio_active:
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                stage_1_pixel_shape,
                channels=config.audio_vae_channels, mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate, hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools = self._create_audio_tools(audio_shape)
            audio_state = audio_tools.create_initial_state(dtype=config.dtype)
            audio_state = noiser(audio_state, noise_scale=1.0)

        def stage_1_cb(step, total):
            if callback:
                callback("stage1_res2s", step, total)

        print(f"  Stage 1: {config.num_inference_steps} Res2s steps at {stage_1_width}x{stage_1_height}...")
        video_state, audio_state = self._res2s_denoise_loop(
            video_state=video_state,
            audio_state=audio_state,
            sigmas=sigmas,
            video_context=positive_encoding,
            audio_context=positive_audio_encoding,
            negative_video_context=negative_encoding,
            negative_audio_context=negative_audio_encoding,
            cfg_scale=config.cfg_scale,
            audio_cfg_scale=config.audio_cfg_scale,
            callback=stage_1_cb,
        )

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        stage_1_latent = video_state.latent

        stage_1_audio_latent = None
        if audio_state is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            stage_1_audio_latent = audio_state.latent

        # ====== STAGE 2: Upscale + refine with distilled sigmas ======
        print("  Upsampling latent 2x with spatial upscaler...")
        latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(stage_1_latent)
        upscaled_unnorm = self.spatial_upscaler(latent_unnorm)
        mx.eval(upscaled_unnorm)
        upscaled_latent = self.video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
        mx.eval(upscaled_latent)

        # Apply distilled LoRA for stage 2
        if config.distilled_lora_config is not None:
            from mlx.utils import tree_flatten
            if self._original_weights is None:
                self._original_weights = list(tree_flatten(self._velocity_model.parameters()))
            flat_params = dict(tree_flatten(self._velocity_model.parameters()))
            fused = fuse_lora_into_weights(flat_params, [config.distilled_lora_config])
            self._velocity_model.load_weights(list(fused.items()))
            mx.eval(self._velocity_model.parameters())

        # Create stage 2 shapes
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

        # Audio for stage 2
        audio_state_2 = None
        if internal_audio_active:
            audio_shape_2 = AudioLatentShape.from_video_pixel_shape(
                stage_2_pixel_shape,
                channels=config.audio_vae_channels, mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate, hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools_2 = self._create_audio_tools(audio_shape_2)
            if stage_1_audio_latent is not None:
                audio_state_2 = audio_tools_2.create_initial_state(dtype=config.dtype, initial_latent=stage_1_audio_latent)
            else:
                audio_state_2 = audio_tools_2.create_initial_state(dtype=config.dtype)
            audio_state_2 = noiser(audio_state_2, noise_scale=float(stage_2_sigmas[0]))

        def stage_2_cb(step, total):
            if callback:
                callback("stage2", step, total)

        print(f"  Stage 2: {len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1} refinement steps at {config.width}x{config.height}...")
        video_state_2, audio_state_2 = self._simple_denoise_loop(
            video_state_2, audio_state_2, stage_2_sigmas,
            positive_encoding, positive_audio_encoding,
            self.euler_step, stage_2_cb,
        )

        # Restore original weights
        if config.distilled_lora_config is not None and self._original_weights is not None:
            self._velocity_model.load_weights(self._original_weights)
            mx.eval(self._velocity_model.parameters())
            self._original_weights = None

        # Unpatchify and decode
        video_state_2 = video_tools_2.clear_conditioning(video_state_2)
        video_state_2 = video_tools_2.unpatchify(video_state_2)
        final_latent = video_state_2.latent

        effective_tiling = config._get_tiling_config()
        if effective_tiling:
            video_chunks = list(decode_tiled(final_latent, self.video_decoder, effective_tiling))
            video = mx.concatenate(video_chunks, axis=2) if len(video_chunks) > 1 else video_chunks[0]
        else:
            video = decode_latent(final_latent, self.video_decoder)

        audio_waveform = None
        if audio_state_2 is not None:
            audio_state_2 = audio_tools_2.clear_conditioning(audio_state_2)
            audio_state_2 = audio_tools_2.unpatchify(audio_state_2)
            audio_waveform = self._decode_audio(audio_state_2.latent)

        if config.audio_enabled:
            return video, audio_waveform
        return video
