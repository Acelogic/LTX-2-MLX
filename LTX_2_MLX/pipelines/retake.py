"""Retake pipeline — regenerate a time region of an existing video.

Given a source video file and a time window [start_time, end_time] in seconds,
this pipeline keeps content outside that window unchanged and regenerates the
content inside using a text prompt. Supports both distilled (8-step) and guided
(N-step with CFG) modes.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .common import (
    apply_conditionings,
    modality_from_state,
    audio_modality_from_state,
    post_process_latent,
)
from ..components import (
    DISTILLED_SIGMA_VALUES,
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
from ..model.audio_vae import AudioDecoder, Vocoder
from ..types import (
    AudioLatentShape,
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
)


@dataclass
class RetakeConfig:
    """Configuration for retake pipeline."""

    start_time: float  # seconds, inclusive
    end_time: float  # seconds, exclusive
    regenerate_video: bool = True
    regenerate_audio: bool = True
    distilled: bool = False
    num_inference_steps: int = 40
    cfg_scale: float = 3.0
    seed: int = 42
    tiling_config: Optional[TilingConfig] = None
    dtype: mx.Dtype = mx.float16

    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError(f"start_time ({self.start_time}) must be < end_time ({self.end_time})")


def get_video_metadata(video_path: str) -> Tuple[float, int, int, int]:
    """Get video metadata using ffprobe.

    Returns:
        Tuple of (fps, num_frames, width, height).
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)

    for stream in data.get("streams", []):
        if stream["codec_type"] == "video":
            width = int(stream["width"])
            height = int(stream["height"])
            # Parse fps from r_frame_rate
            fps_parts = stream.get("r_frame_rate", "24/1").split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            num_frames = int(stream.get("nb_frames", 0))
            if num_frames == 0:
                duration = float(data.get("format", {}).get("duration", 0))
                num_frames = int(duration * fps)
            return fps, num_frames, width, height

    raise ValueError(f"No video stream found in {video_path}")


def load_video_frames(
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Load video frames using ffmpeg and convert to tensor.

    Returns:
        Video tensor of shape (1, 3, F, H, W) normalized to [-1, 1].
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames with ffmpeg
        cmd = [
            "ffmpeg", "-v", "quiet", "-i", video_path,
            "-vf", f"scale={width}:{height}",
            "-frames:v", str(num_frames),
            "-start_number", "0",
            f"{tmpdir}/frame_%06d.png",
        ]
        subprocess.run(cmd, check=True)

        frames = []
        for i in range(num_frames):
            frame_path = f"{tmpdir}/frame_{i:06d}.png"
            try:
                img = Image.open(frame_path).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
                frames.append(img_np)
            except FileNotFoundError:
                break

    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    # Stack: (F, H, W, C) -> (1, C, F, H, W)
    video_np = np.stack(frames, axis=0)  # (F, H, W, C)
    video_np = np.transpose(video_np, (3, 0, 1, 2))  # (C, F, H, W)
    video_np = video_np[np.newaxis, ...]  # (1, C, F, H, W)

    return mx.array(video_np).astype(dtype)


class TemporalRegionMask:
    """Conditioning that sets denoise_mask=1 inside a time range, 0 outside.

    Only the masked region (denoise_mask=1) will be regenerated during denoising.
    Content outside the mask is preserved from the clean_latent.

    Args:
        start_time: Start of region in seconds (inclusive).
        end_time: End of region in seconds (exclusive).
        fps: Video frame rate for time-to-frame conversion.
    """

    def __init__(self, start_time: float, end_time: float, fps: float):
        self.start_time = start_time
        self.end_time = end_time
        self.fps = fps

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """Apply temporal mask to latent state's denoise_mask."""
        target_shape = latent_tools.target_shape

        # Convert time to latent frame indices
        # Video: pixel_frames -> latent_frames (8x temporal downscale)
        start_pixel_frame = int(self.start_time * self.fps)
        end_pixel_frame = int(self.end_time * self.fps)

        # Latent frames = (pixel_frames - 1) / 8 + 1 for 8x temporal downscale
        start_latent_frame = max(0, (start_pixel_frame - 1) // 8)
        end_latent_frame = min(target_shape.frames, (end_pixel_frame - 1) // 8 + 1)

        # Total latent tokens per frame = height * width
        tokens_per_frame = target_shape.height * target_shape.width
        total_tokens = target_shape.frames * tokens_per_frame

        # Build mask: 0 outside region, 1 inside
        mask = mx.zeros((1, total_tokens, 1), dtype=latent_state.denoise_mask.dtype)

        if start_latent_frame < end_latent_frame:
            start_token = start_latent_frame * tokens_per_frame
            end_token = end_latent_frame * tokens_per_frame
            # Set region to 1
            ones = mx.ones((1, end_token - start_token, 1), dtype=mask.dtype)
            mask_before = mask[:, :start_token, :]
            mask_after = mask[:, end_token:, :]
            mask = mx.concatenate([mask_before, ones, mask_after], axis=1)

        return LatentState(
            latent=latent_state.latent,
            denoise_mask=mask,
            positions=latent_state.positions,
            clean_latent=latent_state.clean_latent,
        )


class RetakePipeline:
    """
    Regenerate a time region of an existing video.

    Given a source video and time window, encodes the full video to latents,
    applies a temporal mask, and only denoises the masked region while
    preserving the rest.
    """

    def __init__(
        self,
        transformer: Union[LTXModel, LTXAVModel],
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        audio_decoder: Optional[AudioDecoder] = None,
        vocoder: Optional[Vocoder] = None,
    ):
        if isinstance(transformer, X0Model):
            self.transformer = transformer
        else:
            self.transformer = X0Model(transformer)

        inner = self.transformer.velocity_model if hasattr(self.transformer, 'velocity_model') else transformer
        self.is_av_model = getattr(inner, 'model_type', None) == LTXModelType.AudioVideo

        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.stepper = EulerDiffusionStep()

    def _create_video_tools(self, target_shape, fps):
        return VideoLatentTools(patchifier=self.patchifier, target_shape=target_shape, fps=fps)

    def _denoise_loop(
        self,
        video_state, audio_state, sigmas,
        video_context, audio_context,
        negative_video_context=None, negative_audio_context=None,
        cfg_scale=1.0, callback=None,
    ):
        """Euler denoising loop with optional CFG."""
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Positive pass
            video_mod = modality_from_state(video_state, video_context, sigma)
            if self.is_av_model and audio_state is not None:
                audio_mod = audio_modality_from_state(audio_state, audio_context, sigma)
                result = self.transformer(video_mod, audio_mod)
                if isinstance(result, tuple):
                    denoised_v, denoised_a = result
                else:
                    denoised_v, denoised_a = result, None
            else:
                denoised_v = self.transformer(video_mod)
                denoised_a = None

            # CFG
            if cfg_scale > 1.0 and negative_video_context is not None:
                video_mod_neg = modality_from_state(video_state, negative_video_context, sigma)
                if self.is_av_model and audio_state is not None and negative_audio_context is not None:
                    audio_mod_neg = audio_modality_from_state(audio_state, negative_audio_context, sigma)
                    result_neg = self.transformer(video_mod_neg, audio_mod_neg)
                    uncond_v = result_neg[0] if isinstance(result_neg, tuple) else result_neg
                    uncond_a = result_neg[1] if isinstance(result_neg, tuple) else None
                else:
                    uncond_v = self.transformer(video_mod_neg)
                    uncond_a = None

                denoised_v = uncond_v + cfg_scale * (denoised_v - uncond_v)
                if denoised_a is not None and uncond_a is not None:
                    denoised_a = uncond_a + cfg_scale * (denoised_a - uncond_a)

            # Post-process with mask (preserves clean_latent outside mask)
            denoised_v = post_process_latent(denoised_v, video_state.denoise_mask, video_state.clean_latent)
            new_v = self.stepper.step(video_state.latent, denoised_v, sigmas, step_idx)
            video_state = video_state.replace(latent=new_v)
            mx.eval(video_state.latent)

            if audio_state is not None and denoised_a is not None:
                denoised_a = post_process_latent(denoised_a, audio_state.denoise_mask, audio_state.clean_latent)
                new_a = self.stepper.step(audio_state.latent, denoised_a, sigmas, step_idx)
                audio_state = audio_state.replace(latent=new_a)
                mx.eval(audio_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        return video_state, audio_state

    def __call__(
        self,
        video_path: str,
        text_encoding: mx.array,
        text_mask: mx.array,
        config: RetakeConfig,
        negative_text_encoding: Optional[mx.array] = None,
        audio_encoding: Optional[mx.array] = None,
        negative_audio_encoding: Optional[mx.array] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> mx.array:
        """
        Regenerate a time region of the source video.

        Args:
            video_path: Path to the source video file.
            text_encoding: Encoded text prompt.
            text_mask: Text attention mask.
            config: Retake configuration with time window.
            negative_text_encoding: Optional negative prompt for CFG.
            audio_encoding: Optional audio text encoding.
            negative_audio_encoding: Optional negative audio encoding.
            callback: Optional progress callback(stage, step, total).

        Returns:
            Video tensor (decoded).
        """
        mx.random.seed(config.seed)
        noiser = GaussianNoiser()

        # Get source video metadata
        fps, num_pixel_frames, src_width, src_height = get_video_metadata(video_path)

        # Snap frame count to 8k+1
        num_pixel_frames = ((num_pixel_frames - 1) // 8) * 8 + 1

        print(f"  Source: {src_width}x{src_height}, {num_pixel_frames} frames @ {fps:.1f} FPS")
        print(f"  Retake region: {config.start_time:.1f}s - {config.end_time:.1f}s")

        output_shape = VideoPixelShape(
            batch=1, frames=num_pixel_frames,
            height=src_height, width=src_width, fps=fps,
        )

        # Encode source video to latents
        print("  Encoding source video...")
        video_tensor = load_video_frames(video_path, src_height, src_width, num_pixel_frames, config.dtype)
        initial_video_latent = self.video_encoder(video_tensor)
        mx.eval(initial_video_latent)
        del video_tensor

        # Create latent shape and tools
        latent_shape = VideoLatentShape.from_pixel_shape(output_shape, latent_channels=128)
        video_tools = self._create_video_tools(latent_shape, fps)

        # Create video state with encoded latent as initial + clean reference
        video_state = video_tools.create_initial_state(
            dtype=config.dtype, initial_latent=initial_video_latent
        )

        # Apply temporal mask
        if config.regenerate_video:
            mask = TemporalRegionMask(config.start_time, config.end_time, fps)
            video_state = mask.apply_to(video_state, video_tools)

        # Add noise (only affects masked region due to denoise_mask)
        video_state = noiser(video_state, noise_scale=1.0)

        # Get sigmas
        if config.distilled:
            sigmas = mx.array(DISTILLED_SIGMA_VALUES)
        else:
            sigmas = LTX2Scheduler().execute(steps=config.num_inference_steps)

        # Audio state (placeholder — audio retake needs audio encoder)
        audio_state = None

        def progress_cb(step, total):
            if callback:
                callback("retake", step, total)

        # Run denoising
        mode = "distilled" if config.distilled else f"guided ({config.num_inference_steps} steps)"
        print(f"  Denoising ({mode})...")

        if config.distilled:
            # Simple denoising, no CFG
            video_state, audio_state = self._denoise_loop(
                video_state, audio_state, sigmas,
                text_encoding, audio_encoding,
                callback=progress_cb,
            )
        else:
            # Guided with CFG
            if negative_text_encoding is None:
                negative_text_encoding = mx.zeros_like(text_encoding)
            video_state, audio_state = self._denoise_loop(
                video_state, audio_state, sigmas,
                text_encoding, audio_encoding,
                negative_text_encoding, negative_audio_encoding,
                cfg_scale=config.cfg_scale,
                callback=progress_cb,
            )

        # Unpatchify and decode
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        effective_tiling = config.tiling_config
        if effective_tiling:
            video_chunks = list(decode_tiled(video_state.latent, self.video_decoder, effective_tiling))
            video = mx.concatenate(video_chunks, axis=2) if len(video_chunks) > 1 else video_chunks[0]
        else:
            video = decode_latent(video_state.latent, self.video_decoder)

        return video
