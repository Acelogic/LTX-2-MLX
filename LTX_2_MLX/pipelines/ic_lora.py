"""IC-LoRA (In-Context LoRA) pipeline for LTX-2 MLX.

This pipeline enables video-to-video generation with control signals
such as depth maps, human pose, or edge maps via IC-LoRA conditioning.
Uses a two-stage approach:
  Stage 1: Generate at half resolution with IC-LoRA
  Stage 2: Upsample 2x and refine WITHOUT IC-LoRA (clean model)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import mlx.core as mx
import numpy as np

from .common import (
    ImageCondition,
    apply_conditionings,
    create_image_conditionings,
    modality_from_state,
    post_process_latent,
    timesteps_from_mask,
)
from ..components import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    EulerDiffusionStep,
    GaussianNoiser,
    VideoLatentPatchifier,
)
from ..conditioning.item import ConditioningItem
from ..conditioning.keyframe import VideoConditionByKeyframeIndex
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


class ControlType(Enum):
    """Type of control signal for IC-LoRA conditioning."""
    CANNY = "canny"
    RAW = "raw"  # No preprocessing, use video as-is


def preprocess_canny(
    video_path: Union[str, Path],
    height: int,
    width: int,
    num_frames: int,
    low_threshold: int = 100,
    high_threshold: int = 200,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Apply Canny edge detection to a video, creating a control signal.

    Args:
        video_path: Path to input video.
        height: Target height.
        width: Target width.
        num_frames: Number of frames to process.
        low_threshold: Canny low threshold (0-255).
        high_threshold: Canny high threshold (0-255).
        output_path: Optional path to save the processed video.

    Returns:
        Edge video as numpy array (F, H, W, 3) in [0, 255].
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV required for Canny preprocessing. "
            "Install with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize first
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

        # Convert to grayscale and apply Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert to 3-channel (white edges on black background)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        frames.append(edges_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    # Pad with last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])

    video_np = np.stack(frames, axis=0)  # (F, H, W, 3)

    # Optionally save the processed video
    if output_path:
        _save_control_video(video_np, str(output_path))

    return video_np


def _save_control_video(
    video_np: np.ndarray,
    output_path: str,
    fps: float = 24.0,
) -> None:
    """Save a control video to disk for debugging/visualization."""
    try:
        import cv2
    except ImportError:
        return  # Skip saving if OpenCV not available

    h, w = video_np.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in video_np:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def preprocess_control_signal(
    video_path: Union[str, Path],
    control_type: ControlType,
    height: int,
    width: int,
    num_frames: int,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Preprocess a video to create a control signal for IC-LoRA.

    Args:
        video_path: Path to input video.
        control_type: Type of control signal (canny, raw).
        height: Target height.
        width: Target width.
        num_frames: Number of frames.
        output_path: Optional path to save preprocessed video.
        **kwargs: Control-type-specific parameters:
            - canny: low_threshold (int), high_threshold (int)

    Returns:
        Control signal video (F, H, W, 3) in [0, 255].
    """
    if control_type == ControlType.CANNY:
        return preprocess_canny(
            video_path=video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            low_threshold=kwargs.get("low_threshold", 100),
            high_threshold=kwargs.get("high_threshold", 200),
            output_path=output_path,
        )
    elif control_type == ControlType.RAW:
        # Load video without preprocessing
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV required for video loading. "
                "Install with: pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from {video_path}")

        while len(frames) < num_frames:
            frames.append(frames[-1])

        return np.stack(frames, axis=0)
    else:
        raise ValueError(f"Unknown control type: {control_type}")


def load_control_signal_tensor(
    control_signal: np.ndarray,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Convert a control signal numpy array to MLX tensor for VAE encoding.

    Args:
        control_signal: Control video (F, H, W, 3) in [0, 255].
        dtype: Output dtype.

    Returns:
        Tensor (1, 3, F, H, W) normalized to [-1, 1].
    """
    # Normalize to [-1, 1]
    video_np = control_signal.astype(np.float32) / 127.5 - 1.0

    # Convert to MLX: (F, H, W, C) -> (1, C, F, H, W)
    video_mx = mx.array(video_np)
    video_mx = mx.transpose(video_mx, (3, 0, 1, 2))  # (C, F, H, W)
    video_mx = video_mx[None, :, :, :, :]  # (1, C, F, H, W)

    return video_mx.astype(dtype)


@dataclass
class ICLoraConfig:
    """Configuration for IC-LoRA pipeline."""

    # Video dimensions (output)
    height: int = 480
    width: int = 704
    num_frames: int = 97  # Must be 8k + 1

    # Generation parameters
    stage_1_steps: int = 7  # Distilled model steps
    stage_2_steps: int = 3  # Refinement steps
    seed: int = 42
    fps: float = 24.0

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
        if self.height % 64 != 0 or self.width % 64 != 0:
            raise ValueError(
                f"Resolution ({self.height}x{self.width}) "
                f"must be divisible by 64."
            )


@dataclass
class VideoCondition:
    """A video control signal (depth, pose, canny, etc.) for IC-LoRA."""

    video_path: str
    strength: float = 0.95
    control_type: ControlType = ControlType.RAW
    # Canny-specific parameters
    canny_low: int = 100
    canny_high: int = 200
    # Whether to save the preprocessed control signal
    save_control: bool = False


def load_video_tensor(
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Load a video and prepare for VAE encoding.

    Args:
        video_path: Path to video file.
        height: Target height.
        width: Target width.
        num_frames: Number of frames to load.
        dtype: Data type for output tensor.

    Returns:
        Video tensor of shape (1, 3, F, H, W).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) required for video loading. Install with: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    # Pad with last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Stack frames: (F, H, W, C) -> (1, C, F, H, W)
    video_np = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    video_mx = mx.array(video_np)
    video_mx = mx.transpose(video_mx, (3, 0, 1, 2))  # (C, F, H, W)
    video_mx = video_mx[None, :, :, :, :]  # (1, C, F, H, W)

    return video_mx.astype(dtype)


def create_video_conditionings(
    videos: List[VideoCondition],
    video_encoder: SimpleVideoEncoder,
    height: int,
    width: int,
    num_frames: int,
    dtype: mx.Dtype = mx.float32,
) -> List[ConditioningItem]:
    """
    Create conditionings for control videos (depth, pose, canny, etc.).

    This function handles preprocessing based on the control type:
    - CANNY: Apply Canny edge detection to create edge maps
    - RAW: Use the video as-is (e.g., pre-processed depth maps)

    Args:
        videos: List of VideoCondition objects with paths and settings.
        video_encoder: VAE encoder for encoding control signals.
        height: Target height (will be halved for stage 1).
        width: Target width (will be halved for stage 1).
        num_frames: Number of frames to process.
        dtype: Data type for tensors.

    Returns:
        List of ConditioningItem objects for the denoising loop.
    """
    conditionings = []

    for vid_cond in videos:
        # Preprocess based on control type
        if vid_cond.control_type == ControlType.CANNY:
            # Determine output path for saving preprocessed video
            output_path = None
            if vid_cond.save_control:
                base_path = vid_cond.video_path.rsplit(".", 1)[0]
                output_path = f"{base_path}_canny.mp4"

            control_signal = preprocess_control_signal(
                video_path=vid_cond.video_path,
                control_type=vid_cond.control_type,
                height=height,
                width=width,
                num_frames=num_frames,
                output_path=output_path,
                low_threshold=vid_cond.canny_low,
                high_threshold=vid_cond.canny_high,
            )
            video_tensor = load_control_signal_tensor(control_signal, dtype)
        else:
            # RAW: load video directly
            video_tensor = load_video_tensor(
                vid_cond.video_path, height, width, num_frames, dtype
            )

        # Encode through VAE
        encoded_video = video_encoder(video_tensor)
        mx.eval(encoded_video)

        # Use keyframe conditioning to append the control signal
        conditioning = VideoConditionByKeyframeIndex(
            keyframes=encoded_video,
            frame_idx=0,  # Start from frame 0
            strength=vid_cond.strength,
        )
        conditionings.append(conditioning)

    return conditionings


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support.

    This pipeline enables video-to-video generation with control signals
    such as depth maps, human pose, or edge maps. The control signal is
    encoded via the VAE and passed through IC-LoRA conditioning.

    Stage 1: Generate at half resolution with IC-LoRA applied
    Stage 2: Upsample 2x and refine WITHOUT IC-LoRA (clean model)

    The pipeline expects:
    - A transformer model with base weights
    - IC-LoRA weights to be fused for stage 1
    - The original (unfused) weights to be restored for stage 2
    """

    def __init__(
        self,
        transformer: LTXModel,
        video_encoder: SimpleVideoEncoder,
        video_decoder: SimpleVideoDecoder,
        spatial_upscaler: SpatialUpscaler,
        base_transformer_weights: Dict[str, mx.array],
        lora_configs: Optional[List[LoRAConfig]] = None,
    ):
        """
        Initialize the IC-LoRA pipeline.

        Args:
            transformer: LTX transformer model (base weights).
            video_encoder: VAE encoder for encoding images/videos.
            video_decoder: VAE decoder for decoding latents to video.
            spatial_upscaler: 2x spatial upscaler for stage 2.
            base_transformer_weights: Original transformer weights for restoration.
            lora_configs: IC-LoRA configurations (paths and strengths).
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
        self.base_transformer_weights = base_transformer_weights
        self.lora_configs = lora_configs or []
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

    def _apply_lora(self) -> None:
        """Fuse IC-LoRA weights into the transformer."""
        if not self.lora_configs:
            return

        fused_weights = fuse_lora_into_weights(
            self.base_transformer_weights,
            self.lora_configs,
            verbose=True,
        )

        # Apply fused weights to transformer (use raw velocity model)
        self._velocity_model.load_weights(list(fused_weights.items()))
        mx.eval(self._velocity_model.parameters())

    def _remove_lora(self) -> None:
        """Restore original weights (remove IC-LoRA)."""
        if not self.lora_configs:
            return

        # Restore base weights (use raw velocity model)
        self._velocity_model.load_weights(list(self.base_transformer_weights.items()))
        mx.eval(self._velocity_model.parameters())

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
        Run the denoising loop (no CFG for IC-LoRA, as it uses simple denoising).

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

            # Create modality
            modality = modality_from_state(
                video_state, context, context_mask, sigma
            )

            # Run model
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
        config: ICLoraConfig,
        images: Optional[List[ImageCondition]] = None,
        video_conditioning: Optional[List[VideoCondition]] = None,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> mx.array:
        """
        Generate video with IC-LoRA control.

        Args:
            text_encoding: Encoded text prompt [B, T, D].
            text_mask: Text attention mask [B, T].
            config: Pipeline configuration.
            images: Optional list of image conditions (frame replacements).
            video_conditioning: List of video control signals (depth, pose, etc.).
            callback: Optional callback(stage, step, total_steps).

        Returns:
            Generated video tensor [F, H, W, C] in pixel space (0-255).
        """
        images = images or []
        video_conditioning = video_conditioning or []

        # Set seed
        mx.random.seed(config.seed)

        # Create noiser and stepper
        noiser = GaussianNoiser()
        stepper = self.diffusion_step

        # ====== STAGE 1: Half resolution with IC-LoRA ======
        stage_1_height = config.height // 2
        stage_1_width = config.width // 2

        # Apply IC-LoRA to transformer
        self._apply_lora()

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
        # 1. Image conditions (replace at frame index)
        image_conditionings = create_image_conditionings(
            images,
            self.video_encoder,
            stage_1_height,
            stage_1_width,
            config.dtype,
        )

        # 2. Video control signals (IC-LoRA conditioning)
        video_conditionings = create_video_conditionings(
            video_conditioning,
            self.video_encoder,
            stage_1_height,
            stage_1_width,
            config.num_frames,
            config.dtype,
        )

        stage_1_conditionings = image_conditionings + video_conditionings

        # Create initial state
        video_state = video_tools.create_initial_state(dtype=config.dtype)

        # Apply conditionings
        video_state = apply_conditionings(video_state, stage_1_conditionings, video_tools)

        # Get stage 1 sigmas (distilled)
        sigmas = mx.array(DISTILLED_SIGMA_VALUES[: config.stage_1_steps + 1])

        # Add noise
        video_state = noiser(video_state, noise_scale=1.0)

        # Stage 1 callback wrapper
        def stage_1_callback(step: int, total: int):
            if callback:
                callback("stage1_iclora", step, total)

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

        # ====== STAGE 2: Upsample and refine WITHOUT IC-LoRA ======
        # Remove IC-LoRA, restore base weights
        self._remove_lora()

        # Upsample the latent 2x
        # CRITICAL: Un-normalize before upsampling, re-normalize after
        # The upsampler model is trained on raw (un-normalized) latents
        # Reference: PyTorch upsample_video() in ltx_core/model/upsampler/model.py
        latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(stage_1_latent)
        upscaled_unnorm = self.spatial_upscaler(latent_unnorm)
        mx.eval(upscaled_unnorm)
        upscaled_latent = self.video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
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

        # Create conditionings at full resolution (only image conditions, no IC-LoRA)
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

        # Get stage 2 (distilled) sigmas
        distilled_sigmas = mx.array(
            STAGE_2_DISTILLED_SIGMA_VALUES[: config.stage_2_steps + 1]
        )

        # Add noise at lower scale for refinement
        video_state_2 = noiser(video_state_2, noise_scale=float(distilled_sigmas[0]))

        # Stage 2 callback wrapper
        def stage_2_callback(step: int, total: int):
            if callback:
                callback("stage2_refine", step, total)

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


def create_ic_lora_pipeline(
    transformer: LTXModel,
    video_encoder: SimpleVideoEncoder,
    video_decoder: SimpleVideoDecoder,
    spatial_upscaler: SpatialUpscaler,
    base_transformer_weights: Dict[str, mx.array],
    lora_configs: Optional[List[LoRAConfig]] = None,
) -> ICLoraPipeline:
    """
    Create an IC-LoRA pipeline.

    Args:
        transformer: LTX transformer model.
        video_encoder: VAE encoder.
        video_decoder: VAE decoder.
        spatial_upscaler: 2x spatial upscaler.
        base_transformer_weights: Original weights for LoRA restoration.
        lora_configs: IC-LoRA configurations.

    Returns:
        Configured ICLoraPipeline.
    """
    return ICLoraPipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        spatial_upscaler=spatial_upscaler,
        base_transformer_weights=base_transformer_weights,
        lora_configs=lora_configs,
    )
