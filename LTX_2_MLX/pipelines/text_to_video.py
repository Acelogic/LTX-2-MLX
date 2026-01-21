"""Text-to-video generation pipeline for LTX-2 MLX."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import mlx.core as mx

from ..components.diffusion_steps import EulerDiffusionStep
from ..components.guiders import CFGGuider
from ..components.noisers import GaussianNoiser
from ..components.patchifiers import VideoLatentPatchifier
from ..components.schedulers import get_sigma_schedule
from ..model.transformer import LTXModel, Modality, create_position_grid
from ..model.video_vae import VideoDecoder
from ..types import VideoLatentShape


@dataclass
class GenerationConfig:
    """Configuration for video generation."""

    # Video dimensions
    height: int = 480
    width: int = 704
    num_frames: int = 121  # frames % 8 == 1

    # Generation parameters
    num_inference_steps: int = 50
    cfg_scale: float = 7.5
    seed: Optional[int] = None

    # Model configuration
    use_distilled: bool = False
    precision: str = "float16"  # or "float32", "bfloat16"

    def __post_init__(self):
        # Validate frame count
        if self.num_frames % 8 != 1:
            raise ValueError(
                f"num_frames must be 8*k + 1, got {self.num_frames}. "
                f"Valid values: 1, 9, 17, 25, 33, ..., 121"
            )


@dataclass
class PipelineState:
    """State during pipeline execution."""

    latent: mx.array  # Current noisy/denoised latent
    sigmas: mx.array  # Sigma schedule
    step: int  # Current step
    text_context: mx.array  # Encoded text
    text_mask: mx.array  # Text attention mask


class TextToVideoPipeline:
    """
    Text-to-video generation pipeline.

    This implements the core denoising loop:
    1. Initialize noise latent
    2. Encode text prompt
    3. Iteratively denoise using transformer
    4. Decode latent to video

    The pipeline supports CFG (classifier-free guidance) for
    prompt following.
    """

    def __init__(
        self,
        transformer: LTXModel,
        decoder: VideoDecoder,
        patchifier: VideoLatentPatchifier,
        noiser: Optional[GaussianNoiser] = None,
        guider: Optional[CFGGuider] = None,
        diffusion_step: Optional[EulerDiffusionStep] = None,
    ):
        """
        Initialize pipeline.

        Args:
            transformer: LTX transformer model.
            decoder: Video VAE decoder.
            patchifier: Latent patchifier.
            noiser: Noise generator.
            guider: CFG guider.
            diffusion_step: Euler diffusion step.
        """
        self.transformer = transformer
        self.decoder = decoder
        self.patchifier = patchifier
        self.noiser = noiser or GaussianNoiser()
        self.guider = guider or CFGGuider(scale=7.5)
        self.diffusion_step = diffusion_step or EulerDiffusionStep()

    def get_latent_shape(self, config: GenerationConfig) -> VideoLatentShape:
        """
        Get latent shape for given video dimensions.

        The VAE compresses:
        - Spatial: 32x (height/32, width/32)
        - Temporal: 8x (frames/8, rounded up)

        Args:
            config: Generation configuration.

        Returns:
            VideoLatentShape for the latent.
        """
        # VAE compression factors
        spatial_factor = 32
        temporal_factor = 8

        latent_height = config.height // spatial_factor
        latent_width = config.width // spatial_factor
        latent_frames = (config.num_frames - 1) // temporal_factor + 1

        return VideoLatentShape(
            batch=1,
            channels=128,  # LTX latent channels
            frames=latent_frames,
            height=latent_height,
            width=latent_width,
        )

    def initialize_latent(
        self,
        shape: VideoLatentShape,
        seed: Optional[int] = None,
    ) -> mx.array:
        """
        Initialize noise latent.

        Args:
            shape: Latent shape.
            seed: Random seed.

        Returns:
            Initial noise tensor.
        """
        if seed is not None:
            mx.random.seed(seed)

        return mx.random.normal(shape=(
            shape.batch,
            shape.channels,
            shape.frames,
            shape.height,
            shape.width,
        ))

    def prepare_text_context(
        self,
        text_encoding: mx.array,
        text_mask: mx.array,
        cfg_scale: float,
        negative_encoding: Optional[mx.array] = None,
        negative_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare text context for CFG.

        For CFG, we need both conditional and unconditional contexts.

        Args:
            text_encoding: Encoded text [B, T, D].
            text_mask: Attention mask [B, T].
            cfg_scale: CFG scale.
            negative_encoding: Optional pre-encoded negative prompt [B, T, D].
                If None, uses zeros (less accurate than encoding empty string).
            negative_mask: Optional attention mask for negative prompt [B, T].
                If None and negative_encoding is provided, uses ones.

        Returns:
            Tuple of (doubled_encoding, doubled_mask) for CFG.
        """
        if cfg_scale > 1.0:
            # Use provided negative encoding or fall back to zeros
            if negative_encoding is not None:
                uncond = negative_encoding
                uncond_mask = (
                    negative_mask
                    if negative_mask is not None
                    else mx.ones((negative_encoding.shape[0], negative_encoding.shape[1]))
                )
            else:
                # Fallback to zeros (less accurate than encoded empty string)
                uncond = mx.zeros_like(text_encoding)
                uncond_mask = mx.ones_like(text_mask)

            # Stack conditional and unconditional
            context = mx.concatenate([text_encoding, uncond], axis=0)
            mask = mx.concatenate([text_mask, uncond_mask], axis=0)
            return context, mask

        return text_encoding, text_mask

    def denoise_step(
        self,
        latent: mx.array,
        sigma: float,
        sigma_next: float,
        text_context: mx.array,
        text_mask: mx.array,
        positions: mx.array,
        cfg_scale: float,
    ) -> mx.array:
        """
        Perform one denoising step.

        Args:
            latent: Current noisy latent [B, C, F, H, W].
            sigma: Current noise level.
            sigma_next: Next noise level.
            text_context: Text context for cross-attention.
            text_mask: Text attention mask.
            positions: Position grid for RoPE.
            cfg_scale: CFG scale.

        Returns:
            Denoised latent.
        """
        # Patchify latent: [B, C, F, H, W] -> [B, T, C]
        patchified = self.patchifier.patchify(latent)
        batch_size = patchified.shape[0]

        # Prepare timestep (sigma)
        timestep = mx.array([sigma] * batch_size)

        # For CFG, duplicate latent and timestep
        if cfg_scale > 1.0:
            patchified = mx.concatenate([patchified, patchified], axis=0)
            timestep = mx.concatenate([timestep, timestep], axis=0)
            positions = mx.concatenate([positions, positions], axis=0)

        # Create modality input
        modality = Modality(
            latent=patchified,
            context=text_context,
            context_mask=text_mask,
            timesteps=timestep,
            positions=positions,
            enabled=True,
        )

        # Run transformer
        velocity = self.transformer(modality)

        # Apply CFG
        if cfg_scale > 1.0:
            cond, uncond = mx.split(velocity, 2, axis=0)
            velocity = self.guider.guide(cond, uncond)

        # Unpatchify: [B, T, C] -> [B, C, F, H, W]
        velocity = self.patchifier.unpatchify(
            velocity,
            frames=latent.shape[2],
            height=latent.shape[3],
            width=latent.shape[4],
        )

        # Euler step
        denoised = self.diffusion_step.step(
            latent=latent,
            velocity=velocity,
            sigma=sigma,
            sigma_next=sigma_next,
        )

        return denoised

    def __call__(
        self,
        text_encoding: mx.array,
        text_mask: mx.array,
        config: GenerationConfig,
        callback: Optional[Callable[[int, int, mx.array], None]] = None,
        negative_encoding: Optional[mx.array] = None,
        negative_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Generate video from text.

        Args:
            text_encoding: Encoded text prompt [B, T, D].
            text_mask: Text attention mask [B, T].
            config: Generation configuration.
            callback: Optional callback(step, total_steps, latent).
            negative_encoding: Optional pre-encoded negative prompt [B, T, D].
                For best results, encode an empty string through the text encoder.
                If None, falls back to zeros (matches OneStagePipeline behavior).
            negative_mask: Optional attention mask for negative prompt [B, T].

        Returns:
            Generated video tensor [B, C, F, H, W] in pixel space.
        """
        # Get latent shape
        latent_shape = self.get_latent_shape(config)

        # Initialize noise
        latent = self.initialize_latent(latent_shape, config.seed)

        # Get sigma schedule
        sigmas = get_sigma_schedule(
            num_steps=config.num_inference_steps,
            distilled=config.use_distilled,
        )

        # Prepare text context for CFG
        context, mask = self.prepare_text_context(
            text_encoding,
            text_mask,
            config.cfg_scale,
            negative_encoding=negative_encoding,
            negative_mask=negative_mask,
        )

        # Create position grid
        positions = create_position_grid(
            batch_size=1,
            frames=latent_shape.frames,
            height=latent_shape.height,
            width=latent_shape.width,
        )

        # Denoising loop
        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])

            latent = self.denoise_step(
                latent=latent,
                sigma=sigma,
                sigma_next=sigma_next,
                text_context=context,
                text_mask=mask,
                positions=positions,
                cfg_scale=config.cfg_scale,
            )

            if callback is not None:
                callback(i + 1, len(sigmas) - 1, latent)

            # Evaluate to prevent memory buildup
            mx.eval(latent)

        # Decode latent to video
        video = self.decoder(latent)

        return video


def create_pipeline(
    transformer: LTXModel,
    decoder: VideoDecoder,
    cfg_scale: float = 7.5,
) -> TextToVideoPipeline:
    """
    Create a text-to-video pipeline with default components.

    Args:
        transformer: LTX transformer model.
        decoder: Video VAE decoder.
        cfg_scale: Default CFG scale.

    Returns:
        Configured TextToVideoPipeline.
    """
    return TextToVideoPipeline(
        transformer=transformer,
        decoder=decoder,
        patchifier=VideoLatentPatchifier(),
        noiser=GaussianNoiser(),
        guider=CFGGuider(scale=cfg_scale),
        diffusion_step=EulerDiffusionStep(),
    )
