"""LTX-2 Transformer Model for MLX (Video-Only)."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import rms_norm
from .rope import LTXRopeType, precompute_freqs_cis
from .timestep_embedding import AdaLayerNormSingle
from .transformer import BasicTransformerBlock, TransformerArgs


class LTXModelType(Enum):
    """Model type variants."""

    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings with GELU activation.

    Adapted from PixArt-alpha implementation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, out_features, bias=True)

    def __call__(self, caption: mx.array) -> mx.array:
        hidden_states = self.linear_1(caption)
        hidden_states = nn.gelu_approx(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@dataclass
class Modality:
    """Input modality data (video or audio)."""

    latent: mx.array  # Shape: (B, T, C) - patchified latents
    context: mx.array  # Shape: (B, S, C_ctx) - text context
    context_mask: Optional[mx.array]  # Shape: (B, S) or (B, 1, S, S)
    timesteps: mx.array  # Shape: (B,) or (B, T) - timestep values
    positions: mx.array  # Shape: (B, n_dims, T) - position indices
    enabled: bool = True


class TransformerArgsPreprocessor:
    """
    Preprocesses inputs for transformer blocks.

    Handles:
    - Patchify projection (linear embedding)
    - Timestep embedding via AdaLN
    - Caption projection
    - Position embedding computation (RoPE)
    """

    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: List[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool = True,
        timestep_scale_multiplier: int = 1000,
        positional_embedding_theta: float = 10000.0,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ):
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

    def _prepare_timestep(
        self,
        timestep: mx.array,
        batch_size: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare timestep embeddings.

        Args:
            timestep: Timestep values, shape (B,) or (B, T).
            batch_size: Batch size.

        Returns:
            Tuple of (timestep_emb, embedded_timestep).
        """
        timestep = timestep * self.timestep_scale_multiplier
        emb = self.adaln(timestep.flatten())

        # Reshape to (B, num_embeddings, inner_dim)
        # The adaln returns (B, num_embeddings * inner_dim)
        num_embeddings = 6  # scale, shift, gate for self-attn and ffn
        emb = emb.reshape(batch_size, -1, num_embeddings, self.inner_dim)

        # embedded_timestep is used for final output modulation
        # Just take the mean across embedding dimension for the embedded version
        embedded_timestep = emb[:, :, :2, :].mean(axis=2)

        return emb, embedded_timestep

    def _prepare_context(
        self,
        context: mx.array,
        x: mx.array,
    ) -> mx.array:
        """
        Prepare context (caption) for cross-attention.

        Args:
            context: Caption embeddings, shape (B, S, C_ctx).
            x: Projected hidden states (for batch size).

        Returns:
            Projected context, shape (B, S, inner_dim).
        """
        batch_size = x.shape[0]
        context = self.caption_projection(context)
        context = context.reshape(batch_size, -1, x.shape[-1])
        return context

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[mx.array],
    ) -> Optional[mx.array]:
        """
        Prepare attention mask for cross-attention.

        Converts boolean mask to additive mask for softmax.
        """
        if attention_mask is None:
            return None

        # If already a float mask, return as-is
        if attention_mask.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return attention_mask

        # Convert boolean mask to additive mask
        # True = attend, False = don't attend
        # For additive mask: 0 = attend, -inf = don't attend
        mask = (1 - attention_mask.astype(mx.float32)) * -1e9
        mask = mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
        return mask

    def _prepare_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare RoPE positional embeddings.

        Args:
            positions: Position indices, shape (B, n_dims, T).

        Returns:
            Tuple of (cos_freq, sin_freq) for RoPE.
        """
        pe = precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=mx.float32,
            theta=self.positional_embedding_theta,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
        )
        return pe

    def prepare(self, modality: Modality) -> TransformerArgs:
        """
        Prepare all inputs for transformer blocks.

        Args:
            modality: Input modality data.

        Returns:
            TransformerArgs ready for transformer blocks.
        """
        # Project latents to inner dimension
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]

        # Prepare timestep embeddings
        timestep_emb, embedded_timestep = self._prepare_timestep(
            modality.timesteps, batch_size
        )

        # Prepare context (caption projection)
        context = self._prepare_context(modality.context, x)

        # Prepare attention mask
        attention_mask = self._prepare_attention_mask(modality.context_mask)

        # Prepare positional embeddings (RoPE)
        pe = self._prepare_positional_embeddings(modality.positions)

        return TransformerArgs(
            x=x,
            context=context,
            timesteps=timestep_emb,
            positional_embeddings=pe,
            context_mask=attention_mask,
            embedded_timestep=embedded_timestep,
        )


class LTXModel(nn.Module):
    """
    LTX-2 Transformer model (video-only variant).

    Architecture:
    - Input: Patchified video latents
    - 48 transformer blocks with self-attention, cross-attention, and FFN
    - AdaLN conditioning on timestep
    - Output: Velocity predictions for diffusion

    This is the core denoising model that predicts velocities.
    """

    def __init__(
        self,
        model_type: LTXModelType = LTXModelType.VideoOnly,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        """
        Initialize LTX model.

        Args:
            model_type: Type of model (VideoOnly for this implementation).
            num_attention_heads: Number of attention heads (32).
            attention_head_dim: Dimension per head (128).
            in_channels: Input channels from VAE (128).
            out_channels: Output channels (128).
            num_layers: Number of transformer blocks (48).
            cross_attention_dim: Text context dimension (4096).
            norm_eps: Epsilon for normalization.
            caption_channels: Caption embedding dimension (3840 from Gemma).
            positional_embedding_theta: Base theta for RoPE.
            positional_embedding_max_pos: Max positions [time, height, width].
            timestep_scale_multiplier: Scale for timestep (1000).
            use_middle_indices_grid: Use middle of position bounds for RoPE.
            rope_type: Type of RoPE (INTERLEAVED).
            compute_dtype: Dtype for computation (float32 or float16).
        """
        super().__init__()

        if model_type != LTXModelType.VideoOnly:
            raise NotImplementedError("Only VideoOnly model type is supported in MLX port")

        self.model_type = model_type
        self.rope_type = rope_type
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.use_middle_indices_grid = use_middle_indices_grid
        self.norm_eps = norm_eps
        self.compute_dtype = compute_dtype

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos

        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim

        # Input projection: latent -> inner_dim
        self.patchify_proj = nn.Linear(in_channels, self.inner_dim, bias=True)

        # AdaLN for timestep conditioning
        self.adaln_single = AdaLayerNormSingle(self.inner_dim)

        # Caption projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
        )

        # Transformer blocks
        self.transformer_blocks = [
            BasicTransformerBlock(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_dim=cross_attention_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]

        # Output projection
        self.scale_shift_table = mx.zeros((2, self.inner_dim))
        self.proj_out = nn.Linear(self.inner_dim, out_channels)

        # Create preprocessor
        self._video_args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.patchify_proj,
            adaln=self.adaln_single,
            caption_projection=self.caption_projection,
            inner_dim=self.inner_dim,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            use_middle_indices_grid=self.use_middle_indices_grid,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
        )

    def _process_transformer_blocks(
        self,
        args: TransformerArgs,
    ) -> TransformerArgs:
        """
        Process all transformer blocks.

        Args:
            args: Preprocessed transformer arguments.

        Returns:
            Updated TransformerArgs after all blocks.
        """
        for i, block in enumerate(self.transformer_blocks):
            args = block(args)
            # Force evaluation every 8 layers to release intermediate tensors
            # This reduces peak memory by preventing lazy evaluation buildup
            if (i + 1) % 8 == 0:
                mx.eval(args.x)
        return args

    def _process_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """
        Process output with final normalization and projection.

        Args:
            x: Hidden states from transformer, shape (B, T, inner_dim).
            embedded_timestep: Embedded timestep for modulation.

        Returns:
            Output velocity predictions, shape (B, T, out_channels).
        """
        # Apply scale-shift modulation from timestep
        # scale_shift_table: (2, inner_dim)
        # embedded_timestep: (B, T, inner_dim) or (B, 1, inner_dim)
        scale_shift_values = (
            self.scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        )
        shift = scale_shift_values[:, :, 0, :]  # (B, T, inner_dim)
        scale = scale_shift_values[:, :, 1, :]  # (B, T, inner_dim)

        # Apply layer norm and modulation
        x = rms_norm(x, eps=self.norm_eps)
        x = x * (1 + scale) + shift

        # Project to output channels
        x = self.proj_out(x)

        return x

    def __call__(
        self,
        video: Modality,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            video: Input video modality data.

        Returns:
            Velocity predictions, shape (B, T, out_channels).
        """
        # Cast input to compute dtype for memory efficiency
        if self.compute_dtype != mx.float32:
            video = Modality(
                latent=video.latent.astype(self.compute_dtype),
                context=video.context.astype(self.compute_dtype),
                context_mask=video.context_mask,
                timesteps=video.timesteps,
                positions=video.positions,
                enabled=video.enabled,
            )

        # Preprocess inputs
        args = self._video_args_preprocessor.prepare(video)

        # Process through transformer blocks
        args = self._process_transformer_blocks(args)

        # Process output
        output = self._process_output(args.x, args.embedded_timestep)

        # Cast output back to float32 for numerical stability in diffusion steps
        if self.compute_dtype != mx.float32:
            output = output.astype(mx.float32)

        return output


class X0Model(nn.Module):
    """
    Wrapper that returns denoised outputs instead of velocities.

    Converts velocity predictions to denoised predictions using:
    x0 = x - sigma * velocity
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        video: Modality,
    ) -> mx.array:
        """
        Compute denoised video from noisy input.

        Args:
            video: Input video modality (noisy latent).

        Returns:
            Denoised video latent.
        """
        velocity = self.velocity_model(video)

        # Convert velocity to denoised: x0 = x - sigma * v
        # video.timesteps contains sigma values
        timesteps = video.timesteps
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None, None]
        elif timesteps.ndim == 2:
            timesteps = timesteps[:, :, None]

        denoised = video.latent - timesteps * velocity
        return denoised
