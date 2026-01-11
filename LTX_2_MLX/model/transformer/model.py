"""LTX-2 Transformer Model for MLX (Video-Only)."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .rope import LTXRopeType, precompute_freqs_cis
from .timestep_embedding import AdaLayerNormSingle
from .transformer import BasicTransformerBlock, BasicAVTransformerBlock, TransformerArgs, TransformerConfig




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
    - Position embedding computation (RoPE) with caching
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
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        cache_position_embeddings: bool = True,
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
        self.cache_position_embeddings = cache_position_embeddings
        # Cache for position embeddings indexed by shape tuple
        self._pe_cache = {}

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
        # AdaLayerNormSingle now returns tuple of (processed_emb, raw_embedded_timestep)
        emb, embedded_timestep = self.adaln(timestep.flatten())

        # Reshape processed emb to (B, num_tokens, num_embeddings, inner_dim)
        # The processed emb has shape (B, num_embeddings * inner_dim)
        num_embeddings = 6  # scale, shift, gate for self-attn and ffn
        emb = emb.reshape(batch_size, -1, num_embeddings, self.inner_dim)

        # Reshape raw embedded_timestep to (B, num_tokens, inner_dim)
        # This is the pre-linear embedding used for final output modulation
        embedded_timestep = embedded_timestep.reshape(batch_size, -1, self.inner_dim)

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
        Prepare RoPE positional embeddings with caching.

        Position embeddings are expensive to compute but only depend on the
        position grid shape, not the actual latent values. During denoising,
        the latent shape stays constant, so we can cache and reuse embeddings.

        Args:
            positions: Position indices, shape (B, n_dims, T, 2) where last dim is [start, end].

        Returns:
            Tuple of (cos_freq, sin_freq) for RoPE.
        """
        # Use shape as cache key (positions have same shape across denoising steps)
        cache_key = positions.shape
        if self.cache_position_embeddings and cache_key in self._pe_cache:
            return self._pe_cache[cache_key]

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

        # Cache the result
        if self.cache_position_embeddings:
            self._pe_cache[cache_key] = pe

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


class MultiModalTransformerArgsPreprocessor:
    """
    Preprocesses inputs for AudioVideo transformer blocks.

    Extends TransformerArgsPreprocessor to handle cross-modal attention:
    - Separate positional embeddings for cross-attention (with caching)
    - Cross-attention timestep embeddings (scale/shift and gate)
    """

    def __init__(
        self,
        simple_preprocessor: TransformerArgsPreprocessor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        cross_pe_max_pos: int,
        audio_cross_attention_dim: int,
        av_ca_timestep_scale_multiplier: int = 1000,
    ):
        """
        Initialize multi-modal preprocessor.

        Args:
            simple_preprocessor: Base preprocessor for video/audio.
            cross_scale_shift_adaln: AdaLN for cross-attention scale/shift.
            cross_gate_adaln: AdaLN for cross-attention gate.
            cross_pe_max_pos: Max position for cross-modal RoPE.
            audio_cross_attention_dim: Dimension for audio cross-attention.
            av_ca_timestep_scale_multiplier: Scale for cross-attention timestep.
        """
        self.simple_preprocessor = simple_preprocessor
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
        # Cache for cross-modal position embeddings
        self._cross_pe_cache = {}

    def _prepare_cross_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare cross-modal positional embeddings with caching.

        Uses only the temporal dimension for cross-modal attention.
        """
        # Use shape as cache key
        cache_key = positions.shape
        if cache_key in self._cross_pe_cache:
            return self._cross_pe_cache[cache_key]

        # Use only the first dimension (temporal) for cross-modal attention
        temporal_positions = positions[:, 0:1, :]

        pe = precompute_freqs_cis(
            indices_grid=temporal_positions,
            dim=self.audio_cross_attention_dim,
            out_dtype=mx.float32,
            theta=self.simple_preprocessor.positional_embedding_theta,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            rope_type=self.simple_preprocessor.rope_type,
        )

        # Cache the result
        self._cross_pe_cache[cache_key] = pe
        return pe

    def _prepare_cross_attention_timestep(
        self,
        timestep: mx.array,
        batch_size: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare cross-attention timestep embeddings.

        Returns scale/shift and gate embeddings separately.
        """
        scaled_timestep = timestep * self.simple_preprocessor.timestep_scale_multiplier

        # Scale/shift timestep (AdaLayerNormSingle returns tuple, we only need processed emb)
        scale_shift_emb, _ = self.cross_scale_shift_adaln(scaled_timestep.flatten())
        scale_shift_emb = scale_shift_emb.reshape(batch_size, -1, 4, self.simple_preprocessor.inner_dim)

        # Gate timestep (with AV CA scale)
        av_ca_factor = self.av_ca_timestep_scale_multiplier / self.simple_preprocessor.timestep_scale_multiplier
        gate_emb, _ = self.cross_gate_adaln((scaled_timestep * av_ca_factor).flatten())
        gate_emb = gate_emb.reshape(batch_size, -1, 1, self.simple_preprocessor.inner_dim)

        return scale_shift_emb, gate_emb

    def prepare(self, modality: Modality) -> TransformerArgs:
        """
        Prepare all inputs for AudioVideo transformer blocks.

        Args:
            modality: Input modality data (video or audio).

        Returns:
            TransformerArgs with cross-modal attention fields populated.
        """
        # Get basic transformer args
        args = self.simple_preprocessor.prepare(modality)

        # Add cross-modal positional embeddings
        cross_pe = self._prepare_cross_positional_embeddings(modality.positions)

        # Add cross-attention timestep embeddings
        batch_size = args.x.shape[0]
        cross_scale_shift, cross_gate = self._prepare_cross_attention_timestep(
            modality.timesteps, batch_size
        )

        return args.replace(
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift,
            cross_gate_timestep=cross_gate,
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
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        compute_dtype: mx.Dtype = mx.float32,
        low_memory: bool = False,
        fast_mode: bool = False,
        cross_attn_scale_late: float = 1.0,
        cross_attn_scale_start_layer: int = 40,
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
            rope_type: Type of RoPE (SPLIT).
            compute_dtype: Dtype for computation (float32 or float16).
            low_memory: If True, use aggressive memory optimization (eval every 4 layers).
            fast_mode: If True, skip intermediate evals for faster inference (uses more memory).
                This allows MLX's lazy evaluation to batch more operations together.
            cross_attn_scale_late: Cross-attention scaling for late layers.
                Higher values (5-10) increase text conditioning influence and
                preserve semantic differentiation in late transformer layers.
                Default 1.0 (no scaling).
            cross_attn_scale_start_layer: Layer at which to start applying
                cross_attn_scale_late. Default 40.
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
        self.low_memory = low_memory
        self.fast_mode = fast_mode
        # Eval frequency: 0 for fast_mode (no intermediate evals),
        # 4 for low_memory, 8 otherwise
        if fast_mode:
            self._eval_frequency = 0  # Skip all intermediate evals
        elif low_memory:
            self._eval_frequency = 4
        else:
            self._eval_frequency = 8

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
        # Apply cross_attn_scale_late to late layers for better text differentiation
        self.transformer_blocks = [
            BasicTransformerBlock(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_dim=cross_attention_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                cross_attn_scale=cross_attn_scale_late if i >= cross_attn_scale_start_layer else 1.0,
            )
            for i in range(num_layers)
        ]

        # Output projection
        # Note: scale_shift_table kept as float32 for numerical stability, even in FP16 mode
        self.scale_shift_table = mx.zeros((2, self.inner_dim), dtype=mx.float32)
        self.norm_out = nn.LayerNorm(self.inner_dim, affine=False, eps=norm_eps)
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

    def set_cross_attn_scale(
        self,
        scale: float,
        start_layer: int = 40,
    ) -> None:
        """
        Set cross-attention scaling for late layers.

        This can be called after model initialization/weight loading to
        adjust cross-attention scaling without recreating the model.

        Args:
            scale: Cross-attention scale factor. Higher values (5-10) increase
                text conditioning influence in late layers.
            start_layer: Layer at which to start applying scaling. Default 40.
        """
        for i, block in enumerate(self.transformer_blocks):
            block.cross_attn_scale = scale if i >= start_layer else 1.0

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
            # Force evaluation periodically to release intermediate tensors
            # This reduces peak memory by preventing lazy evaluation buildup
            # fast_mode: no intermediate evals (faster but more memory)
            # low_memory mode: every 4 layers, normal: every 8 layers
            if self._eval_frequency > 0 and (i + 1) % self._eval_frequency == 0:
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
        x = self.norm_out(x)
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


class LTXAVModel(nn.Module):
    """
    LTX-2 AudioVideo Transformer model.

    Architecture:
    - Input: Patchified video latents + audio latents
    - 48 transformer blocks with:
      - Video self-attention, cross-attention (text), cross-attention (audio)
      - Audio self-attention, cross-attention (text), cross-attention (video)
      - Video and audio FFN
    - AdaLN conditioning on timestep
    - Output: Velocity predictions for both video and audio

    The cross-modal attention enables synchronized audio-video generation.
    """

    # Audio configuration constants
    AUDIO_ATTENTION_HEADS = 32
    AUDIO_HEAD_DIM = 64
    AUDIO_IN_CHANNELS = 128  # Audio VAE latent channels
    AUDIO_OUT_CHANNELS = 128
    AUDIO_CROSS_PE_MAX_POS = 16384

    def __init__(
        self,
        # Video config
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        # Shared config
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: int = 1000,
        av_ca_timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        compute_dtype: mx.Dtype = mx.float32,
        low_memory: bool = False,
    ):
        """
        Initialize LTX AudioVideo model.

        Args:
            num_attention_heads: Number of video attention heads (32).
            attention_head_dim: Video dimension per head (128).
            in_channels: Video input channels from VAE (128).
            out_channels: Video output channels (128).
            num_layers: Number of transformer blocks (48).
            cross_attention_dim: Text context dimension (4096).
            norm_eps: Epsilon for normalization.
            caption_channels: Caption embedding dimension (3840 from Gemma).
            positional_embedding_theta: Base theta for RoPE.
            positional_embedding_max_pos: Max positions [time, height, width].
            timestep_scale_multiplier: Scale for timestep (1000).
            av_ca_timestep_scale_multiplier: Scale for AV cross-attention timestep.
            use_middle_indices_grid: Use middle of position bounds for RoPE.
            rope_type: Type of RoPE (INTERLEAVED).
            compute_dtype: Dtype for computation (float32 or float16).
            low_memory: If True, use aggressive memory optimization (eval every 4 layers).
        """
        super().__init__()

        self.rope_type = rope_type
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.use_middle_indices_grid = use_middle_indices_grid
        self.norm_eps = norm_eps
        self.compute_dtype = compute_dtype
        self.num_layers = num_layers
        self.low_memory = low_memory
        # Eval frequency: every 4 layers for low_memory, every 8 otherwise
        self._eval_frequency = 4 if low_memory else 8

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos

        self.num_attention_heads = num_attention_heads
        self.video_inner_dim = num_attention_heads * attention_head_dim  # 4096
        self.audio_inner_dim = self.AUDIO_ATTENTION_HEADS * self.AUDIO_HEAD_DIM  # 2048

        # =================
        # VIDEO COMPONENTS
        # =================
        # Input projection: latent -> inner_dim
        self.patchify_proj = nn.Linear(in_channels, self.video_inner_dim, bias=True)

        # AdaLN for timestep conditioning
        self.adaln_single = AdaLayerNormSingle(self.video_inner_dim)

        # Caption projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.video_inner_dim,
        )

        # Output projection
        # Note: scale_shift_table kept as float32 for numerical stability, even in FP16 mode
        self.scale_shift_table = mx.zeros((2, self.video_inner_dim), dtype=mx.float32)
        self.norm_out = nn.LayerNorm(self.video_inner_dim, affine=False, eps=norm_eps)
        self.proj_out = nn.Linear(self.video_inner_dim, out_channels)

        # Cross-modal attention AdaLN (video side)
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.video_inner_dim, num_embeddings=4
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.video_inner_dim, num_embeddings=1
        )

        # =================
        # AUDIO COMPONENTS
        # =================
        # Input projection: audio latent -> audio inner_dim
        self.audio_patchify_proj = nn.Linear(
            self.AUDIO_IN_CHANNELS, self.audio_inner_dim, bias=True
        )

        # AdaLN for timestep conditioning
        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim)

        # Caption projection for audio
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
        )

        # Output projection
        # Note: scale_shift_table kept as float32 for numerical stability, even in FP16 mode
        self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim), dtype=mx.float32)
        self.audio_norm_out = nn.LayerNorm(self.audio_inner_dim, affine=False, eps=norm_eps)
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, self.AUDIO_OUT_CHANNELS)

        # Cross-modal attention AdaLN (audio side)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim, num_embeddings=4
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim, num_embeddings=1
        )

        # =================
        # TRANSFORMER BLOCKS
        # =================
        video_config = TransformerConfig(
            dim=self.video_inner_dim,
            heads=num_attention_heads,
            d_head=attention_head_dim,
            context_dim=cross_attention_dim,
        )
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.AUDIO_ATTENTION_HEADS,
            d_head=self.AUDIO_HEAD_DIM,
            context_dim=cross_attention_dim,  # Same text context
        )

        self.transformer_blocks = [
            BasicAVTransformerBlock(
                idx=i,
                video_config=video_config,
                audio_config=audio_config,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            for i in range(num_layers)
        ]

        # =================
        # PREPROCESSORS
        # =================
        # Video preprocessor
        video_simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.patchify_proj,
            adaln=self.adaln_single,
            caption_projection=self.caption_projection,
            inner_dim=self.video_inner_dim,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            use_middle_indices_grid=self.use_middle_indices_grid,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
        )
        self._video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
            simple_preprocessor=video_simple_preprocessor,
            cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
            cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
            audio_cross_attention_dim=self.audio_inner_dim,
            av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
        )

        # Audio preprocessor
        audio_simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            caption_projection=self.audio_caption_projection,
            inner_dim=self.audio_inner_dim,
            max_pos=[self.AUDIO_CROSS_PE_MAX_POS],  # 1D for audio
            num_attention_heads=self.AUDIO_ATTENTION_HEADS,
            use_middle_indices_grid=True,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
        )
        self._audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
            simple_preprocessor=audio_simple_preprocessor,
            cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
            cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
            audio_cross_attention_dim=self.audio_inner_dim,
            av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
        )

    def _process_transformer_blocks(
        self,
        video_args: TransformerArgs,
        audio_args: TransformerArgs,
    ) -> Tuple[TransformerArgs, TransformerArgs]:
        """
        Process all transformer blocks for both video and audio.

        Args:
            video_args: Preprocessed video transformer arguments.
            audio_args: Preprocessed audio transformer arguments.

        Returns:
            Tuple of (updated_video_args, updated_audio_args) after all blocks.
        """
        for i, block in enumerate(self.transformer_blocks):
            video_args, audio_args = block(video_args, audio_args)
            # Force evaluation periodically to release intermediate tensors
            # low_memory mode: every 4 layers, normal: every 8 layers
            if (i + 1) % self._eval_frequency == 0:
                mx.eval(video_args.x)
                mx.eval(audio_args.x)
        return video_args, audio_args

    def _process_video_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """Process video output with final normalization and projection."""
        scale_shift_values = (
            self.scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        )
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]

        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)
        return x

    def _process_audio_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """Process audio output with final normalization and projection."""
        scale_shift_values = (
            self.audio_scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        )
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]

        x = self.audio_norm_out(x)
        x = x * (1 + scale) + shift
        x = self.audio_proj_out(x)
        return x

    def __call__(
        self,
        video: Modality,
        audio: Modality,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass for AudioVideo model.

        Args:
            video: Input video modality data.
            audio: Input audio modality data.

        Returns:
            Tuple of (video_velocity, audio_velocity) predictions.
        """
        # Cast inputs to compute dtype for memory efficiency
        if self.compute_dtype != mx.float32:
            video = Modality(
                latent=video.latent.astype(self.compute_dtype),
                context=video.context.astype(self.compute_dtype),
                context_mask=video.context_mask,
                timesteps=video.timesteps,
                positions=video.positions,
                enabled=video.enabled,
            )
            audio = Modality(
                latent=audio.latent.astype(self.compute_dtype),
                context=audio.context.astype(self.compute_dtype),
                context_mask=audio.context_mask,
                timesteps=audio.timesteps,
                positions=audio.positions,
                enabled=audio.enabled,
            )

        # Preprocess inputs
        video_args = self._video_args_preprocessor.prepare(video)

        # Only preprocess audio if enabled and has tokens
        # The transformer blocks check `audio.enabled and ax.size > 0` anyway,
        # so we can skip preprocessing for disabled audio to avoid errors
        if audio.enabled and audio.latent.size > 0:
            audio_args = self._audio_args_preprocessor.prepare(audio)
        else:
            # Create minimal disabled TransformerArgs
            audio_args = TransformerArgs(
                x=mx.zeros((video_args.x.shape[0], 0, self.audio_inner_dim)),
                context=mx.zeros((video_args.x.shape[0], 0, self.audio_inner_dim)),
                timesteps=mx.zeros((video_args.x.shape[0], 0, 6, self.audio_inner_dim)),
                positional_embeddings=(mx.zeros((1,)), mx.zeros((1,))),  # Minimal PE
                enabled=False,
            )

        # Process through transformer blocks
        video_args, audio_args = self._process_transformer_blocks(video_args, audio_args)

        # Process outputs
        video_output = self._process_video_output(video_args.x, video_args.embedded_timestep)

        # Only process audio output if audio was enabled
        if audio_args.enabled and audio_args.x.size > 0:
            audio_output = self._process_audio_output(audio_args.x, audio_args.embedded_timestep)
        else:
            # Return empty audio output for disabled audio
            audio_output = mx.zeros((video_args.x.shape[0], 0, self.AUDIO_OUT_CHANNELS))

        # Cast outputs back to float32 for numerical stability
        if self.compute_dtype != mx.float32:
            video_output = video_output.astype(mx.float32)
            audio_output = audio_output.astype(mx.float32)

        return video_output, audio_output


class X0AVModel(nn.Module):
    """
    Wrapper that returns denoised outputs instead of velocities for AudioVideo model.

    Converts velocity predictions to denoised predictions using:
    x0 = x - sigma * velocity
    """

    def __init__(self, velocity_model: LTXAVModel):
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        video: Modality,
        audio: Modality,
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute denoised video and audio from noisy inputs.

        Args:
            video: Input video modality (noisy latent).
            audio: Input audio modality (noisy latent).

        Returns:
            Tuple of (denoised_video, denoised_audio).
        """
        video_velocity, audio_velocity = self.velocity_model(video, audio)

        # Convert velocity to denoised: x0 = x - sigma * v
        video_timesteps = video.timesteps
        if video_timesteps.ndim == 1:
            video_timesteps = video_timesteps[:, None, None]
        elif video_timesteps.ndim == 2:
            video_timesteps = video_timesteps[:, :, None]

        audio_timesteps = audio.timesteps
        if audio_timesteps.ndim == 1:
            audio_timesteps = audio_timesteps[:, None, None]
        elif audio_timesteps.ndim == 2:
            audio_timesteps = audio_timesteps[:, :, None]

        denoised_video = video.latent - video_timesteps * video_velocity
        denoised_audio = audio.latent - audio_timesteps * audio_velocity

        return denoised_video, denoised_audio
