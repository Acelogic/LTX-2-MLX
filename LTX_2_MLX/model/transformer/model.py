"""LTX-2 Transformer Model for MLX (Unified Video/Audio)."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .rope import LTXRopeType, precompute_freqs_cis
from .timestep_embedding import AdaLayerNormSingle
from .transformer import BasicTransformerBlock, BasicAVTransformerBlock, TransformerArgs, TransformerConfig
from ...components.perturbations import BatchedPerturbationConfig




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
        rope_type: LTXRopeType = LTXRopeType.SPLIT,  # LTX-2 distilled uses SPLIT
        compute_dtype: mx.Dtype = mx.float32,
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
        self.compute_dtype = compute_dtype

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
        target_dtype: mx.Dtype = mx.float32,
    ) -> Optional[mx.array]:
        """
        Prepare attention mask for cross-attention.

        Converts boolean mask to additive mask for softmax.
        Uses dtype-appropriate masking values to match PyTorch finfo behavior.

        Args:
            attention_mask: Boolean or float mask of shape (B, S).
            target_dtype: Target dtype for the mask (determines mask value).

        Returns:
            Additive attention mask of shape (B, 1, 1, S) or None.
        """
        if attention_mask is None:
            return None

        # If already a float mask, return as-is
        if attention_mask.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return attention_mask

        # Use dtype-appropriate max value (matches PyTorch finfo behavior)
        # PyTorch uses (mask - 1) * finfo(dtype).max
        if target_dtype == mx.float16:
            mask_value = -65504.0  # ~-finfo(float16).max
        elif target_dtype == mx.bfloat16:
            mask_value = -3.38e38  # ~-finfo(bfloat16).max
        else:
            mask_value = -3.40e38  # ~-finfo(float32).max

        # Convert boolean mask to additive mask
        # True = attend (0), False = don't attend (large negative)
        mask = (1 - attention_mask.astype(mx.float32)) * mask_value
        mask = mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
        return mask.astype(target_dtype)

    def _prepare_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare RoPE positional embeddings.

        No caching - matches PyTorch behavior and avoids stale cache bugs when
        fps, causal_fix, or max_pos change while shape stays constant.

        Args:
            positions: Position indices, shape (B, n_dims, T, 2) where last dim is [start, end].

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

        # Prepare attention mask with dtype-appropriate masking values
        attention_mask = self._prepare_attention_mask(
            modality.context_mask, target_dtype=self.compute_dtype
        )

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
    - Separate positional embeddings for cross-attention
    - Cross-attention timestep embeddings (scale/shift and gate)
    """

    def __init__(
        self,
        simple_preprocessor: TransformerArgsPreprocessor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        cross_pe_max_pos: int,
        audio_cross_attention_dim: int,
        av_ca_timestep_scale_multiplier: int = 1,  # PyTorch default (av_ca_factor = 1/1000)
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

    def _prepare_cross_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare cross-modal positional embeddings.

        No caching - matches PyTorch behavior and avoids stale cache bugs.
        Uses only the temporal dimension for cross-modal attention.
        """
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
    LTX-2 Transformer Model (Unified Video/Audio).

    Wrapper for both VideoOnly and AudioVideo variants.
    Architecture:
    - Input: Patchified video latents (and optional audio latents)
    - 48 transformer blocks with self-attention, cross-attention, and FFN
    - AdaLN conditioning on timestep
    - Output: Velocity predictions for diffusion

    This is the core denoising model that predicts velocities.
    """

    # Audio configuration constants
    AUDIO_ATTENTION_HEADS = 32
    AUDIO_HEAD_DIM = 64
    AUDIO_IN_CHANNELS = 128  # Audio VAE latent channels
    AUDIO_OUT_CHANNELS = 128
    # Audio cross-PE max position - PyTorch uses 20 for audio positional max
    # and max(video_t, audio_t) for cross-modal positions (typically 20)
    AUDIO_CROSS_PE_MAX_POS = 20

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
        # AV cross-attn timestep scale: PyTorch defaults to 1, giving av_ca_factor = 1/1000
        # This controls the timestep scaling for audio-video cross attention
        av_ca_timestep_scale_multiplier: int = 1,
        use_middle_indices_grid: bool = True,
        # RoPE type: LTX-2 distilled weights use SPLIT
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        compute_dtype: mx.Dtype = mx.float32,
        low_memory: bool = False,
        fast_mode: bool = False,
    ):
        """
        Initialize LTX model.

        Args:
            model_type: Type of model (VideoOnly, AudioVideo, AudioOnly).
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
            av_ca_timestep_scale_multiplier: Scale for AV cross-attention timestep.
                PyTorch default is 1, giving av_ca_factor = 1/1000.
            use_middle_indices_grid: Use middle of position bounds for RoPE.
            rope_type: Type of RoPE. LTX-2 distilled uses SPLIT.
            compute_dtype: Dtype for computation (float32 or float16).
            low_memory: If True, use aggressive memory optimization (eval every 4 layers).
            fast_mode: If True, skip intermediate evals for faster inference (uses more memory).
        """
        super().__init__()

        self.model_type = model_type
        self.rope_type = rope_type
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.use_middle_indices_grid = use_middle_indices_grid
        self.norm_eps = norm_eps
        self.compute_dtype = compute_dtype
        self.low_memory = low_memory
        self.fast_mode = fast_mode
        
        # Eval frequency setup
        if fast_mode:
            self._eval_frequency = 0
        elif low_memory:
            self._eval_frequency = 4
        else:
            self._eval_frequency = 8

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos

        self.num_attention_heads = num_attention_heads
        
        # Video dimensions
        self.video_inner_dim = num_attention_heads * attention_head_dim
        # Map generic inner_dim to video_inner_dim for compatibility
        self.inner_dim = self.video_inner_dim
        
        # Audio dimensions
        self.audio_inner_dim = self.AUDIO_ATTENTION_HEADS * self.AUDIO_HEAD_DIM

        # =================
        # VIDEO COMPONENTS
        # =================
        if self.model_type.is_video_enabled():
            # Input projection
            self.patchify_proj = nn.Linear(in_channels, self.video_inner_dim, bias=True)

            # AdaLN
            self.adaln_single = AdaLayerNormSingle(self.video_inner_dim)

            # Caption projection
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=self.video_inner_dim,
            )

            # Output projection
            self.scale_shift_table = mx.zeros((2, self.video_inner_dim), dtype=mx.float32)
            self.norm_out = nn.LayerNorm(self.video_inner_dim, affine=False, eps=norm_eps)
            self.proj_out = nn.Linear(self.video_inner_dim, out_channels)

        # =================
        # AUDIO COMPONENTS
        # =================
        if self.model_type.is_audio_enabled():
            # Input projection
            self.audio_patchify_proj = nn.Linear(
                self.AUDIO_IN_CHANNELS, self.audio_inner_dim, bias=True
            )

            # AdaLN
            self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim)

            # Caption projection
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=self.audio_inner_dim,
            )

            # Output projection
            self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim), dtype=mx.float32)
            self.audio_norm_out = nn.LayerNorm(self.audio_inner_dim, affine=False, eps=norm_eps)
            self.audio_proj_out = nn.Linear(self.audio_inner_dim, self.AUDIO_OUT_CHANNELS)

        # =================
        # CROSS-MODAL COMPONENTS
        # =================
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            # Video side
            self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=4
            )
            self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=1
            )
            # Audio side
            self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=4
            )
            self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=1
            )

        # =================
        # TRANSFORMER BLOCKS
        # =================
        video_config = None
        if self.model_type.is_video_enabled():
            video_config = TransformerConfig(
                dim=self.video_inner_dim,
                heads=num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
            )
            
        audio_config = None
        if self.model_type.is_audio_enabled():
            audio_config = TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.AUDIO_ATTENTION_HEADS,
                d_head=self.AUDIO_HEAD_DIM,
                context_dim=cross_attention_dim,
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
        if self.model_type.is_video_enabled():
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
                compute_dtype=self.compute_dtype,
            )
            if self.model_type.is_audio_enabled():
                self._video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                    simple_preprocessor=video_simple_preprocessor,
                    cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                    cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                    cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
                    audio_cross_attention_dim=self.audio_inner_dim,
                    av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
                )
            else:
                self._video_args_preprocessor = video_simple_preprocessor

        if self.model_type.is_audio_enabled():
            audio_simple_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=[self.AUDIO_CROSS_PE_MAX_POS],
                num_attention_heads=self.AUDIO_ATTENTION_HEADS,
                use_middle_indices_grid=True,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                compute_dtype=self.compute_dtype,
            )
            if self.model_type.is_video_enabled():
                self._audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                    simple_preprocessor=audio_simple_preprocessor,
                    cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                    cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                    cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
                    audio_cross_attention_dim=self.audio_inner_dim,
                    av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
                )
            else:
                self._audio_args_preprocessor = audio_simple_preprocessor

    def _process_transformer_blocks(
        self,
        video_args: Optional[TransformerArgs] = None,
        audio_args: Optional[TransformerArgs] = None,
        perturbations: Optional[BatchedPerturbationConfig] = None,
    ) -> Tuple[Optional[TransformerArgs], Optional[TransformerArgs]]:
        """Process transformer blocks."""
        for i, block in enumerate(self.transformer_blocks):
            video_args, audio_args = block(video_args, audio_args, perturbations=perturbations)

            # Reduce eval frequency for performance
            if self._eval_frequency > 0 and (i + 1) % self._eval_frequency == 0:
                if video_args is not None:
                    mx.eval(video_args.x)
                if audio_args is not None:
                    mx.eval(audio_args.x)
        return video_args, audio_args

    def _process_video_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """Process video output."""
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
        """Process audio output."""
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
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
        perturbations: Optional[BatchedPerturbationConfig] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass.

        Args:
            video: Input video modality (required for VideoOnly/AudioVideo).
            audio: Input audio modality (required for AudioVideo/AudioOnly).
            perturbations: Optional perturbation config for STG guidance.
                Supports 4 types: skip_video_self_attn, skip_audio_self_attn,
                skip_a2v_cross_attn, skip_v2a_cross_attn.

        Returns:
            VideoOnly: video_velocity
            AudioOnly: audio_velocity
            AudioVideo: (video_velocity, audio_velocity)
        """
        # --- Type Casting ---
        if self.compute_dtype != mx.float32:
            if video is not None:
                video = Modality(
                    latent=video.latent.astype(self.compute_dtype),
                    context=video.context.astype(self.compute_dtype),
                    context_mask=video.context_mask,
                    timesteps=video.timesteps,
                    positions=video.positions,
                    enabled=video.enabled,
                )
            if audio is not None:
                audio = Modality(
                    latent=audio.latent.astype(self.compute_dtype),
                    context=audio.context.astype(self.compute_dtype),
                    context_mask=audio.context_mask,
                    timesteps=audio.timesteps,
                    positions=audio.positions,
                    enabled=audio.enabled,
                )

        # --- Preprocessing ---
        video_args = None
        if self.model_type.is_video_enabled():
            if video is None:
                raise ValueError("Video modality required for video-enabled model")
            video_args = self._video_args_preprocessor.prepare(video)

        audio_args = None
        if self.model_type.is_audio_enabled():
            if audio is None:
                raise ValueError("Audio modality required for audio-enabled model")
            # Only preprocess if has tokens
            if audio.latent.size > 0:
                audio_args = self._audio_args_preprocessor.prepare(audio)
            else:
                 # Minimal dummy args (should be handled by block enabled check, but safe fallback)
                audio_args = TransformerArgs(
                    x=mx.zeros((video_args.x.shape[0] if video_args else 1, 0, self.audio_inner_dim)),
                    context=mx.zeros((1, 0, self.audio_inner_dim)),
                    timesteps=mx.zeros((1, 0, 6, self.audio_inner_dim)),
                    positional_embeddings=(mx.zeros((1,)), mx.zeros((1,))),
                    enabled=False,
                )

        # --- Transformer Blocks ---
        video_args, audio_args = self._process_transformer_blocks(
            video_args, audio_args, perturbations=perturbations
        )

        # --- Output Processing ---
        video_out = None
        if self.model_type.is_video_enabled():
            video_out = self._process_video_output(video_args.x, video_args.embedded_timestep)
            if self.compute_dtype != mx.float32:
                video_out = video_out.astype(mx.float32)

        audio_out = None
        current_batch_size = video_out.shape[0] if video_out is not None else (audio.latent.shape[0] if audio else 1)
        if self.model_type.is_audio_enabled():
            if audio_args.enabled and audio_args.x.size > 0:
                audio_out = self._process_audio_output(audio_args.x, audio_args.embedded_timestep)
                if self.compute_dtype != mx.float32:
                    audio_out = audio_out.astype(mx.float32)
            else:
                audio_out = mx.zeros((current_batch_size, 0, self.AUDIO_OUT_CHANNELS))

        # --- Return Logic ---
        if self.model_type == LTXModelType.VideoOnly:
            return video_out
        elif self.model_type == LTXModelType.AudioOnly:
            return audio_out
        else:
            return video_out, audio_out


class X0Model(nn.Module):
    """
    Wrapper that returns denoised outputs instead of velocities.
    
    Handles both VideoOnly (returns tensor) and AudioVideo (returns tuple).
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
        perturbations: Optional[BatchedPerturbationConfig] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """
        Compute denoised outputs.

        Args:
            video: Video modality.
            audio: Audio modality.
            perturbations: Optional perturbation config for STG guidance.
        """
        output = self.velocity_model(video, audio, perturbations=perturbations)

        # Helper to denoise
        def denoise(modality, velocity):
            timesteps = modality.timesteps
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None, None]
            elif timesteps.ndim == 2:
                timesteps = timesteps[:, :, None]
            return modality.latent - timesteps * velocity

        if isinstance(output, tuple):
            # AudioVideo case
            video_vel, audio_vel = output
            denoised_video = denoise(video, video_vel)
            denoised_audio = denoise(audio, audio_vel)
            return denoised_video, denoised_audio
        else:
            # VideoOnly or AudioOnly case
            if video is not None:
                return denoise(video, output)
            elif audio is not None:
                return denoise(audio, output)
            return output


# Aliases for backward compatibility
LTXAVModel = LTXModel
X0AVModel = X0Model
