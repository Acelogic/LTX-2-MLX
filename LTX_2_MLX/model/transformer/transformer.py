"""Transformer blocks for LTX-2."""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, rms_norm
from .feed_forward import FeedForward
from .rope import LTXRopeType


# Compiled AdaLN helper - fuses normalization with scale/shift for better performance
@mx.compile
def _compiled_adaln_forward(
    x: mx.array,
    scale: mx.array,
    shift: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """
    Compiled AdaLN forward: RMSNorm + scale + shift.

    Fuses the normalization and modulation into a single compiled graph.
    """
    # RMS normalization
    normed = mx.fast.rms_norm(x, None, eps)
    # Apply adaptive scale and shift
    return normed * (1 + scale) + shift


# Compiled residual + gate - fuses residual connection with gating
@mx.compile
def _compiled_residual_gate(
    x: mx.array,
    residual: mx.array,
    gate: mx.array,
) -> mx.array:
    """
    Compiled residual with gating: x + residual * gate.

    Fuses the gate multiplication and residual addition.
    """
    return x + residual * gate


@dataclass
class TransformerConfig:
    """Configuration for a transformer stream."""

    dim: int
    heads: int
    d_head: int
    context_dim: int


@dataclass
class TransformerArgs:
    """Arguments passed to transformer blocks during forward pass."""

    x: mx.array  # Hidden states
    context: mx.array  # Text context for cross-attention
    timesteps: mx.array  # Timestep embeddings (for AdaLN)
    positional_embeddings: tuple  # RoPE (cos, sin)
    context_mask: Optional[mx.array] = None
    embedded_timestep: Optional[mx.array] = None
    # Cross-modal attention fields (for AudioVideo mode)
    cross_positional_embeddings: Optional[tuple] = None  # RoPE for cross-modal attention
    cross_scale_shift_timestep: Optional[mx.array] = None  # AdaLN for cross-attention scale/shift
    cross_gate_timestep: Optional[mx.array] = None  # AdaLN for cross-attention gate
    enabled: bool = True  # Whether this modality is enabled

    def replace(self, **kwargs) -> "TransformerArgs":
        """Return a new TransformerArgs with specified fields replaced."""
        return TransformerArgs(
            x=kwargs.get("x", self.x),
            context=kwargs.get("context", self.context),
            timesteps=kwargs.get("timesteps", self.timesteps),
            positional_embeddings=kwargs.get("positional_embeddings", self.positional_embeddings),
            context_mask=kwargs.get("context_mask", self.context_mask),
            embedded_timestep=kwargs.get("embedded_timestep", self.embedded_timestep),
            cross_positional_embeddings=kwargs.get("cross_positional_embeddings", self.cross_positional_embeddings),
            cross_scale_shift_timestep=kwargs.get("cross_scale_shift_timestep", self.cross_scale_shift_timestep),
            cross_gate_timestep=kwargs.get("cross_gate_timestep", self.cross_gate_timestep),
            enabled=kwargs.get("enabled", self.enabled),
        )


class BasicTransformerBlock(nn.Module):
    """
    A basic transformer block with self-attention, cross-attention, and feed-forward.

    Uses AdaLN (Adaptive Layer Norm) for timestep conditioning:
    - scale and shift parameters are computed from timestep embeddings
    - applied to normalized hidden states before each sub-layer

    Architecture:
        1. Self-attention with RoPE and AdaLN
        2. Cross-attention to text context
        3. Feed-forward network with AdaLN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        cross_attn_scale: float = 1.0,
    ):
        """
        Initialize transformer block.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            context_dim: Dimension of cross-attention context.
            rope_type: Type of RoPE to use.
            norm_eps: Epsilon for normalization.
            cross_attn_scale: Scaling factor for cross-attention output.
                Higher values increase the influence of text conditioning.
                Default 1.0. Values like 5-10 can help preserve text
                differentiation in late layers.
        """
        super().__init__()
        self.cross_attn_scale = cross_attn_scale

        self.norm_eps = norm_eps

        # Self-attention
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            context_dim=None,  # Self-attention
            rope_type=rope_type,
            norm_eps=norm_eps,
        )

        # Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=head_dim,
            rope_type=rope_type,
            norm_eps=norm_eps,
        )

        # Feed-forward
        self.ff = FeedForward(dim, dim_out=dim)

        # AdaLN scale-shift table: 6 values (scale, shift, gate) x 2 (attn, ff)
        # Note: kept as float32 for numerical stability
        self.scale_shift_table = mx.zeros((6, dim), dtype=mx.float32)

    def get_ada_values(
        self,
        batch_size: int,
        timestep: mx.array,
        start: int,
        end: int,
    ) -> tuple:
        """
        Get adaptive normalization values from timestep embedding.

        Args:
            batch_size: Batch size.
            timestep: Timestep embedding of shape (B, T, 6, D).
            start: Start index in scale_shift_table.
            end: End index in scale_shift_table.

        Returns:
            Tuple of (shift, scale, gate) tensors.
        """
        # scale_shift_table: (6, D)
        # timestep: (B, T, 6, D) where T is the number of tokens
        table_slice = self.scale_shift_table[start:end]  # (num_values, D)

        # Broadcast and add
        # table_slice: (1, 1, num_values, D) + timestep: (B, T, num_values, D)
        ada_values = table_slice[None, None, :, :] + timestep[:, :, start:end, :]

        # Split into individual values
        return tuple(ada_values[:, :, i, :] for i in range(end - start))

    def __call__(self, args: TransformerArgs) -> TransformerArgs:
        """
        Forward pass through transformer block.

        Uses compiled helpers for AdaLN and residual operations for better performance.

        Args:
            args: TransformerArgs containing hidden states and context.

        Returns:
            Updated TransformerArgs with processed hidden states.
        """
        x = args.x
        batch_size = x.shape[0]

        # Get AdaLN values for self-attention
        shift_msa, scale_msa, gate_msa = self.get_ada_values(
            batch_size, args.timesteps, 0, 3
        )

        # Self-attention with AdaLN (using compiled helpers)
        norm_x = _compiled_adaln_forward(x, scale_msa, shift_msa, self.norm_eps)
        attn_out = self.attn1(norm_x, pe=args.positional_embeddings)
        x = _compiled_residual_gate(x, attn_out, gate_msa)

        # Cross-attention (no AdaLN, just RMSNorm)
        # Apply cross_attn_scale to increase text conditioning influence
        cross_out = self.attn2(
            rms_norm(x, eps=self.norm_eps),
            context=args.context,
            mask=args.context_mask,
        )
        x = x + cross_out * self.cross_attn_scale

        # Get AdaLN values for FFN
        shift_mlp, scale_mlp, gate_mlp = self.get_ada_values(
            batch_size, args.timesteps, 3, 6
        )

        # Feed-forward with AdaLN (using compiled helpers)
        x_scaled = _compiled_adaln_forward(x, scale_mlp, shift_mlp, self.norm_eps)
        ff_out = self.ff(x_scaled)
        x = _compiled_residual_gate(x, ff_out, gate_mlp)

        return args.replace(x=x)


class BasicAVTransformerBlock(nn.Module):
    """
    Audio-Video transformer block with cross-modal attention.

    Architecture:
        1. Video: self-attention → cross-attention (text) → cross-attention (audio)
        2. Audio: self-attention → cross-attention (text) → cross-attention (video)
        3. Video: FFN
        4. Audio: FFN

    The cross-modal attention allows audio and video features to inform each other,
    enabling synchronized audio-video generation.
    """

    def __init__(
        self,
        idx: int,
        video_config: Optional[TransformerConfig] = None,
        audio_config: Optional[TransformerConfig] = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
    ):
        """
        Initialize AudioVideo transformer block.

        Args:
            idx: Block index.
            video_config: Configuration for video stream (dim=4096, heads=32, head_dim=128).
            audio_config: Configuration for audio stream (dim=2048, heads=32, head_dim=64).
            rope_type: Type of RoPE to use.
            norm_eps: Epsilon for normalization.
        """
        super().__init__()

        self.idx = idx
        self.norm_eps = norm_eps

        # Video components
        if video_config is not None:
            self.attn1 = Attention(
                query_dim=video_config.dim,
                heads=video_config.heads,
                dim_head=video_config.d_head,
                context_dim=None,  # Self-attention
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.attn2 = Attention(
                query_dim=video_config.dim,
                context_dim=video_config.context_dim,
                heads=video_config.heads,
                dim_head=video_config.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.ff = FeedForward(video_config.dim, dim_out=video_config.dim)
            # Note: kept as float32 for numerical stability
            self.scale_shift_table = mx.zeros((6, video_config.dim), dtype=mx.float32)

        # Audio components
        if audio_config is not None:
            self.audio_attn1 = Attention(
                query_dim=audio_config.dim,
                heads=audio_config.heads,
                dim_head=audio_config.d_head,
                context_dim=None,  # Self-attention
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_attn2 = Attention(
                query_dim=audio_config.dim,
                context_dim=audio_config.context_dim,
                heads=audio_config.heads,
                dim_head=audio_config.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_ff = FeedForward(audio_config.dim, dim_out=audio_config.dim)
            # Note: kept as float32 for numerical stability
            self.audio_scale_shift_table = mx.zeros((6, audio_config.dim), dtype=mx.float32)

        # Cross-modal attention (audio ↔ video)
        if audio_config is not None and video_config is not None:
            # Q: Video, K,V: Audio (audio informs video)
            self.audio_to_video_attn = Attention(
                query_dim=video_config.dim,
                context_dim=audio_config.dim,
                heads=audio_config.heads,  # Use audio heads
                dim_head=audio_config.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Q: Audio, K,V: Video (video informs audio)
            self.video_to_audio_attn = Attention(
                query_dim=audio_config.dim,
                context_dim=video_config.dim,
                heads=audio_config.heads,
                dim_head=audio_config.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Cross-attention AdaLN tables
            # 5 values: scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate
            # Note: kept as float32 for numerical stability
            self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio_config.dim), dtype=mx.float32)
            self.scale_shift_table_a2v_ca_video = mx.zeros((5, video_config.dim), dtype=mx.float32)

    def get_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        timestep: mx.array,
        start: int,
        end: int,
    ) -> tuple:
        """
        Get adaptive normalization values from timestep embedding.

        Args:
            scale_shift_table: Table of learnable parameters (N, D).
            batch_size: Batch size.
            timestep: Timestep embedding (B, T, N, D) or (B, T, D).
            start: Start index.
            end: End index.

        Returns:
            Tuple of adaptive values.
        """
        table_slice = scale_shift_table[start:end]  # (num_values, D)
        ada_values = table_slice[None, None, :, :] + timestep[:, :, start:end, :]
        return tuple(ada_values[:, :, i, :] for i in range(end - start))

    def get_av_ca_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        scale_shift_timestep: mx.array,
        gate_timestep: mx.array,
        num_scale_shift_values: int = 4,
    ) -> tuple:
        """
        Get adaptive values for cross-modal attention.

        Args:
            scale_shift_table: Table of learnable parameters (5, D).
            batch_size: Batch size.
            scale_shift_timestep: Timestep for scale/shift (B, T, 4, D).
            gate_timestep: Timestep for gate (B, T, 1, D).
            num_scale_shift_values: Number of scale/shift values (4).

        Returns:
            Tuple of (scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate).
        """
        # Scale/shift values
        table_slice = scale_shift_table[:num_scale_shift_values]
        scale_shift_ada = table_slice[None, None, :, :] + scale_shift_timestep
        scale_shift_values = tuple(scale_shift_ada[:, :, i, :] for i in range(num_scale_shift_values))

        # Gate value
        gate_table = scale_shift_table[num_scale_shift_values:]
        gate_ada = gate_table[None, None, :, :] + gate_timestep
        gate_values = tuple(gate_ada[:, :, i, :] for i in range(gate_ada.shape[2]))

        return (*scale_shift_values, *gate_values)

    def __call__(
        self,
        video: Optional[TransformerArgs],
        audio: Optional[TransformerArgs],
    ) -> tuple:
        """
        Forward pass through AudioVideo transformer block.

        Args:
            video: Video TransformerArgs (or None if video disabled).
            audio: Audio TransformerArgs (or None if audio disabled).

        Returns:
            Tuple of (updated_video_args, updated_audio_args).
        """
        vx: Optional[mx.array] = video.x if video is not None else None
        ax: Optional[mx.array] = audio.x if audio is not None else None

        run_vx = vx is not None and video is not None and video.enabled and vx.size > 0
        run_ax = ax is not None and audio is not None and audio.enabled and ax.size > 0

        run_a2v = run_vx and ax is not None and ax.size > 0
        run_v2a = run_ax and vx is not None and vx.size > 0

        # Video self-attention + cross-attention to text
        if run_vx:
            assert video is not None and vx is not None  # Type narrowing
            shift_msa, scale_msa, gate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, 0, 3
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
            vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * gate_msa
            vx = vx + self.attn2(
                rms_norm(vx, eps=self.norm_eps),
                context=video.context,
                mask=video.context_mask,
            )

        # Audio self-attention + cross-attention to text
        if run_ax:
            assert audio is not None and ax is not None  # Type narrowing
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, 0, 3
            )
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa
            ax = ax + self.audio_attn2(
                rms_norm(ax, eps=self.norm_eps),
                context=audio.context,
                mask=audio.context_mask,
            )

        # Audio-Video cross-modal attention
        if run_a2v or run_v2a:
            # Both must be valid for cross-modal attention
            assert video is not None and vx is not None
            assert audio is not None and ax is not None

            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            # Get adaptive values for audio side of cross-attention
            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            # Get adaptive values for video side of cross-attention
            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            # Audio to Video attention (audio features inform video)
            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                )

            # Video to Audio attention (video features inform audio)
            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                )

        # Video feed-forward
        if run_vx:
            assert video is not None and vx is not None  # Type narrowing
            shift_mlp, scale_mlp, gate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, 3, 6
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
            vx = vx + self.ff(vx_scaled) * gate_mlp

        # Audio feed-forward
        if run_ax:
            assert audio is not None and ax is not None  # Type narrowing
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, 3, 6
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        # Return updated args
        video_out = video.replace(x=vx) if video is not None else None
        audio_out = audio.replace(x=ax) if audio is not None else None

        return video_out, audio_out


class TransformerBlocks(nn.Module):
    """
    Stack of transformer blocks.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
    ):
        """
        Initialize transformer block stack.

        Args:
            num_layers: Number of transformer blocks.
            dim: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            context_dim: Dimension of cross-attention context.
            rope_type: Type of RoPE to use.
            norm_eps: Epsilon for normalization.
        """
        super().__init__()

        self.blocks = [
            BasicTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                context_dim=context_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, args: TransformerArgs) -> TransformerArgs:
        """Process through all transformer blocks."""
        for block in self.blocks:
            args = block(args)
        return args
