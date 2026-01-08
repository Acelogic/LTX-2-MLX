"""Transformer blocks for LTX-2."""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, rms_norm
from .feed_forward import FeedForward
from .rope import LTXRopeType


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

    def replace(self, **kwargs) -> "TransformerArgs":
        """Return a new TransformerArgs with specified fields replaced."""
        return TransformerArgs(
            x=kwargs.get("x", self.x),
            context=kwargs.get("context", self.context),
            timesteps=kwargs.get("timesteps", self.timesteps),
            positional_embeddings=kwargs.get("positional_embeddings", self.positional_embeddings),
            context_mask=kwargs.get("context_mask", self.context_mask),
            embedded_timestep=kwargs.get("embedded_timestep", self.embedded_timestep),
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
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
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
        """
        super().__init__()

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
        self.scale_shift_table = mx.zeros((6, dim))

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

        # Self-attention with AdaLN
        norm_x = rms_norm(x, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
        x = x + self.attn1(norm_x, pe=args.positional_embeddings) * gate_msa

        # Cross-attention (no AdaLN, just RMSNorm)
        x = x + self.attn2(
            rms_norm(x, eps=self.norm_eps),
            context=args.context,
            mask=args.context_mask,
        )

        # Get AdaLN values for FFN
        shift_mlp, scale_mlp, gate_mlp = self.get_ada_values(
            batch_size, args.timesteps, 3, 6
        )

        # Feed-forward with AdaLN
        x_scaled = rms_norm(x, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
        x = x + self.ff(x_scaled) * gate_mlp

        return args.replace(x=x)


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
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
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
