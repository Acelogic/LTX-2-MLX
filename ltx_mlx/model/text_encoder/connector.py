"""Embeddings connector for LTX-2 text encoder."""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..transformer.attention import Attention, rms_norm
from ..transformer.feed_forward import FeedForward
from ..transformer.rope import LTXRopeType, precompute_freqs_cis


class BasicTransformerBlock1D(nn.Module):
    """
    Simple 1D transformer block for sequence processing.

    Architecture:
    1. RMSNorm -> Self-attention with RoPE
    2. RMSNorm -> Feed-forward

    No cross-attention or AdaLN conditioning.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ):
        """
        Initialize 1D transformer block.

        Args:
            dim: Model dimension.
            heads: Number of attention heads.
            dim_head: Dimension per head.
            rope_type: Type of RoPE.
            norm_eps: Epsilon for normalization.
        """
        super().__init__()
        self.norm_eps = norm_eps

        # Self-attention
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            context_dim=None,  # Self-attention
            rope_type=rope_type,
            norm_eps=norm_eps,
        )

        # Feed-forward
        self.ff = FeedForward(dim, dim_out=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        pe: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor of shape [B, T, D].
            attention_mask: Optional attention mask.
            pe: Optional position embeddings (cos, sin).

        Returns:
            Processed tensor of shape [B, T, D].
        """
        # Handle potential extra dimensions
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Self-attention with residual
        norm_hidden = rms_norm(hidden_states, eps=self.norm_eps)
        attn_output = self.attn1(norm_hidden, mask=attention_mask, pe=pe)
        hidden_states = hidden_states + attn_output

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward with residual
        norm_hidden = rms_norm(hidden_states, eps=self.norm_eps)
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + ff_output

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(nn.Module):
    """
    1D embeddings connector for processing text features.

    Applies a stack of 1D transformer blocks with RoPE to process
    sequential embeddings. Supports learnable registers to replace
    padded positions.

    This connector bridges the Gemma text encoder output to the
    diffusion transformer input format.
    """

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[List[int]] = None,
        num_learnable_registers: Optional[int] = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ):
        """
        Initialize embeddings connector.

        Args:
            attention_head_dim: Dimension per attention head (128).
            num_attention_heads: Number of attention heads (30).
            num_layers: Number of transformer blocks (2).
            positional_embedding_theta: RoPE theta parameter.
            positional_embedding_max_pos: Max positions for RoPE.
            num_learnable_registers: Number of learnable register tokens.
            rope_type: Type of RoPE.
            norm_eps: Epsilon for normalization.
        """
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [1]
        self.rope_type = rope_type
        self.norm_eps = norm_eps

        # Transformer blocks
        self.transformer_1d_blocks = [
            BasicTransformerBlock1D(
                dim=self.inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]

        # Learnable registers (replace padded tokens)
        self.num_learnable_registers = num_learnable_registers
        if num_learnable_registers:
            # Initialize with uniform random in [-1, 1]
            self.learnable_registers = mx.random.uniform(
                low=-1.0,
                high=1.0,
                shape=(num_learnable_registers, self.inner_dim),
            )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Replace padded positions with learnable register tokens.

        This allows the model to process fixed-length sequences while
        maintaining meaningful content in padded positions.

        Args:
            hidden_states: Input tensor [B, T, D].
            attention_mask: Additive attention mask.

        Returns:
            Tuple of (modified_hidden_states, modified_attention_mask).
        """
        seq_len = hidden_states.shape[1]

        if seq_len % self.num_learnable_registers != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by "
                f"num_learnable_registers {self.num_learnable_registers}"
            )

        # Tile registers to match sequence length
        num_duplications = seq_len // self.num_learnable_registers
        tiled_registers = mx.tile(
            self.learnable_registers[None, :, :],
            (hidden_states.shape[0], num_duplications, 1),
        )

        # Create binary mask from attention mask
        # attention_mask is additive: 0 = attend, large negative = don't attend
        mask_squeezed = attention_mask.squeeze(1).squeeze(1)  # [B, T]
        is_valid = (mask_squeezed >= -9000.0)  # [B, T]
        is_valid_expanded = is_valid[:, :, None]  # [B, T, 1]

        # Replace padded positions with learnable registers
        # Where is_valid is False (padded), use registers
        is_padded = ~is_valid_expanded  # [B, T, 1]
        hidden_states = mx.where(
            is_padded,
            tiled_registers,
            hidden_states,
        )

        # Clear the attention mask (all positions now valid)
        new_mask = mx.zeros_like(attention_mask)

        return hidden_states, new_mask

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Process embeddings through connector.

        Args:
            hidden_states: Input tensor [B, T, D].
            attention_mask: Optional additive attention mask.

        Returns:
            Tuple of (processed_hidden_states, attention_mask).
        """
        # Replace padded positions with learnable registers
        if self.num_learnable_registers and attention_mask is not None:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states, attention_mask
            )

        # Create position indices for RoPE: [1, 1, T]
        seq_len = hidden_states.shape[1]
        indices_grid = mx.arange(seq_len, dtype=mx.float32)[None, None, :]

        # Compute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
        )

        # Process through transformer blocks
        for block in self.transformer_1d_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                pe=freqs_cis,
            )

        # Final normalization
        hidden_states = rms_norm(hidden_states, eps=self.norm_eps)

        if attention_mask is None:
            attention_mask = mx.zeros((hidden_states.shape[0], 1, 1, hidden_states.shape[1]))

        return hidden_states, attention_mask
