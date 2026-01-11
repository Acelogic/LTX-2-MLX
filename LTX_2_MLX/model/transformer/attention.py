"""Attention mechanisms for LTX-2 Transformer."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .rope import LTXRopeType, apply_rotary_emb


def rms_norm(x: mx.array, weight: Optional[mx.array] = None, eps: float = 1e-6) -> mx.array:
    """
    Apply RMS normalization.

    Args:
        x: Input tensor.
        weight: Optional learnable scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        RMS normalized tensor.
    """
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    x_normed = x * mx.rsqrt(variance + eps)

    if weight is not None:
        x_normed = x_normed * weight

    return x_normed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array] = None,
    scale: Optional[float] = None,
) -> mx.array:
    """
    Scaled dot-product attention.

    Args:
        q: Query tensor of shape (B, H, T_q, D).
        k: Key tensor of shape (B, H, T_k, D).
        v: Value tensor of shape (B, H, T_k, D).
        mask: Optional attention mask.
        scale: Optional scale factor (default: 1/sqrt(D)).

    Returns:
        Attention output of shape (B, H, T_q, D).
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    # Compute attention scores: (B, H, T_q, T_k)
    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Compute output: (B, H, T_q, D)
    out = mx.matmul(weights, v)

    return out


class Attention(nn.Module):
    """
    Multi-head attention with RMSNorm on Q/K and optional RoPE.

    This attention module follows the LTX-2 architecture:
    - RMSNorm applied to Q and K before attention
    - RoPE applied to Q and K (if position embeddings provided)
    - Standard scaled dot-product attention
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
    ):
        """
        Initialize attention module.

        Args:
            query_dim: Dimension of query input.
            context_dim: Dimension of key/value input (defaults to query_dim).
            heads: Number of attention heads.
            dim_head: Dimension per head.
            norm_eps: Epsilon for RMSNorm.
            rope_type: Type of RoPE to use.
        """
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        context_dim = query_dim if context_dim is None else context_dim

        # RMSNorm for Q and K
        self.q_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = RMSNorm(inner_dim, eps=norm_eps)

        # Linear projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        # Output projection
        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        pe: Optional[tuple] = None,
        k_pe: Optional[tuple] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Query input of shape (B, T, D).
            context: Key/Value input (defaults to x for self-attention).
            mask: Optional attention mask.
            pe: Position embeddings for Q and K (cos, sin tuple).
            k_pe: Separate position embeddings for K (if different from Q).

        Returns:
            Attention output of shape (B, T, D).
        """
        # Project to Q, K, V
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        # Apply RMSNorm to Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if position embeddings provided
        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Reshape for multi-head attention: (B, T, H*D) -> (B, H, T, D)
        b, t_q, _ = q.shape
        _, t_k, _ = k.shape

        q = q.reshape(b, t_q, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, t_k, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t_k, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # Handle mask dimensions
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]  # Add batch and head dims
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]  # Add head dim

        # Compute attention
        out = scaled_dot_product_attention(q, k, v, mask)

        # Reshape back: (B, H, T, D) -> (B, T, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(b, t_q, self.heads * self.dim_head)

        # Output projection
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention layer (convenience wrapper)."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
    ):
        super().__init__()
        self.attn = Attention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        pe: Optional[tuple] = None,
    ) -> mx.array:
        return self.attn(x, context=None, mask=mask, pe=pe)


class CrossAttention(nn.Module):
    """Cross-attention layer (convenience wrapper)."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Cross-attention typically doesn't use RoPE
        self.attn = Attention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        return self.attn(x, context=context, mask=mask, pe=None)
