"""Timestep embeddings for LTX-2 Transformer."""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """
    Create sinusoidal timestep embeddings.

    This matches the implementation in Denoising Diffusion Probabilistic Models.

    Args:
        timesteps: 1-D array of N indices (may be fractional).
        embedding_dim: Dimension of the output embeddings.
        flip_sin_to_cos: Whether to use cos, sin order instead of sin, cos.
        downscale_freq_shift: Controls frequency delta between dimensions.
        scale: Scaling factor applied to embeddings.
        max_period: Controls maximum frequency of embeddings.

    Returns:
        Array of shape (N, embedding_dim) with positional embeddings.
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2

    # Compute frequency exponents
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    # Compute embeddings
    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]

    # Scale embeddings
    emb = scale * emb

    # Concatenate sine and cosine embeddings
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    # Flip sine and cosine if requested
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # Zero pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding generator."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """MLP to project timestep embeddings to model dimension."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: Optional[int] = None,
        cond_proj_dim: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=bias)

    def __call__(
        self,
        sample: mx.array,
        condition: Optional[mx.array] = None,
    ) -> mx.array:
        if condition is not None and self.cond_proj is not None:
            sample = sample + self.cond_proj(condition)

        sample = self.linear_1(sample)
        sample = nn.silu(sample)
        sample = self.linear_2(sample)

        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    Combined timestep and size embeddings for PixArt-Alpha style models.

    This is used by LTX-2 for timestep conditioning.
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int = 0,
    ):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def __call__(self, timestep: mx.array) -> mx.array:
        """
        Compute timestep embeddings.

        Args:
            timestep: Timestep values, shape (N,).

        Returns:
            Timestep embeddings, shape (N, embedding_dim).
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class AdaLayerNormSingle(nn.Module):
    """
    Adaptive Layer Norm with scale and shift from timestep embedding.

    Used for conditioning transformer blocks on timestep information.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 6,  # scale, shift for self-attn, cross-attn, ffn
    ):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim=embedding_dim,
            size_emb_dim=0,
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, num_embeddings * embedding_dim, bias=True)

    def __call__(self, timestep: mx.array) -> mx.array:
        """
        Compute AdaLN parameters from timestep.

        Args:
            timestep: Timestep values.

        Returns:
            AdaLN parameters (scale, shift, gate values).
        """
        emb = self.emb(timestep)
        emb = self.silu(emb)
        emb = self.linear(emb)
        return emb
