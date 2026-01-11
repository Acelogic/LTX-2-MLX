"""3D Rotary Position Embeddings (RoPE) for LTX-2 Transformer."""

import math
from enum import Enum
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import numpy as np

# Import fused RoPE kernel for interleaved format
try:
    from LTX_2_MLX.kernels import interleaved_rope as _fused_interleaved_rope
    _HAS_FUSED_ROPE = True
except ImportError:
    _HAS_FUSED_ROPE = False
    _fused_interleaved_rope = None


class LTXRopeType(Enum):
    """RoPE implementation variants."""

    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: mx.array,
    freqs_cis: Tuple[mx.array, mx.array],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> mx.array:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        input_tensor: Input tensor to apply RoPE to.
        freqs_cis: Tuple of (cos_freqs, sin_freqs).
        rope_type: Type of RoPE implementation.

    Returns:
        Tensor with rotary embeddings applied.
    """
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """
    Apply interleaved rotary embeddings using fused Metal kernel.

    The interleaved format pairs adjacent dimensions: (d0, d1), (d2, d3), ...
    Rotation is applied to each pair.

    Uses a custom Metal kernel for ~1.2x speedup over the naive implementation.

    Args:
        input_tensor: Input tensor of shape (..., dim).
        cos_freqs: Cosine frequencies.
        sin_freqs: Sine frequencies.

    Returns:
        Tensor with rotary embeddings applied.
    """
    # Use fused kernel if available (1.2x faster)
    if _HAS_FUSED_ROPE and _fused_interleaved_rope is not None:
        return _fused_interleaved_rope(input_tensor, cos_freqs, sin_freqs)

    # Fallback to naive implementation
    shape = input_tensor.shape
    t_dup = input_tensor.reshape(*shape[:-1], shape[-1] // 2, 2)

    # Split into t1, t2
    t1 = t_dup[..., 0]  # Even indices
    t2 = t_dup[..., 1]  # Odd indices

    # Compute rotated version: (-t2, t1)
    t_rot = mx.stack([-t2, t1], axis=-1)
    input_tensor_rot = t_rot.reshape(shape)

    # Apply rotation: x * cos + x_rot * sin
    return input_tensor * cos_freqs + input_tensor_rot * sin_freqs


def apply_split_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """
    Apply split rotary embeddings.

    The split format divides the dimension in half: first_half rotates with second_half.

    Args:
        input_tensor: Input tensor.
        cos_freqs: Cosine frequencies of shape (B, H, T, D//2).
        sin_freqs: Sine frequencies of shape (B, H, T, D//2).

    Returns:
        Tensor with rotary embeddings applied.
    """
    needs_reshape = False
    original_shape = input_tensor.shape

    # Handle dimension mismatch
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        # Reshape from (B, T, H*D) to (B, H, T, D)
        input_tensor = input_tensor.reshape(b, t, h, -1)
        input_tensor = input_tensor.transpose(0, 2, 1, 3)
        needs_reshape = True

    # Split into two halves: (..., dim) -> (..., 2, dim//2)
    dim = input_tensor.shape[-1]
    split_input = input_tensor.reshape(*input_tensor.shape[:-1], 2, dim // 2)

    first_half = split_input[..., 0, :]  # Shape: (..., dim//2)
    second_half = split_input[..., 1, :]  # Shape: (..., dim//2)

    # Apply rotation (functional, no in-place ops)
    # first_half_out = first_half * cos - second_half * sin
    # second_half_out = second_half * cos + first_half * sin
    first_half_out = first_half * cos_freqs - second_half * sin_freqs
    second_half_out = second_half * cos_freqs + first_half * sin_freqs

    # Stack back together
    output = mx.stack([first_half_out, second_half_out], axis=-2)
    output = output.reshape(*output.shape[:-2], dim)

    if needs_reshape:
        b, h, t, d = output.shape
        # Reshape from (B, H, T, D) back to (B, T, H*D)
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(b, t, h * d)

    return output


@lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> mx.array:
    """
    Generate frequency grid using numpy (cached).

    Args:
        positional_embedding_theta: Base theta value.
        positional_embedding_max_pos_count: Maximum position count.
        inner_dim: Inner dimension size.

    Returns:
        Frequency indices array.
    """
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return mx.array(pow_indices * math.pi / 2, dtype=mx.float32)


def generate_freq_grid(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> mx.array:
    """
    Generate frequency grid using MLX.

    Args:
        positional_embedding_theta: Base theta value.
        positional_embedding_max_pos_count: Maximum position count.
        inner_dim: Inner dimension size.

    Returns:
        Frequency indices array.
    """
    theta = positional_embedding_theta
    start = 1.0
    end = float(theta)
    n_elem = 2 * positional_embedding_max_pos_count

    # Generate logarithmically spaced indices
    log_start = math.log(start) / math.log(theta)
    log_end = math.log(end) / math.log(theta)
    num_indices = inner_dim // n_elem

    linspace = mx.linspace(log_start, log_end, num_indices)
    indices = theta ** linspace
    indices = indices * (math.pi / 2)

    return indices.astype(mx.float32)


def get_fractional_positions(
    indices_grid: mx.array,
    max_pos: List[int],
) -> mx.array:
    """
    Convert position indices to fractional positions in [0, 1].

    Args:
        indices_grid: Grid of position indices, shape (B, n_pos_dims, T).
        max_pos: Maximum position for each dimension.

    Returns:
        Fractional positions, shape (B, T, n_pos_dims).
    """
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )

    # Normalize each dimension by its max
    fractional = []
    for i in range(n_pos_dims):
        fractional.append(indices_grid[:, i, :] / max_pos[i])

    # Stack along last dimension: (B, T, n_pos_dims)
    return mx.stack(fractional, axis=-1)


def generate_freqs(
    indices: mx.array,
    indices_grid: mx.array,
    max_pos: List[int],
    use_middle_indices_grid: bool,
) -> mx.array:
    """
    Generate frequencies from position indices.

    Args:
        indices: Frequency indices.
        indices_grid: Position grid, shape (B, n_dims, T) or (B, n_dims, T, 2).
        max_pos: Maximum positions per dimension.
        use_middle_indices_grid: If True, use middle of start/end bounds.

    Returns:
        Frequencies array.
    """
    # Handle indices_grid with bounds (start, end)
    if use_middle_indices_grid:
        assert indices_grid.ndim == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start = indices_grid[..., 0]
        indices_grid_end = indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif indices_grid.ndim == 4:
        indices_grid = indices_grid[..., 0]

    # Get fractional positions: (B, T, n_dims)
    fractional_positions = get_fractional_positions(indices_grid, max_pos)

    # Compute frequencies: scale fractional positions to [-1, 1] range
    # freqs = indices * (fractional * 2 - 1)
    # Shape: (B, T, n_dims, n_freq) -> (B, T, n_dims * n_freq)
    scaled_positions = fractional_positions * 2 - 1  # (B, T, n_dims)
    scaled_positions = scaled_positions[..., None]  # (B, T, n_dims, 1)

    # indices shape: (n_freq,) -> broadcast to (1, 1, 1, n_freq)
    indices = indices[None, None, None, :]

    # freqs shape: (B, T, n_dims, n_freq)
    freqs = indices * scaled_positions

    # Transpose and flatten: (B, T, n_freq, n_dims) -> (B, T, n_freq * n_dims)
    freqs = freqs.transpose(0, 1, 3, 2)
    freqs = freqs.reshape(freqs.shape[0], freqs.shape[1], -1)

    return freqs


def split_freqs_cis(
    freqs: mx.array,
    pad_size: int,
    num_attention_heads: int,
) -> Tuple[mx.array, mx.array]:
    """
    Compute cos/sin frequencies for split RoPE format.

    Args:
        freqs: Frequency array, shape (B, T, freq_dim).
        pad_size: Padding size for dimensions that don't get RoPE.
        num_attention_heads: Number of attention heads.

    Returns:
        Tuple of (cos_freq, sin_freq), each shape (B, H, T, D//2).
    """
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    if pad_size != 0:
        # Pad with 1s for cos and 0s for sin (identity transform)
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape for multi-head attention: (B, T, D) -> (B, H, T, D//H)
    b, t, _ = cos_freq.shape
    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)

    # Transpose to (B, H, T, D//H)
    cos_freq = cos_freq.transpose(0, 2, 1, 3)
    sin_freq = sin_freq.transpose(0, 2, 1, 3)

    return cos_freq, sin_freq


def interleaved_freqs_cis(
    freqs: mx.array,
    pad_size: int,
) -> Tuple[mx.array, mx.array]:
    """
    Compute cos/sin frequencies for interleaved RoPE format.

    Args:
        freqs: Frequency array, shape (B, T, freq_dim).
        pad_size: Padding size.

    Returns:
        Tuple of (cos_freq, sin_freq), each shape (B, T, dim).
    """
    # Compute cos and sin, then repeat each value twice for interleaved format
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    # Repeat interleave: each element appears twice
    # (B, T, D) -> (B, T, 2*D)
    cos_freq = mx.repeat(cos_freq, 2, axis=-1)
    sin_freq = mx.repeat(sin_freq, 2, axis=-1)

    if pad_size != 0:
        # Pad with identity transform
        cos_padding = mx.ones((cos_freq.shape[0], cos_freq.shape[1], pad_size))
        sin_padding = mx.zeros((sin_freq.shape[0], sin_freq.shape[1], pad_size))

        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: mx.array,
    dim: int,
    out_dtype: mx.Dtype = mx.float32,
    theta: float = 10000.0,
    max_pos: Optional[List[int]] = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> Tuple[mx.array, mx.array]:
    """
    Precompute cosine and sine frequencies for RoPE.

    Args:
        indices_grid: Position indices grid, shape (B, n_dims, T) or (B, n_dims, T, 2).
        dim: Dimension of the embedding.
        out_dtype: Output data type.
        theta: Base theta for frequency computation.
        max_pos: Maximum positions per dimension [time, height, width].
        use_middle_indices_grid: If True, use middle of position bounds.
        num_attention_heads: Number of attention heads.
        rope_type: Type of RoPE (INTERLEAVED or SPLIT).

    Returns:
        Tuple of (cos_freqs, sin_freqs).
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]  # Default: [time, height, width]

    # Generate frequency indices
    n_pos_dims = indices_grid.shape[1]
    indices = generate_freq_grid(theta, n_pos_dims, dim)

    # Generate frequencies from positions
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    # Compute cos/sin based on RoPE type
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # Interleaved format
        n_elem = 2 * n_pos_dims
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq.astype(out_dtype), sin_freq.astype(out_dtype)


def create_position_grid(
    batch_size: int,
    frames: int,
    height: int,
    width: int,
) -> mx.array:
    """
    Create a 3D position grid for video tokens.

    Args:
        batch_size: Batch size.
        frames: Number of frames.
        height: Height in latent space.
        width: Width in latent space.

    Returns:
        Position grid of shape (B, 3, T) where T = frames * height * width.
    """
    # Create coordinate grids
    t_coords = mx.arange(frames)
    h_coords = mx.arange(height)
    w_coords = mx.arange(width)

    # Create meshgrid
    t_grid, h_grid, w_grid = mx.meshgrid(t_coords, h_coords, w_coords, indexing="ij")

    # Flatten and stack: (3, F*H*W)
    positions = mx.stack([
        t_grid.flatten(),
        h_grid.flatten(),
        w_grid.flatten(),
    ], axis=0)

    # Expand for batch: (B, 3, T)
    positions = mx.broadcast_to(positions[None, ...], (batch_size, 3, frames * height * width))

    return positions
