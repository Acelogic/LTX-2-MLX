"""Fused operations using custom Metal kernels for LTX-2 MLX.

These kernels combine multiple operations into single GPU passes,
reducing memory bandwidth and kernel launch overhead.
"""

import mlx.core as mx
import numpy as np

# Fused SiLU-Multiply kernel: silu(a) * b
# Used in SwiGLU and gated MLPs
_silu_mul_kernel = mx.fast.metal_kernel(
    name="silu_mul",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
        uint idx = thread_position_in_grid.x;
        if (idx < a_shape[0] * a_shape[1] * a_shape[2]) {
            T val_a = a[idx];
            T val_b = b[idx];
            // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            T sigmoid_a = T(1.0) / (T(1.0) + exp(-val_a));
            out[idx] = val_a * sigmoid_a * val_b;
        }
    """,
)

# Fused GELU-Multiply kernel: gelu_approx(a) * b
# Used in some gated architectures
_gelu_mul_kernel = mx.fast.metal_kernel(
    name="gelu_mul",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
        uint idx = thread_position_in_grid.x;
        if (idx < a_shape[0] * a_shape[1] * a_shape[2]) {
            T val_a = a[idx];
            T val_b = b[idx];
            // GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const T sqrt_2_over_pi = T(0.7978845608028654);
            const T coeff = T(0.044715);
            T inner = sqrt_2_over_pi * (val_a + coeff * val_a * val_a * val_a);
            T gelu_a = T(0.5) * val_a * (T(1.0) + tanh(inner));
            out[idx] = gelu_a * val_b;
        }
    """,
)


def silu_mul(a: mx.array, b: mx.array) -> mx.array:
    """
    Fused SiLU activation with element-wise multiply: silu(a) * b

    This is more efficient than separate silu() and multiply() calls
    as it avoids intermediate memory allocation and extra kernel launch.

    Used in SwiGLU: down(silu(gate(x)) * up(x))

    Args:
        a: Input to SiLU activation
        b: Input to multiply with SiLU output

    Returns:
        silu(a) * b
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Ensure contiguous for kernel
    a = mx.contiguous(a)
    b = mx.contiguous(b)

    # Flatten for simple 1D indexing, will reshape back
    original_shape = a.shape
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    # Pad shape to 3D for kernel (kernel expects 3D shape array)
    n = a_flat.size
    kernel_shape = (n, 1, 1)
    a_3d = a_flat.reshape(kernel_shape)
    b_3d = b_flat.reshape(kernel_shape)

    outputs = _silu_mul_kernel(
        inputs=[a_3d, b_3d],
        template=[("T", a.dtype)],
        output_shapes=[kernel_shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
    )

    return outputs[0].reshape(original_shape)


def gelu_mul(a: mx.array, b: mx.array) -> mx.array:
    """
    Fused GELU activation (tanh approx) with element-wise multiply: gelu(a) * b

    Args:
        a: Input to GELU activation
        b: Input to multiply with GELU output

    Returns:
        gelu_approx(a) * b
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Ensure contiguous for kernel
    a = mx.contiguous(a)
    b = mx.contiguous(b)

    # Flatten for simple 1D indexing
    original_shape = a.shape
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    n = a_flat.size
    kernel_shape = (n, 1, 1)
    a_3d = a_flat.reshape(kernel_shape)
    b_3d = b_flat.reshape(kernel_shape)

    outputs = _gelu_mul_kernel(
        inputs=[a_3d, b_3d],
        template=[("T", a.dtype)],
        output_shapes=[kernel_shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
    )

    return outputs[0].reshape(original_shape)


# Fused Interleaved RoPE kernel
# Combines reshape, rotation, and multiply in one pass
_interleaved_rope_kernel = mx.fast.metal_kernel(
    name="interleaved_rope",
    input_names=["x", "cos_freq", "sin_freq"],
    output_names=["out"],
    source="""
        // x shape: (batch, seq, dim) where dim is even
        // cos/sin shape: (batch, seq, dim) - pre-broadcasted
        // Each pair (x[2i], x[2i+1]) is rotated together

        uint idx = thread_position_in_grid.x;
        uint total = x_shape[0] * x_shape[1] * x_shape[2];

        if (idx < total) {
            // Get position in output
            uint dim = x_shape[2];
            uint seq_dim = x_shape[1] * dim;

            uint batch_idx = idx / seq_dim;
            uint remainder = idx % seq_dim;
            uint seq_idx = remainder / dim;
            uint dim_idx = remainder % dim;

            // Check if this is even or odd index in the pair
            uint pair_idx = dim_idx / 2;
            bool is_even = (dim_idx % 2) == 0;

            // Calculate indices for the pair
            uint even_idx = batch_idx * seq_dim + seq_idx * dim + pair_idx * 2;
            uint odd_idx = even_idx + 1;

            T x_even = x[even_idx];
            T x_odd = x[odd_idx];
            T cos_val = cos_freq[idx];
            T sin_val = sin_freq[idx];

            // Interleaved rotation: out_even = x_even * cos - x_odd * sin
            //                       out_odd = x_odd * cos + x_even * sin
            if (is_even) {
                out[idx] = x_even * cos_val - x_odd * sin_val;
            } else {
                out[idx] = x_odd * cos_val + x_even * sin_val;
            }
        }
    """,
)


def interleaved_rope(
    x: mx.array, cos_freqs: mx.array, sin_freqs: mx.array
) -> mx.array:
    """
    Apply interleaved rotary position embeddings using fused kernel.

    This fuses the reshape, rotation computation, and application into one kernel,
    avoiding intermediate memory allocations.

    Args:
        x: Input tensor of shape (..., dim) where dim is even
        cos_freqs: Cosine frequencies, broadcastable to x shape
        sin_freqs: Sine frequencies, broadcastable to x shape

    Returns:
        Tensor with rotary embeddings applied
    """
    original_shape = x.shape
    original_ndim = x.ndim

    # Flatten to 3D: (batch, seq, dim)
    if x.ndim == 2:
        x = x[None, :, :]
        cos_freqs = cos_freqs[None, :, :] if cos_freqs.ndim == 2 else cos_freqs
        sin_freqs = sin_freqs[None, :, :] if sin_freqs.ndim == 2 else sin_freqs
    elif x.ndim > 3:
        # Flatten batch dimensions
        batch_dims = x.shape[:-2]
        batch_size = int(np.prod(batch_dims))
        x = x.reshape(batch_size, x.shape[-2], x.shape[-1])
        cos_freqs = mx.broadcast_to(cos_freqs, x.shape)
        sin_freqs = mx.broadcast_to(sin_freqs, x.shape)

    # Ensure shapes match
    cos_freqs = mx.broadcast_to(cos_freqs, x.shape)
    sin_freqs = mx.broadcast_to(sin_freqs, x.shape)

    x = mx.contiguous(x)
    cos_freqs = mx.contiguous(cos_freqs)
    sin_freqs = mx.contiguous(sin_freqs)

    n = x.size
    outputs = _interleaved_rope_kernel(
        inputs=[x, cos_freqs, sin_freqs],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
    )

    result = outputs[0]

    # Restore original shape
    if original_ndim == 2:
        result = result[0]
    elif original_ndim > 3:
        result = result.reshape(original_shape)

    return result
