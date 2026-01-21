"""
Native MLX implementation of Gemma 3 for LTX-2 text encoding.

This module implements Gemma 3 12B with the ability to extract hidden states
from all layers, which is required by LTX-2's text encoder pipeline.

Architecture:
- 48 transformer layers
- 3840 hidden dimension
- 16 attention heads (query), 8 KV heads (grouped query attention)
- 256 head dimension
- 15360 intermediate (MLP) dimension
- RoPE with linear scaling (factor 8.0)
- RMSNorm with +1 offset
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from LTX_2_MLX.kernels import silu_mul


@dataclass
class Gemma3Config:
    """Configuration for Gemma 3 model."""

    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0  # Gemma3 uses 10^6, not 10^4
    rope_scaling_factor: float = 8.0
    max_position_embeddings: int = 131072
    sliding_window: int = 1024


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """RMSNorm with Gemma-style +1 offset on weight, using optimized mx.fast kernel."""
    # Use optimized kernel for the normalization, then apply Gemma's (1 + weight) scaling
    # mx.fast.rms_norm handles the variance computation efficiently
    normed = mx.fast.rms_norm(x, None, eps)
    # Gemma uses (1 + weight) instead of just weight
    return normed * (1 + weight)


class Gemma3RMSNorm(nn.Module):
    """RMSNorm layer for Gemma 3."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


class Gemma3RotaryEmbedding:
    """Rotary position embeddings for Gemma 3 with linear scaling."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        scaling_factor: float = 8.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, positions: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute cos and sin for rotary embeddings."""
        # Apply linear scaling
        positions = positions.astype(mx.float32) / self.scaling_factor

        # Compute angles: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = positions[:, None] * self.inv_freq[None, :]

        # Create cos and sin
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        return cos, sin


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors."""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim/2]

    # Split into two halves
    q1, q2 = mx.split(q, 2, axis=-1)
    k1, k2 = mx.split(k, 2, axis=-1)

    # Reshape cos/sin for broadcasting: [1, 1, seq_len, head_dim/2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Apply rotation
    q_embed = mx.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
    k_embed = mx.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

    return q_embed, k_embed


class Gemma3Attention(nn.Module):
    """Multi-head attention with grouped query attention for Gemma 3."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Q/K normalization (Gemma 3 specific)
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = Gemma3RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
        )

        # Attention scale
        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, seq, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Q/K normalization (per-head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to [batch, num_heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Get position embeddings
        if position_ids is None:
            position_ids = mx.arange(seq_len)
        elif position_ids.ndim == 2:
            # Handle batch dimension - use first batch item for rotary embeddings
            # This is valid when all batch items have the same position pattern
            # (typical for same-length padded sequences)
            position_ids = position_ids[0]
        cos, sin = self.rotary_emb(position_ids)

        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads for grouped query attention
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        # Compute attention using optimized Flash Attention kernel
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )

        # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        return self.o_proj(attn_output)


class Gemma3MLP(nn.Module):
    """Gated MLP for Gemma 3."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # SiLU gated activation with fused kernel
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))


class Gemma3DecoderLayer(nn.Module):
    """Single transformer layer for Gemma 3."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.self_attn = Gemma3Attention(config)
        self.mlp = Gemma3MLP(config)

        # Gemma 3 has 4 layer norms per block
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3Model(nn.Module):
    """
    Gemma 3 model for text encoding.

    This model outputs hidden states from all layers, which is required
    by LTX-2's text encoder pipeline.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Embedding scale factor (Gemma multiplies embeddings by sqrt(hidden_size))
        self.embed_scale = config.hidden_size ** 0.5

        # Transformer layers
        self.layers = [Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)]

        # Final norm
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: bool = True,
    ) -> Tuple[mx.array, Optional[List[mx.array]]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            position_ids: Position IDs [seq_len].
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Tuple of (last_hidden_state, all_hidden_states).
            all_hidden_states includes embedding layer + all 48 layers = 49 states.
        """
        batch_size, seq_len = input_ids.shape

        # Match HF Gemma behavior: use sequential positions [0, 1, 2, ..., seq_len-1]
        # regardless of attention mask. The attention mask only affects attention
        # computation, not RoPE positions.
        if position_ids is None:
            position_ids = mx.arange(seq_len).astype(mx.int32)

        # Get embeddings and scale by sqrt(hidden_size)
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        # Collect hidden states (PyTorch adds hidden states at START of each layer iteration)
        all_hidden_states = [] if output_hidden_states else None

        # Create causal mask if needed
        if attention_mask is not None:
            binary_attention_mask = attention_mask
            # Convert binary mask to additive mask
            # 1 -> 0, 0 -> -inf
            causal_mask = mx.triu(
                mx.full((seq_len, seq_len), float("-inf")),
                k=1,
            )
            # Add padding mask: 1 -> 0, 0 -> -inf
            # Use mx.where to avoid 0 * -inf = NaN
            padding_mask = mx.where(
                binary_attention_mask[:, None, None, :] == 1,
                mx.zeros((1,)),
                mx.full((1,), float("-inf")),
            )
            # Combine causal and padding masks first
            combined_mask = causal_mask[None, None, :, :] + padding_mask

            # Fix for left-padded sequences: if a query position is padded and would
            # result in all -inf (can't attend to anything), allow it to attend to
            # everything to avoid NaN in softmax. This matches PyTorch behavior.
            # See: https://github.com/pytorch/pytorch/issues/110213
            query_is_pad = binary_attention_mask[:, None, :, None] == 0
            attention_mask = mx.where(query_is_pad, mx.zeros_like(combined_mask), combined_mask)

        # Process through layers
        try:
            from tqdm import tqdm
            layer_iter = tqdm(self.layers, desc="Gemma forward", ncols=80, leave=False)
        except ImportError:
            layer_iter = self.layers

        for layer in layer_iter:
            # Add hidden states BEFORE each layer (matching PyTorch behavior)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask, position_ids)
            mx.eval(hidden_states)  # Force eval for progress tracking

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Add normalized final hidden state (matching PyTorch behavior)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


def load_gemma3_weights(
    model: Gemma3Model,
    weights_dir: str,
    use_fp16: bool = True,
) -> None:
    """
    Load Gemma 3 weights from safetensors files.

    Args:
        model: Gemma3Model instance.
        weights_dir: Directory containing model-0000X-of-00005.safetensors files.
        use_fp16: Whether to use float16 for weights (saves ~50% memory).
    """
    from safetensors import safe_open
    import torch

    weights_path = Path(weights_dir)
    shard_files = sorted(weights_path.glob("model-*.safetensors"))

    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {weights_dir}")

    try:
        from tqdm import tqdm
        shard_iter = tqdm(shard_files, desc="Loading Gemma shards", ncols=80)
    except ImportError:
        shard_iter = shard_files
        print(f"Loading Gemma 3 weights from {len(shard_files)} shards...")

    loaded_count = 0
    target_dtype = mx.float16 if use_fp16 else mx.float32

    for shard_file in shard_iter:
        with safe_open(str(shard_file), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Convert to float32 first (bfloat16 not directly supported)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)

                # Convert to MLX array with target dtype
                value = mx.array(tensor.numpy()).astype(target_dtype)

                # Parse key and set weight
                if key == "language_model.model.embed_tokens.weight":
                    model.embed_tokens.weight = value
                    loaded_count += 1

                elif key == "language_model.model.norm.weight":
                    model.norm.weight = value
                    loaded_count += 1

                elif key.startswith("language_model.model.layers."):
                    # Parse layer index
                    parts = key.split(".")
                    layer_idx = int(parts[3])
                    layer = model.layers[layer_idx]

                    # Route to correct component
                    if "self_attn" in key:
                        attn = layer.self_attn
                        if "q_proj.weight" in key:
                            attn.q_proj.weight = value  # No transpose - MLX does x @ W.T
                        elif "k_proj.weight" in key:
                            attn.k_proj.weight = value
                        elif "v_proj.weight" in key:
                            attn.v_proj.weight = value
                        elif "o_proj.weight" in key:
                            attn.o_proj.weight = value
                        elif "q_norm.weight" in key:
                            attn.q_norm.weight = value
                        elif "k_norm.weight" in key:
                            attn.k_norm.weight = value
                        else:
                            continue

                    elif "mlp" in key:
                        mlp = layer.mlp
                        if "gate_proj.weight" in key:
                            mlp.gate_proj.weight = value  # No transpose
                        elif "up_proj.weight" in key:
                            mlp.up_proj.weight = value
                        elif "down_proj.weight" in key:
                            mlp.down_proj.weight = value
                        else:
                            continue

                    elif "input_layernorm.weight" in key:
                        layer.input_layernorm.weight = value
                    elif "post_attention_layernorm.weight" in key:
                        layer.post_attention_layernorm.weight = value
                    elif "pre_feedforward_layernorm.weight" in key:
                        layer.pre_feedforward_layernorm.weight = value
                    elif "post_feedforward_layernorm.weight" in key:
                        layer.post_feedforward_layernorm.weight = value
                    else:
                        continue

                    loaded_count += 1

    print(f"  Loaded {loaded_count} weight tensors")


def create_gemma3_model(weights_dir: Optional[str] = None) -> Gemma3Model:
    """
    Create and optionally load a Gemma 3 model.

    Args:
        weights_dir: Optional path to weights directory.

    Returns:
        Gemma3Model instance.
    """
    config = Gemma3Config()
    model = Gemma3Model(config)

    if weights_dir:
        load_gemma3_weights(model, weights_dir)

    return model
