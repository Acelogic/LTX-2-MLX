"""Gemma feature extractor for LTX-2."""

import mlx.core as mx
import mlx.nn as nn


def norm_and_concat_padded_batch(
    encoded_text: mx.array,
    sequence_lengths: mx.array,
    padding_side: str = "right",
) -> mx.array:
    """
    Normalize and flatten multi-layer hidden states, respecting padding.

    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.

    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".

    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, num_layers = encoded_text.shape
    eps = 1e-6

    # Build mask: [B, T]
    token_indices = mx.arange(t)[None, :]  # [1, T]

    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = t - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    # Expand mask for broadcasting: [B, T, 1, 1]
    mask_expanded = mask[:, :, None, None]

    # Zero out padded positions for mean calculation
    masked = mx.where(mask_expanded, encoded_text, mx.zeros_like(encoded_text))

    # Compute masked mean per batch per layer: sum / (valid_tokens * hidden_dim)
    # Shape: [B, 1, 1, L]
    denom = (sequence_lengths * d).reshape(b, 1, 1, 1)
    mean = masked.sum(axis=(1, 2), keepdims=True) / (denom + eps)

    # Compute masked min/max per batch per layer
    # For min: set padded to +inf, for max: set padded to -inf
    large_val = 1e9
    x_for_min = mx.where(mask_expanded, encoded_text, mx.ones_like(encoded_text) * large_val)
    x_for_max = mx.where(mask_expanded, encoded_text, mx.ones_like(encoded_text) * (-large_val))

    x_min = x_for_min.min(axis=(1, 2), keepdims=True)  # [B, 1, 1, L]
    x_max = x_for_max.max(axis=(1, 2), keepdims=True)  # [B, 1, 1, L]
    range_ = x_max - x_min

    # Normalize: scale to [-4, 4] range (8 * normalized)
    normed = 8 * (encoded_text - mean) / (range_ + eps)

    # Concatenate layers: [B, T, D, L] -> [B, T, D*L]
    normed = normed.reshape(b, t, d * num_layers)

    # Zero out padded positions in final output
    mask_flat = mask[:, :, None]  # [B, T, 1]
    normed = mx.where(mask_flat, normed, mx.zeros_like(normed))

    return normed


class GemmaFeaturesExtractorProjLinear(nn.Module):
    """
    Feature extractor for Gemma hidden states.

    Takes the concatenated hidden states from all Gemma layers (49 layers)
    and projects them to a fixed embedding dimension.

    Input: [batch, seq_len, 3840 * 49] (flattened across layers)
    Output: [batch, seq_len, 3840]
    """

    def __init__(
        self,
        hidden_dim: int = 3840,
        num_layers: int = 49,
    ):
        """
        Initialize feature extractor.

        Args:
            hidden_dim: Hidden dimension per layer (3840 for Gemma).
            num_layers: Number of hidden layers to aggregate (49).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Linear projection: hidden_dim * num_layers -> hidden_dim
        self.aggregate_embed = nn.Linear(
            hidden_dim * num_layers,
            hidden_dim,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Project concatenated hidden states.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim * num_layers].

        Returns:
            Projected tensor of shape [batch, seq_len, hidden_dim].
        """
        return self.aggregate_embed(x)

    def extract_from_hidden_states(
        self,
        hidden_states: list,
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> mx.array:
        """
        Extract features from Gemma hidden states.

        Uses ALL 49 hidden states, applies per-layer normalization, concatenates,
        and projects through the learned linear layer - matching PyTorch exactly.

        Args:
            hidden_states: List of hidden states from each Gemma layer (49 layers).
            attention_mask: Attention mask indicating valid tokens.
            padding_side: Side where padding is applied.

        Returns:
            Projected features of shape [batch, seq_len, hidden_dim].
        """
        # Stack all hidden states: list of [B, T, 3840] -> [B, T, 3840, 49]
        stacked = mx.stack(hidden_states, axis=-1)

        # Get sequence lengths from attention mask
        sequence_lengths = attention_mask.sum(axis=-1).astype(mx.int32)

        # Apply per-layer normalization and concatenation: [B, T, 3840, 49] -> [B, T, 3840*49]
        normed_concat = norm_and_concat_padded_batch(
            stacked, sequence_lengths, padding_side=padding_side
        )

        # Project through linear layer: [B, T, 3840*49] -> [B, T, 3840]
        return self.aggregate_embed(normed_concat)
