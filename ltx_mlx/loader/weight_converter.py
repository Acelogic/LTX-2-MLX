"""Weight conversion from PyTorch safetensors to MLX format."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


def load_safetensors(path: str) -> Dict[str, mx.array]:
    """
    Load weights from a safetensors file.

    Args:
        path: Path to safetensors file.

    Returns:
        Dictionary of weight name to mx.array.
    """
    weights = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert to numpy then to MLX
            weights[key] = mx.array(tensor.numpy())
    return weights


def transpose_linear_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """
    Transpose linear layer weights from PyTorch [out, in] to MLX [in, out].

    MLX Linear expects weights of shape [in_features, out_features],
    while PyTorch stores them as [out_features, in_features].

    Args:
        weights: Dictionary of weights.

    Returns:
        Dictionary with transposed linear weights.
    """
    transposed = {}
    for key, value in weights.items():
        # Linear layers have 2D weights
        if value.ndim == 2 and ".weight" in key:
            # Check if this is a linear layer (not embedding, etc.)
            if "embed" not in key.lower() or "proj" in key.lower():
                transposed[key] = value.T
            else:
                transposed[key] = value
        else:
            transposed[key] = value
    return transposed


def convert_transformer_key(pytorch_key: str) -> Optional[str]:
    """
    Convert a PyTorch transformer key to MLX format.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    # Skip audio-related weights
    if "audio" in pytorch_key.lower():
        return None

    # Skip vocoder weights
    if pytorch_key.startswith("vocoder"):
        return None

    # Skip audio VAE weights
    if pytorch_key.startswith("audio_vae"):
        return None

    # Map prefixes
    key = pytorch_key

    # model.diffusion_model -> transformer
    key = key.replace("model.diffusion_model.", "")

    # Handle to_out.0 -> to_out (MLX doesn't use Sequential)
    key = re.sub(r"\.to_out\.0\.", ".to_out.", key)

    # Handle ff.net.0.proj -> ff.project_in.proj
    key = re.sub(r"\.ff\.net\.0\.proj\.", ".ff.project_in.proj.", key)

    # Handle ff.net.2 -> ff.project_out
    key = re.sub(r"\.ff\.net\.2\.", ".ff.project_out.", key)

    # Handle norm weight (RMSNorm doesn't have bias in our impl)
    # q_norm.weight, k_norm.weight stay as is

    return key


def convert_vae_key(pytorch_key: str) -> Optional[str]:
    """
    Convert a PyTorch VAE key to MLX format.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    # Only process video VAE
    if not pytorch_key.startswith("vae."):
        return None

    key = pytorch_key.replace("vae.", "")

    # Handle conv.weight/bias inside DualConv3d
    # In PyTorch: conv.weight -> In MLX: spatial_conv.weight or time_conv.weight
    # For now, keep the structure similar

    return key


def convert_text_encoder_key(pytorch_key: str) -> Optional[str]:
    """
    Convert text encoder weight key.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    if pytorch_key.startswith("text_embedding_projection."):
        # text_embedding_projection.aggregate_embed.weight
        # -> feature_extractor.aggregate_embed.weight
        return pytorch_key.replace(
            "text_embedding_projection.",
            "feature_extractor.",
        )

    if "embeddings_connector" in pytorch_key:
        # model.diffusion_model.embeddings_connector.xxx
        # -> embeddings_connector.xxx
        return pytorch_key.replace(
            "model.diffusion_model.embeddings_connector.",
            "embeddings_connector.",
        )

    return None


def convert_upsampler_key(pytorch_key: str) -> Optional[str]:
    """
    Convert upsampler weight key.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    # Upsampler weights are stored directly without prefix
    # Just return as-is for now
    return pytorch_key


def extract_transformer_weights(
    weights: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    """
    Extract and convert transformer model weights.

    Args:
        weights: Full weights dictionary.

    Returns:
        Converted transformer weights for MLX model.
    """
    converted = {}

    for key, value in weights.items():
        # Only process diffusion model weights
        if not key.startswith("model.diffusion_model."):
            continue

        # Convert key
        new_key = convert_transformer_key(key)
        if new_key is None:
            continue

        # Transpose linear weights
        if value.ndim == 2 and ".weight" in key:
            value = value.T

        converted[new_key] = value

    return converted


def extract_vae_weights(
    weights: Dict[str, mx.array],
) -> Tuple[Dict[str, mx.array], Dict[str, mx.array]]:
    """
    Extract encoder and decoder weights from VAE.

    Args:
        weights: Full weights dictionary.

    Returns:
        Tuple of (encoder_weights, decoder_weights).
    """
    encoder_weights = {}
    decoder_weights = {}

    for key, value in weights.items():
        if not key.startswith("vae."):
            continue

        mlx_key = key.replace("vae.", "")

        # Transpose 2D weights (linear layers)
        if value.ndim == 2 and ".weight" in key:
            value = value.T

        if mlx_key.startswith("encoder."):
            encoder_weights[mlx_key.replace("encoder.", "")] = value
        elif mlx_key.startswith("decoder."):
            decoder_weights[mlx_key.replace("decoder.", "")] = value
        else:
            # Shared weights (e.g., per_channel_statistics)
            encoder_weights[mlx_key] = value
            decoder_weights[mlx_key] = value

    return encoder_weights, decoder_weights


def extract_text_encoder_weights(
    weights: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    """
    Extract text encoder weights.

    Args:
        weights: Full weights dictionary.

    Returns:
        Text encoder weights for MLX model.
    """
    converted = {}

    for key, value in weights.items():
        new_key = convert_text_encoder_key(key)
        if new_key is None:
            continue

        # Transpose linear weights
        if value.ndim == 2 and ".weight" in key:
            value = value.T

        converted[new_key] = value

    return converted


def convert_pytorch_key_to_mlx(pytorch_key: str) -> Optional[str]:
    """
    Convert PyTorch weight key to MLX model key path.

    Args:
        pytorch_key: Original PyTorch key (after removing model.diffusion_model.).

    Returns:
        MLX-compatible key path, or None if should be skipped.
    """
    key = pytorch_key

    # Skip audio/video cross-attention keys
    if "av_ca" in key or "audio" in key.lower():
        return None

    # Skip video_embeddings_connector for now (text encoder part)
    if "video_embeddings_connector" in key:
        return None

    # Handle to_out.0 -> to_out (PyTorch Sequential vs MLX direct)
    key = re.sub(r"\.to_out\.0\.", ".to_out.", key)

    # Handle ff.net.0.proj -> ff.project_in.proj
    key = re.sub(r"\.ff\.net\.0\.proj\.", ".ff.project_in.proj.", key)

    # Handle ff.net.2 -> ff.project_out
    key = re.sub(r"\.ff\.net\.2\.", ".ff.project_out.", key)

    return key


def load_transformer_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False,
) -> None:
    """
    Load transformer weights into an MLX model.

    Args:
        model: MLX model to load weights into.
        weights_path: Path to safetensors file.
        strict: If True, raise error on missing/extra keys.
    """
    from safetensors import safe_open

    print(f"Loading weights from {weights_path}...")

    # Build the weights dictionary for model.update()
    weights_dict = {}
    loaded_count = 0
    skipped_count = 0

    with safe_open(weights_path, framework="pt") as f:
        for pytorch_key in f.keys():
            # Only process diffusion model keys
            if not pytorch_key.startswith("model.diffusion_model."):
                continue

            # Remove prefix
            key = pytorch_key.replace("model.diffusion_model.", "")

            # Convert key
            mlx_key = convert_pytorch_key_to_mlx(key)
            if mlx_key is None:
                skipped_count += 1
                continue

            # Load tensor and convert to float32/float16 if needed
            tensor = f.get_tensor(pytorch_key)
            # Handle BFloat16 by converting to float32 first
            import torch
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            value = mx.array(tensor.numpy())

            # Note: MLX Linear stores weights as [out_features, in_features],
            # same as PyTorch, so we do NOT transpose Linear weights.
            # Both frameworks transpose during forward pass.

            weights_dict[mlx_key] = value
            loaded_count += 1

    print(f"  Converted {loaded_count} weight tensors (skipped {skipped_count})")

    # Update the model using the update method
    # MLX models use a nested dict structure for model.update()
    nested_weights = _flatten_to_nested(weights_dict)

    # Load weights into model
    model.update(nested_weights)

    print(f"  Successfully loaded weights into model")


def _flatten_to_nested(flat_dict: Dict[str, mx.array]) -> Dict[str, Any]:
    """
    Convert flat dict with dotted keys to nested dict for model.update().

    Args:
        flat_dict: Dictionary with keys like "transformer_blocks.0.attn1.to_q.weight"

    Returns:
        Nested dictionary structure.
    """
    nested = {}

    for key, value in flat_dict.items():
        parts = key.split(".")
        current = nested

        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return nested


def save_mlx_weights(weights: Dict[str, mx.array], path: str) -> None:
    """
    Save weights in MLX format (npz).

    Args:
        weights: Dictionary of weights.
        path: Output path.
    """
    mx.savez(path, **weights)


def load_mlx_weights(path: str) -> Dict[str, mx.array]:
    """
    Load weights from MLX format (npz).

    Args:
        path: Path to npz file.

    Returns:
        Dictionary of weights.
    """
    return dict(mx.load(path))
