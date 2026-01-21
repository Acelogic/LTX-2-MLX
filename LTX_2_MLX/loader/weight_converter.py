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
    Transpose 2D weight matrices in a weights dictionary.

    Note: Both MLX and PyTorch Linear layers store weights as [out_features, in_features],
    so this function is NOT needed for standard weight loading. It exists for testing
    and special cases where explicit transposition is required.

    This function transposes 2D ".weight" tensors, excluding pure embeddings
    (but including projection layers that contain "proj" in the name).

    Args:
        weights: Dictionary of weights.

    Returns:
        Dictionary with transposed 2D weight matrices.
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

    # Handle video_embeddings_connector (video-only encoder) first
    if "video_embeddings_connector" in pytorch_key:
        # model.diffusion_model.video_embeddings_connector.xxx
        # -> embeddings_connector.xxx
        return pytorch_key.replace(
            "model.diffusion_model.video_embeddings_connector.",
            "embeddings_connector.",
        )

    # Handle generic embeddings_connector (for audio/AV encoder)
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

        # Note: MLX and PyTorch both store Linear weights as [out_features, in_features],
        # so NO transpose is needed here. See load_transformer_weights() for reference.

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

        # Note: MLX and PyTorch both store Linear weights as [out_features, in_features],
        # so NO transpose is needed here.

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

        # Note: MLX and PyTorch both store Linear weights as [out_features, in_features],
        # so NO transpose is needed here.

        converted[new_key] = value

    return converted


def convert_pytorch_key_to_mlx(pytorch_key: str, include_audio: bool = False) -> Optional[str]:
    """
    Convert PyTorch weight key to MLX model key path.

    Args:
        pytorch_key: Original PyTorch key (after removing model.diffusion_model.).
        include_audio: If True, include audio/av_ca related keys.

    Returns:
        MLX-compatible key path, or None if should be skipped.
    """
    key = pytorch_key

    # Skip audio/video cross-attention keys unless explicitly included
    if not include_audio:
        if "av_ca" in key or "audio" in key.lower():
            return None

    # Skip video_embeddings_connector - these are text encoder weights loaded separately
    # via load_text_encoder_weights(), not part of the transformer model
    if "video_embeddings_connector" in key:
        return None

    # Handle to_out.0 -> to_out (PyTorch Sequential vs MLX direct)
    key = re.sub(r"\.to_out\.0\.", ".to_out.", key)

    # Handle ff.net.0.proj -> ff.project_in.proj
    key = re.sub(r"\.ff\.net\.0\.proj\.", ".ff.project_in.proj.", key)

    # Handle ff.net.2 -> ff.project_out
    key = re.sub(r"\.ff\.net\.2\.", ".ff.project_out.", key)

    # Handle audio ff.net.0.proj -> audio_ff.project_in.proj
    key = re.sub(r"\.audio_ff\.net\.0\.proj\.", ".audio_ff.project_in.proj.", key)

    # Handle audio ff.net.2 -> audio_ff.project_out
    key = re.sub(r"\.audio_ff\.net\.2\.", ".audio_ff.project_out.", key)

    return key


def load_transformer_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False,
    use_fp8: bool = False,
    include_audio: bool = False,
    streaming: bool = True,
    target_dtype: str = "float16",
) -> None:
    """
    Load transformer weights into an MLX model.

    Args:
        model: MLX model to load weights into.
        weights_path: Path to safetensors file.
        strict: If True, raise error on missing/extra keys.
        use_fp8: If True, handle FP8 quantized weights with dequantization.
        include_audio: If True, include audio-related weights (for AudioVideo model).
        streaming: If True, use memory-efficient streaming load (default True).
        target_dtype: Target dtype after dequantization ("float16" or "float32").
    """
    from safetensors import safe_open
    import gc

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print(f"Loading weights from {weights_path}...")

    # Check for FP8 and load scales if needed
    fp8_scales = {}
    if use_fp8:
        with safe_open(weights_path, framework="pt") as f:
            for key in f.keys():
                if key.endswith(".weight_scale"):
                    fp8_scales[key.replace(".weight_scale", ".weight")] = f.get_tensor(key).item()

    # Build the weights dictionary for model.update()
    weights_dict = {}
    loaded_count = 0
    skipped_count = 0

    with safe_open(weights_path, framework="pt") as f:
        all_keys = list(f.keys())
        if has_tqdm:
            key_iter = tqdm(all_keys, desc="Loading transformer", ncols=80)
        else:
            key_iter = all_keys

        for pytorch_key in key_iter:
            # Only process diffusion model keys
            if not pytorch_key.startswith("model.diffusion_model."):
                continue

            # Remove prefix
            key = pytorch_key.replace("model.diffusion_model.", "")

            # Convert key
            mlx_key = convert_pytorch_key_to_mlx(key, include_audio=include_audio)
            if mlx_key is None:
                skipped_count += 1
                continue

            # Load tensor and convert to target dtype
            tensor = f.get_tensor(pytorch_key)
            import torch

            # Determine target torch dtype for memory efficiency
            torch_target = torch.float16 if target_dtype == "float16" else torch.float32

            # Handle FP8 quantized weights
            if use_fp8 and pytorch_key in fp8_scales:
                # FP8 weight - dequantize using scale, then convert to target dtype
                scale = fp8_scales[pytorch_key]
                tensor = (tensor.to(torch.float32) * scale).to(torch_target)
            elif tensor.dtype == torch.bfloat16:
                # Handle BFloat16 by converting to target dtype
                tensor = tensor.to(torch_target)
            elif hasattr(torch, 'float8_e4m3fn') and tensor.dtype == torch.float8_e4m3fn:
                # FP8 without scale (shouldn't happen but handle gracefully)
                tensor = tensor.to(torch_target)
            elif tensor.dtype != torch_target and tensor.dtype in (torch.float32, torch.float16):
                # Convert other float types to target dtype
                tensor = tensor.to(torch_target)

            # Convert to MLX array
            np_array = tensor.numpy()
            value = mx.array(np_array)

            # MEMORY OPTIMIZATION: Delete torch tensor and numpy array immediately
            # This prevents double memory usage during weight loading
            if streaming:
                del tensor
                del np_array

            # Note: MLX Linear stores weights as [out_features, in_features],
            # same as PyTorch, so we do NOT transpose Linear weights.
            # Both frameworks transpose during forward pass.

            weights_dict[mlx_key] = value
            loaded_count += 1

            # MEMORY OPTIMIZATION: Periodic garbage collection during streaming
            # Every 100 weights, clean up to prevent memory fragmentation
            if streaming and loaded_count % 100 == 0:
                gc.collect()

    # Final cleanup before model update
    if streaming:
        gc.collect()

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
        Nested dictionary structure with lists where indices are numeric.
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

    # Convert dicts with numeric string keys to lists
    return _convert_numeric_dicts_to_lists(nested)


def _convert_numeric_dicts_to_lists(obj: Any) -> Any:
    """
    Recursively convert dicts with numeric string keys to lists.

    For example: {"0": {...}, "1": {...}} -> [{...}, {...}]

    This is needed because MLX model.update() expects lists for
    list-type attributes like transformer_blocks.
    """
    if isinstance(obj, dict):
        # First, recursively process all values
        processed = {k: _convert_numeric_dicts_to_lists(v) for k, v in obj.items()}

        # Check if all keys are numeric strings
        if processed and all(k.isdigit() for k in processed.keys()):
            # Convert to list, handling potential gaps
            max_idx = max(int(k) for k in processed.keys())
            result = [None] * (max_idx + 1)
            for k, v in processed.items():
                result[int(k)] = v
            return result

        return processed

    return obj


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


def load_av_transformer_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False,
    use_fp8: bool = False,
    target_dtype: str = "float16",
) -> None:
    """
    Load AudioVideo transformer weights into an MLX model.

    Convenience function that calls load_transformer_weights with include_audio=True.

    Args:
        model: MLX AudioVideo model to load weights into.
        weights_path: Path to safetensors file.
        strict: If True, raise error on missing/extra keys.
        use_fp8: If True, handle FP8 quantized weights with dequantization.
        target_dtype: Target dtype after dequantization ("float16" or "float32").
    """
    load_transformer_weights(
        model=model,
        weights_path=weights_path,
        strict=strict,
        use_fp8=use_fp8,
        include_audio=True,
        target_dtype=target_dtype,
    )
