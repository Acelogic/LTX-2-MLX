"""FP8 weight loading and dequantization for LTX-2 MLX."""

from typing import Dict, Optional, Tuple
import mlx.core as mx

try:
    import torch
    from safetensors import safe_open
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def dequantize_fp8_weight(weight, scale: float) -> mx.array:
    """
    Dequantize FP8 weight to FP32.

    FP8 E4M3 format:
    - 1 sign bit, 4 exponent bits, 3 mantissa bits
    - Range: [-448, 448]
    - Dequantization: fp32_weight = fp8_weight * scale

    Args:
        weight: FP8 tensor (torch.float8_e4m3fn)
        scale: Per-tensor scale factor

    Returns:
        Dequantized MLX array in float32
    """
    # Convert FP8 to FP32 via PyTorch, then to MLX
    fp32_weight = weight.to(torch.float32) * scale
    return mx.array(fp32_weight.numpy())


def is_fp8_checkpoint(weights_path: str) -> bool:
    """
    Check if a checkpoint contains FP8 quantized weights.

    Args:
        weights_path: Path to safetensors file

    Returns:
        True if checkpoint contains FP8 weights with scales
    """
    if not HAS_TORCH:
        return False

    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())
        # FP8 checkpoints have weight_scale keys
        return any("weight_scale" in k for k in keys)


def load_fp8_weights(
    weights_path: str,
    key_filter: Optional[str] = None,
    target_dtype: mx.Dtype = mx.float16,
) -> Tuple[Dict[str, mx.array], int, int]:
    """
    Load FP8 weights with automatic dequantization.

    Handles both FP8 quantized weights (with scales) and regular BF16 weights
    that may be mixed in the same checkpoint.

    Args:
        weights_path: Path to safetensors file
        key_filter: Optional prefix to filter keys (e.g., "model.diffusion_model")
        target_dtype: Target dtype for dequantized weights (default: float16)

    Returns:
        Tuple of (weights_dict, num_fp8_weights, num_regular_weights)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for FP8 weight loading")

    weights = {}
    num_fp8 = 0
    num_regular = 0

    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())

        # Build set of weight keys that have FP8 scales
        fp8_weight_keys = set()
        scale_keys = {}

        for k in keys:
            if k.endswith(".weight_scale"):
                weight_key = k.replace(".weight_scale", ".weight")
                fp8_weight_keys.add(weight_key)
                scale_keys[weight_key] = k

        # Process all keys
        for key in keys:
            # Skip scale keys and input_scale keys
            if key.endswith("_scale"):
                continue

            # Apply filter if specified
            if key_filter and not key.startswith(key_filter):
                continue

            tensor = f.get_tensor(key)

            if key in fp8_weight_keys:
                # FP8 quantized weight - dequantize
                scale = f.get_tensor(scale_keys[key]).item()
                dequantized = dequantize_fp8_weight(tensor, scale)

                # Convert to target dtype
                if target_dtype != mx.float32:
                    dequantized = dequantized.astype(target_dtype)

                weights[key] = dequantized
                num_fp8 += 1
            else:
                # Regular weight (BF16 or FP32)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)

                mlx_tensor = mx.array(tensor.numpy())

                # Convert to target dtype
                if target_dtype != mx.float32:
                    mlx_tensor = mlx_tensor.astype(target_dtype)

                weights[key] = mlx_tensor
                num_regular += 1

    return weights, num_fp8, num_regular


def get_fp8_checkpoint_info(weights_path: str) -> Dict:
    """
    Get information about an FP8 checkpoint.

    Args:
        weights_path: Path to safetensors file

    Returns:
        Dictionary with checkpoint information
    """
    if not HAS_TORCH:
        return {"error": "PyTorch required"}

    info = {
        "path": weights_path,
        "is_fp8": False,
        "num_fp8_weights": 0,
        "num_regular_weights": 0,
        "num_scale_tensors": 0,
        "components": set(),
    }

    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())

        for key in keys:
            # Count scales
            if key.endswith(".weight_scale"):
                info["num_scale_tensors"] += 1
                info["is_fp8"] = True
            elif key.endswith(".input_scale"):
                info["num_scale_tensors"] += 1
            elif not key.endswith("_scale"):
                # Count weights by component
                if key.startswith("model."):
                    info["components"].add("transformer")
                elif key.startswith("vae."):
                    info["components"].add("video_vae")
                elif key.startswith("audio_vae."):
                    info["components"].add("audio_vae")
                elif key.startswith("vocoder."):
                    info["components"].add("vocoder")
                elif key.startswith("text_embedding"):
                    info["components"].add("text_encoder")

                # Check if this weight has a scale
                scale_key = key.replace(".weight", ".weight_scale")
                if key.endswith(".weight") and scale_key in keys:
                    info["num_fp8_weights"] += 1
                else:
                    info["num_regular_weights"] += 1

    info["components"] = list(info["components"])
    return info


if __name__ == "__main__":
    # Test FP8 loading
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "weights/ltx-2/ltx-2-19b-distilled-fp8.safetensors"

    print(f"Checking: {path}")
    info = get_fp8_checkpoint_info(path)

    print(f"\nCheckpoint Info:")
    print(f"  Is FP8: {info['is_fp8']}")
    print(f"  FP8 weights: {info['num_fp8_weights']}")
    print(f"  Regular weights: {info['num_regular_weights']}")
    print(f"  Scale tensors: {info['num_scale_tensors']}")
    print(f"  Components: {info['components']}")
