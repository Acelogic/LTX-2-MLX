"""LoRA (Low-Rank Adaptation) loading and fusion for LTX-2 models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter."""

    path: str
    strength: float = 1.0

    def __post_init__(self):
        if not -2.0 <= self.strength <= 2.0:
            raise ValueError(f"LoRA strength should be between -2.0 and 2.0, got {self.strength}")


def load_lora_weights(path: str) -> Dict[str, mx.array]:
    """
    Load LoRA weights from a safetensors file.

    Args:
        path: Path to the LoRA weights file.

    Returns:
        Dictionary mapping weight names to arrays.
    """
    from safetensors import safe_open
    import torch

    weights = {}

    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            # Convert to float32 if needed
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            elif tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                tensor = tensor.to(torch.float32)

            weights[key] = mx.array(tensor.numpy())

    return weights


def find_lora_keys_for_weight(
    lora_weights: Dict[str, mx.array],
    base_key: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find LoRA A and B keys corresponding to a base weight key.

    LoRA weights typically follow naming patterns like:
    - base_key.lora_A.weight / base_key.lora_B.weight
    - base_key.lora_down.weight / base_key.lora_up.weight

    Args:
        lora_weights: Dictionary of LoRA weights.
        base_key: The base model weight key (e.g., "transformer.layers.0.attn.to_q.weight").

    Returns:
        Tuple of (lora_A_key, lora_B_key) or (None, None) if not found.
    """
    # Remove .weight suffix if present
    prefix = base_key.replace(".weight", "")

    # Try different LoRA naming conventions
    patterns = [
        (f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"),
        (f"{prefix}.lora_down.weight", f"{prefix}.lora_up.weight"),
        (f"{prefix}.lora_A", f"{prefix}.lora_B"),
        (f"{prefix}.lora_down", f"{prefix}.lora_up"),
    ]

    for key_a, key_b in patterns:
        if key_a in lora_weights and key_b in lora_weights:
            return key_a, key_b

    return None, None


def compute_lora_delta(
    lora_weights: Dict[str, mx.array],
    key_a: str,
    key_b: str,
    strength: float = 1.0,
) -> mx.array:
    """
    Compute the LoRA delta: strength * (lora_B @ lora_A).

    Args:
        lora_weights: Dictionary of LoRA weights.
        key_a: Key for LoRA A weights (down projection).
        key_b: Key for LoRA B weights (up projection).
        strength: Scaling factor for the delta.

    Returns:
        The computed delta to add to base weights.
    """
    lora_a = lora_weights[key_a]
    lora_b = lora_weights[key_b]

    # LoRA computation: delta = B @ A
    # A: (rank, in_features)
    # B: (out_features, rank)
    # Result: (out_features, in_features)
    delta = mx.matmul(lora_b, lora_a) * strength

    return delta


def fuse_lora_into_weights(
    model_weights: Dict[str, mx.array],
    lora_configs: List[LoRAConfig],
    target_dtype: Optional[mx.Dtype] = None,
    verbose: bool = True,
) -> Dict[str, mx.array]:
    """
    Fuse LoRA weights into model weights.

    Computes W_final = W_base + sum(strength_i * (B_i @ A_i)) for each LoRA.

    Args:
        model_weights: Base model weights dictionary.
        lora_configs: List of LoRA configurations to apply.
        target_dtype: Target dtype for output weights. If None, uses base weight dtype.
        verbose: Whether to print progress information.

    Returns:
        New dictionary with LoRA-fused weights.
    """
    # Load all LoRA weights
    all_loras = []
    for config in lora_configs:
        if verbose:
            print(f"Loading LoRA: {config.path} (strength={config.strength})")
        lora_weights = load_lora_weights(config.path)
        all_loras.append((lora_weights, config.strength))

    # Create output dictionary
    fused_weights = {}
    fused_count = 0
    skipped_count = 0

    for key, base_weight in model_weights.items():
        # Determine target dtype
        out_dtype = target_dtype if target_dtype is not None else base_weight.dtype

        # Start with base weight
        fused = base_weight.astype(mx.float32) if base_weight.dtype != mx.float32 else base_weight

        # Try to find and apply LoRA for this key
        applied = False
        for lora_weights, strength in all_loras:
            key_a, key_b = find_lora_keys_for_weight(lora_weights, key)
            if key_a is not None and key_b is not None:
                delta = compute_lora_delta(lora_weights, key_a, key_b, strength)

                # Ensure shapes match
                if delta.shape == fused.shape:
                    fused = fused + delta
                    applied = True
                elif verbose:
                    print(f"  Shape mismatch for {key}: base={fused.shape}, delta={delta.shape}")

        if applied:
            fused_count += 1
        else:
            skipped_count += 1

        # Convert to target dtype
        fused_weights[key] = fused.astype(out_dtype)

    if verbose:
        print(f"Fused LoRA into {fused_count} weights, skipped {skipped_count}")

    return fused_weights


def apply_lora_to_model(
    model,
    lora_configs: List[LoRAConfig],
    verbose: bool = True,
) -> None:
    """
    Apply LoRA weights directly to a model's parameters in-place.

    Args:
        model: MLX model with parameters to modify.
        lora_configs: List of LoRA configurations to apply.
        verbose: Whether to print progress information.
    """
    # Get current model parameters
    params = dict(model.parameters())

    # Flatten nested parameters
    flat_params = {}
    def flatten_params(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_params(v, full_key)
            else:
                flat_params[full_key] = v
    flatten_params(params)

    # Load all LoRAs
    all_loras = []
    for config in lora_configs:
        if verbose:
            print(f"Loading LoRA: {config.path} (strength={config.strength})")
        lora_weights = load_lora_weights(config.path)
        all_loras.append((lora_weights, config.strength))

    # Apply LoRAs to matching parameters
    fused_count = 0
    for key, param in flat_params.items():
        for lora_weights, strength in all_loras:
            key_a, key_b = find_lora_keys_for_weight(lora_weights, key)
            if key_a is not None and key_b is not None:
                delta = compute_lora_delta(lora_weights, key_a, key_b, strength)

                if delta.shape == param.shape:
                    # Update parameter
                    new_value = param + delta.astype(param.dtype)

                    # Navigate to and update the nested parameter
                    parts = key.split(".")
                    obj = model
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], new_value)
                    fused_count += 1

    if verbose:
        print(f"Applied LoRA to {fused_count} parameters")

    # Evaluate to ensure changes are applied
    mx.eval(model.parameters())


# Common LoRA target modules in transformer models
LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "ff.project_in",
    "ff.project_out",
    "attn.query",
    "attn.key",
    "attn.value",
    "attn.output",
]


def get_lora_target_keys(model_weights: Dict[str, mx.array]) -> List[str]:
    """
    Find weight keys that are common LoRA targets.

    Args:
        model_weights: Model weights dictionary.

    Returns:
        List of keys that are typically modified by LoRA.
    """
    target_keys = []
    for key in model_weights.keys():
        for target in LORA_TARGET_MODULES:
            if target in key and key.endswith(".weight"):
                target_keys.append(key)
                break
    return target_keys
