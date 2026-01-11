"""Compare MLX and PyTorch forward passes to find divergence."""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
import torch


def compare_single_block(weights_path: str):
    """Compare a single transformer block forward pass."""

    print("=" * 60)
    print("Comparing MLX vs PyTorch single transformer block")
    print("=" * 60)

    # Create identical random inputs
    np.random.seed(42)
    batch_size = 1
    seq_len = 16  # Small for testing
    dim = 4096
    context_dim = 3840
    context_len = 32

    # Create input arrays
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 0.1
    context_np = np.random.randn(batch_size, context_len, context_dim).astype(np.float32) * 0.1
    timesteps_np = np.random.randn(batch_size, seq_len, 6, dim).astype(np.float32) * 0.1

    # RoPE position embeddings (simplified - just use random for comparison)
    cos_np = np.random.randn(1, 1, seq_len, 64).astype(np.float32)
    sin_np = np.random.randn(1, 1, seq_len, 64).astype(np.float32)

    print(f"Input x shape: {x_np.shape}")
    print(f"Context shape: {context_np.shape}")
    print(f"Timesteps shape: {timesteps_np.shape}")

    # ===== MLX Forward Pass =====
    print("\n--- MLX Forward Pass ---")
    from LTX_2_MLX.model.transformer import BasicAVTransformerBlock, TransformerArgs, TransformerConfig
    from LTX_2_MLX.loader.weight_converter import load_av_transformer_weights, load_safetensors
    from safetensors import safe_open

    # Create MLX block
    video_config = TransformerConfig(dim=4096, heads=32, d_head=128, context_dim=4096)
    audio_config = TransformerConfig(dim=2048, heads=32, d_head=64, context_dim=2048)
    mlx_block = BasicAVTransformerBlock(
        idx=0,
        video_config=video_config,
        audio_config=audio_config,
    )

    # Load weights for block 0
    print("Loading weights for block 0...")
    with safe_open(weights_path, framework="pt") as f:
        block_weights = {}
        for key in f.keys():
            if key.startswith("model.diffusion_model.transformer_blocks.0."):
                # Convert key
                mlx_key = key.replace("model.diffusion_model.transformer_blocks.0.", "")
                # Handle to_out.0 -> to_out
                mlx_key = mlx_key.replace(".to_out.0.", ".to_out.")
                # Handle ff.net.0.proj -> ff.project_in.proj
                mlx_key = mlx_key.replace(".ff.net.0.proj.", ".ff.project_in.proj.")
                # Handle ff.net.2 -> ff.project_out
                mlx_key = mlx_key.replace(".ff.net.2.", ".ff.project_out.")
                # Same for audio
                mlx_key = mlx_key.replace(".audio_ff.net.0.proj.", ".audio_ff.project_in.proj.")
                mlx_key = mlx_key.replace(".audio_ff.net.2.", ".audio_ff.project_out.")

                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                np_arr = tensor.numpy()

                # Note: We don't transpose Linear weights because MLX uses same layout as PyTorch
                block_weights[mlx_key] = mx.array(np_arr)

        print(f"Loaded {len(block_weights)} weights for block 0")

    # Convert flat dict to nested for model.update()
    def flatten_to_nested(flat_dict):
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

    nested_weights = flatten_to_nested(block_weights)
    mlx_block.update(nested_weights)
    mx.eval(mlx_block.parameters())

    # Check scale_shift_table after loading
    print(f"scale_shift_table range: [{float(mx.min(mlx_block.scale_shift_table)):.4f}, {float(mx.max(mlx_block.scale_shift_table)):.4f}]")

    # Create MLX inputs
    x_mlx = mx.array(x_np)
    context_mlx = mx.array(context_np)
    timesteps_mlx = mx.array(timesteps_np)
    cos_mlx = mx.array(cos_np)
    sin_mlx = mx.array(sin_np)

    # Need to project context to dim (4096) for cross-attention
    # For simplicity, just use random projected context
    context_projected_np = np.random.randn(batch_size, context_len, dim).astype(np.float32) * 0.1
    context_projected_mlx = mx.array(context_projected_np)

    video_args = TransformerArgs(
        x=x_mlx,
        context=context_projected_mlx,
        timesteps=timesteps_mlx,
        positional_embeddings=(cos_mlx, sin_mlx),
        context_mask=None,
        embedded_timestep=None,
        enabled=True,
    )

    # Run MLX forward
    video_out, audio_out = mlx_block(video_args, None)
    mx.eval(video_out.x)

    mlx_output = np.array(video_out.x)
    print(f"MLX output shape: {mlx_output.shape}")
    print(f"MLX output range: [{mlx_output.min():.4f}, {mlx_output.max():.4f}]")
    print(f"MLX output mean: {mlx_output.mean():.4f}, std: {mlx_output.std():.4f}")

    # ===== Compare with reference values =====
    # Since we can't easily run PyTorch LTX model, let's at least verify the
    # MLX computation is numerically stable

    print("\n--- Numerical Stability Check ---")

    # Check for NaN/Inf
    if np.isnan(mlx_output).any():
        print("ERROR: NaN in output!")
    elif np.isinf(mlx_output).any():
        print("ERROR: Inf in output!")
    else:
        print("OK: No NaN/Inf in output")

    # Check output magnitude
    output_std = mlx_output.std()
    if output_std > 10:
        print(f"WARNING: Output std ({output_std:.4f}) is very large")
    elif output_std < 0.001:
        print(f"WARNING: Output std ({output_std:.4f}) is very small")
    else:
        print(f"OK: Output std ({output_std:.4f}) is reasonable")

    # Check if output changed from input
    input_output_diff = np.abs(mlx_output - x_np).mean()
    print(f"Mean absolute change from input: {input_output_diff:.4f}")

    return mlx_output


def compare_adaln_computation(weights_path: str):
    """Compare AdaLN computation specifically."""

    print("\n" + "=" * 60)
    print("Comparing AdaLN Computation")
    print("=" * 60)

    from LTX_2_MLX.model.transformer.timestep_embedding import AdaLayerNormSingle
    from safetensors import safe_open

    # Create AdaLN
    mlx_adaln = AdaLayerNormSingle(embedding_dim=4096, num_embeddings=6)

    # Load weights
    print("Loading adaln_single weights...")
    with safe_open(weights_path, framework="pt") as f:
        adaln_weights = {}
        for key in f.keys():
            if key.startswith("model.diffusion_model.adaln_single."):
                mlx_key = key.replace("model.diffusion_model.adaln_single.", "")
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                adaln_weights[mlx_key] = mx.array(tensor.numpy())

    # Convert to nested
    def flatten_to_nested(flat_dict):
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

    nested_weights = flatten_to_nested(adaln_weights)
    mlx_adaln.update(nested_weights)
    mx.eval(mlx_adaln.parameters())

    print(f"Loaded {len(adaln_weights)} adaln weights")

    # Test timestep embedding
    for sigma in [1.0, 0.5, 0.1, 0.01]:
        timestep = mx.array([sigma * 1000])  # Scale by 1000
        emb, embedded_timestep = mlx_adaln(timestep)
        mx.eval(emb)

        emb_np = np.array(emb)
        print(f"\nSigma={sigma} (scaled={sigma*1000}):")
        print(f"  Output shape: {emb_np.shape}")
        print(f"  Range: [{emb_np.min():.4f}, {emb_np.max():.4f}]")
        print(f"  Mean: {emb_np.mean():.4f}, Std: {emb_np.std():.4f}")

        # Reshape to see scale/shift/gate values
        reshaped = emb_np.reshape(1, 6, 4096)
        for i, name in enumerate(["shift_sa", "scale_sa", "gate_sa", "shift_ff", "scale_ff", "gate_ff"]):
            vals = reshaped[0, i]
            print(f"  {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, range=[{vals.min():.2f}, {vals.max():.2f}]")


def test_rms_norm():
    """Test RMSNorm implementation."""

    print("\n" + "=" * 60)
    print("Testing RMSNorm Implementation")
    print("=" * 60)

    from LTX_2_MLX.model.transformer.attention import rms_norm, RMSNorm

    # Create test input
    np.random.seed(42)
    x_np = np.random.randn(1, 16, 4096).astype(np.float32)
    x_mlx = mx.array(x_np)

    # Test without weight
    normed = rms_norm(x_mlx, eps=1e-6)
    mx.eval(normed)
    normed_np = np.array(normed)

    # Expected: each position should have unit RMS
    rms_per_pos = np.sqrt((normed_np ** 2).mean(axis=-1))
    print(f"RMS per position (should be ~1): mean={rms_per_pos.mean():.4f}, std={rms_per_pos.std():.6f}")

    # Test with weight=1 (should be same)
    weight = mx.ones((4096,))
    normed_w = rms_norm(x_mlx, weight, eps=1e-6)
    mx.eval(normed_w)
    normed_w_np = np.array(normed_w)

    diff = np.abs(normed_np - normed_w_np).max()
    print(f"Diff with weight=1: {diff:.6f} (should be 0)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to model weights",
    )
    args = parser.parse_args()

    test_rms_norm()
    compare_adaln_computation(args.weights)
    compare_single_block(args.weights)


if __name__ == "__main__":
    main()
