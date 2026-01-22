"""
Full parity test for Spatial Upscaler - PyTorch vs MLX.
Compares outputs at each layer to identify and fix any discrepancies.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-MLX')
sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-Pytorch/packages/ltx-core/src')

import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


def load_pytorch_upsampler():
    """Load the PyTorch upsampler with weights."""
    from ltx_core.model.upsampler.model import LatentUpsampler

    weights_path = "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"

    model = LatentUpsampler(
        in_channels=128,
        mid_channels=1024,
        num_blocks_per_stage=4,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=True,
    )

    state_dict = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_mlx_upsampler():
    """Load the MLX upsampler with weights."""
    from LTX_2_MLX.model.upscaler.spatial import SpatialUpscaler, load_spatial_upscaler_weights

    weights_path = "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"

    model = SpatialUpscaler(
        in_channels=128,
        mid_channels=1024,
        num_blocks_per_stage=4,
        num_groups=32,
    )
    load_spatial_upscaler_weights(model, weights_path)

    return model


def compare_tensors(name: str, pt_tensor: torch.Tensor, mlx_tensor: mx.array, atol: float = 1e-4) -> bool:
    """Compare PyTorch and MLX tensors and report differences."""
    pt_np = pt_tensor.detach().cpu().numpy()
    mlx_np = np.array(mlx_tensor)

    match = np.allclose(pt_np, mlx_np, atol=atol, rtol=1e-4)

    diff = np.abs(pt_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()

    status = "✓" if match else "✗"
    print(f"  {status} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if not match:
        # Show sample values
        print(f"      PT sample: {pt_np.flatten()[:5]}")
        print(f"      MLX sample: {mlx_np.flatten()[:5]}")
        print(f"      PT stats: mean={pt_np.mean():.6f}, std={pt_np.std():.6f}")
        print(f"      MLX stats: mean={mlx_np.mean():.6f}, std={mlx_np.std():.6f}")

    return match


def test_step_by_step_parity():
    """Test each component step by step."""
    print("=" * 70)
    print("STEP-BY-STEP PARITY TEST")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    pt_model = load_pytorch_upsampler()
    mlx_model = load_mlx_upsampler()

    # Create test input
    np.random.seed(42)
    test_np = np.random.randn(1, 128, 7, 8, 11).astype(np.float32) * 0.5

    pt_input = torch.from_numpy(test_np)
    mlx_input = mx.array(test_np)

    print(f"\nInput shape: {test_np.shape}")

    # ===== Step 1: Initial Conv =====
    print("\n--- Step 1: Initial Conv ---")
    with torch.no_grad():
        pt_conv_out = pt_model.initial_conv(pt_input)

    from LTX_2_MLX.model.upscaler.spatial import conv3d
    mlx_conv_out = conv3d(
        mlx_input,
        mlx_model.initial_conv_weight,
        mlx_model.initial_conv_bias,
        padding=1
    )

    compare_tensors("initial_conv", pt_conv_out, mlx_conv_out)

    # ===== Step 2: Initial Norm =====
    print("\n--- Step 2: Initial Norm ---")
    with torch.no_grad():
        pt_norm_out = pt_model.initial_norm(pt_conv_out)

    from LTX_2_MLX.model.upscaler.spatial import group_norm_5d
    mlx_norm_out = group_norm_5d(
        mlx_conv_out,
        mlx_model.num_groups,
        mlx_model.initial_norm.weight,
        mlx_model.initial_norm.bias
    )

    compare_tensors("initial_norm", pt_norm_out, mlx_norm_out)

    # ===== Step 3: Initial Activation =====
    print("\n--- Step 3: Initial Activation (SiLU) ---")
    with torch.no_grad():
        pt_act_out = pt_model.initial_activation(pt_norm_out)

    mlx_act_out = nn.silu(mlx_norm_out)

    compare_tensors("initial_activation", pt_act_out, mlx_act_out)

    # ===== Step 4: ResBlocks =====
    print("\n--- Step 4: ResBlocks (4 blocks) ---")
    pt_res_out = pt_act_out
    mlx_res_out = mlx_act_out

    for i, (pt_block, mlx_block) in enumerate(zip(pt_model.res_blocks, mlx_model.res_blocks)):
        with torch.no_grad():
            pt_res_out = pt_block(pt_res_out)
        mlx_res_out = mlx_block(mlx_res_out)
        mx.eval(mlx_res_out)

        compare_tensors(f"res_block_{i}", pt_res_out, mlx_res_out)

    # ===== Step 5: Upsampler =====
    print("\n--- Step 5: Upsampler (SpatialRationalResampler) ---")
    with torch.no_grad():
        pt_up_out = pt_model.upsampler(pt_res_out)

    mlx_up_out = mlx_model.upsampler(mlx_res_out)
    mx.eval(mlx_up_out)

    compare_tensors("upsampler", pt_up_out, mlx_up_out)

    # ===== Step 6: Post-upsample ResBlocks =====
    print("\n--- Step 6: Post-upsample ResBlocks (4 blocks) ---")
    pt_post_out = pt_up_out
    mlx_post_out = mlx_up_out

    for i, (pt_block, mlx_block) in enumerate(zip(pt_model.post_upsample_res_blocks, mlx_model.post_upsample_res_blocks)):
        with torch.no_grad():
            pt_post_out = pt_block(pt_post_out)
        mlx_post_out = mlx_block(mlx_post_out)
        mx.eval(mlx_post_out)

        compare_tensors(f"post_res_block_{i}", pt_post_out, mlx_post_out)

    # ===== Step 7: Final Conv =====
    print("\n--- Step 7: Final Conv ---")
    with torch.no_grad():
        pt_final_out = pt_model.final_conv(pt_post_out)

    mlx_final_out = conv3d(
        mlx_post_out,
        mlx_model.final_conv_weight,
        mlx_model.final_conv_bias,
        padding=1
    )

    compare_tensors("final_conv", pt_final_out, mlx_final_out)

    # ===== Full Forward Pass =====
    print("\n--- Full Forward Pass ---")
    with torch.no_grad():
        pt_full_out = pt_model(pt_input)

    mlx_full_out = mlx_model(mlx_input)
    mx.eval(mlx_full_out)

    full_match = compare_tensors("full_forward", pt_full_out, mlx_full_out)

    return full_match


def test_upsampler_component():
    """Test the SpatialRationalResampler in detail."""
    print("\n" + "=" * 70)
    print("UPSAMPLER COMPONENT DETAILED TEST")
    print("=" * 70)

    from ltx_core.model.upsampler.spatial_rational_resampler import SpatialRationalResampler as PT_SRR
    from LTX_2_MLX.model.upscaler.spatial import SpatialRationalResampler as MLX_SRR

    # Load weights
    weights_path = "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"

    with safe_open(weights_path, framework="pt") as f:
        pt_conv_weight = f.get_tensor("upsampler.conv.weight")
        pt_conv_bias = f.get_tensor("upsampler.conv.bias")
        # Handle bfloat16
        if pt_conv_weight.dtype == torch.bfloat16:
            pt_conv_weight = pt_conv_weight.float()
            pt_conv_bias = pt_conv_bias.float()

    # Create PyTorch model
    pt_model = PT_SRR(mid_channels=1024, scale=2.0)
    pt_model.conv.weight.data = pt_conv_weight
    pt_model.conv.bias.data = pt_conv_bias
    pt_model.eval()

    # Create MLX model
    mlx_model = MLX_SRR(mid_channels=1024, scale=2.0)
    # PyTorch Conv2d: (out_C, in_C, kH, kW) -> MLX: (out_C, kH, kW, in_C)
    mlx_model.conv_weight = mx.array(pt_conv_weight.numpy().transpose(0, 2, 3, 1))
    mlx_model.conv_bias = mx.array(pt_conv_bias.numpy())

    # Test input
    np.random.seed(42)
    test_np = np.random.randn(1, 1024, 7, 8, 11).astype(np.float32) * 0.5

    pt_input = torch.from_numpy(test_np)
    mlx_input = mx.array(test_np)

    print(f"\nInput shape: {test_np.shape}")

    # Test step by step
    from einops import rearrange

    # Step 1: Rearrange to 2D
    b, c, f, h, w = pt_input.shape
    pt_2d = rearrange(pt_input, "b c f h w -> (b f) c h w")
    mlx_2d = mlx_input.transpose(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    # MLX needs NHWC
    mlx_2d_nhwc = mx.array(np.array(mlx_2d).transpose(0, 2, 3, 1))

    print(f"\n2D shape - PT: {pt_2d.shape}, MLX NHWC: {mlx_2d_nhwc.shape}")

    # Step 2: Conv2d
    with torch.no_grad():
        pt_conv_out = pt_model.conv(pt_2d)

    mlx_conv_out = mx.conv2d(mlx_2d_nhwc, mlx_model.conv_weight, padding=1)
    mlx_conv_out = mlx_conv_out + mlx_model.conv_bias

    # Compare (convert MLX to NCHW)
    mlx_conv_out_nchw = np.array(mlx_conv_out).transpose(0, 3, 1, 2)

    print("\n--- Conv2d ---")
    compare_tensors("conv2d", pt_conv_out, mx.array(mlx_conv_out_nchw))

    # Step 3: PixelShuffle
    with torch.no_grad():
        pt_ps_out = pt_model.pixel_shuffle(pt_conv_out)

    mlx_ps_out = mlx_model.pixel_shuffle(mlx_conv_out)
    mlx_ps_out_nchw = np.array(mlx_ps_out).transpose(0, 3, 1, 2)

    print("\n--- PixelShuffle ---")
    compare_tensors("pixel_shuffle", pt_ps_out, mx.array(mlx_ps_out_nchw))

    # Step 4: BlurDownsample (stride=1 for 2x scale, should be identity)
    with torch.no_grad():
        pt_blur_out = pt_model.blur_down(pt_ps_out)

    # For stride=1, blur_down returns x unchanged
    mlx_blur_out = mlx_ps_out  # Same as input for stride=1
    mlx_blur_out_nchw = np.array(mlx_blur_out).transpose(0, 3, 1, 2)

    print("\n--- BlurDownsample (stride=1) ---")
    compare_tensors("blur_down", pt_blur_out, mx.array(mlx_blur_out_nchw))

    # Step 5: Rearrange back to 5D
    _, c_out, h_out, w_out = pt_blur_out.shape
    pt_5d = rearrange(pt_blur_out, "(b f) c h w -> b c f h w", b=b, f=f)

    mlx_5d = mlx_blur_out.reshape(b, f, h_out, w_out, c_out).transpose(0, 4, 1, 2, 3)

    print("\n--- Rearrange to 5D ---")
    compare_tensors("rearrange_5d", pt_5d, mlx_5d)

    # Full forward
    print("\n--- Full SpatialRationalResampler ---")
    with torch.no_grad():
        pt_full = pt_model(pt_input)

    mlx_full = mlx_model(mlx_input)

    compare_tensors("full_resampler", pt_full, mlx_full)


def test_with_real_latent():
    """Test with a realistic latent tensor."""
    print("\n" + "=" * 70)
    print("TEST WITH REALISTIC LATENT")
    print("=" * 70)

    pt_model = load_pytorch_upsampler()
    mlx_model = load_mlx_upsampler()

    # Realistic latent: similar stats to actual VAE output
    np.random.seed(123)
    test_np = np.random.randn(1, 128, 7, 16, 22).astype(np.float32)
    # Scale to realistic range
    test_np = test_np * 0.5

    pt_input = torch.from_numpy(test_np)
    mlx_input = mx.array(test_np)

    print(f"\nInput shape: {test_np.shape}")
    print(f"Input stats: mean={test_np.mean():.4f}, std={test_np.std():.4f}")

    with torch.no_grad():
        pt_output = pt_model(pt_input)

    mlx_output = mlx_model(mlx_input)
    mx.eval(mlx_output)

    print(f"\nOutput shape - PT: {pt_output.shape}, MLX: {mlx_output.shape}")

    match = compare_tensors("realistic_latent", pt_output, mlx_output)

    # Compute correlation
    pt_np = pt_output.numpy().flatten()
    mlx_np = np.array(mlx_output).flatten()
    correlation = np.corrcoef(pt_np, mlx_np)[0, 1]
    print(f"\n  Correlation: {correlation:.6f}")

    return match, correlation


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SPATIAL UPSCALER FULL PARITY TEST")
    print("=" * 70)

    # Test 1: Step by step
    step_match = test_step_by_step_parity()

    # Test 2: Upsampler component
    test_upsampler_component()

    # Test 3: Realistic latent
    real_match, correlation = test_with_real_latent()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Step-by-step parity: {'✓ PASS' if step_match else '✗ FAIL'}")
    print(f"Realistic latent parity: {'✓ PASS' if real_match else '✗ FAIL'}")
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.9999:
        print("\n🎉 Excellent parity achieved!")
    elif correlation > 0.999:
        print("\n✓ Good parity - minor numerical differences")
    else:
        print("\n⚠ Parity needs improvement")
