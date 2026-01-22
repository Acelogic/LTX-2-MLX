"""Parity test for Spatial Upscaler between PyTorch and MLX implementations."""

import sys
import numpy as np

# Add both repos to path
sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-MLX')
sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-Pytorch/packages/ltx-core/src')

import torch
import mlx.core as mx
from einops import rearrange


def test_pixel_shuffle_parity():
    """Test PixelShuffleND parity between PyTorch and MLX."""
    print("=" * 60)
    print("Testing PixelShuffle Parity")
    print("=" * 60)

    # Import PyTorch implementation
    from ltx_core.model.upsampler.pixel_shuffle import PixelShuffleND

    # Import MLX implementation
    from LTX_2_MLX.model.upscaler.spatial import PixelShuffle2d

    # Create test input (NCHW format for PyTorch)
    np.random.seed(42)
    test_np = np.random.randn(1, 16, 4, 4).astype(np.float32)  # B, C*4, H, W

    # PyTorch
    pt_input = torch.from_numpy(test_np)
    pt_ps = PixelShuffleND(dims=2, upscale_factors=(2, 2))
    pt_output = pt_ps(pt_input)
    print(f"PyTorch input shape: {pt_input.shape}")
    print(f"PyTorch output shape: {pt_output.shape}")

    # MLX - needs NHWC format
    mlx_input = mx.array(test_np.transpose(0, 2, 3, 1))  # B, H, W, C
    mlx_ps = PixelShuffle2d(upscale_factor=2)
    mlx_output = mlx_ps(mlx_input)
    print(f"MLX input shape (NHWC): {mlx_input.shape}")
    print(f"MLX output shape (NHWC): {mlx_output.shape}")

    # Convert MLX output to NCHW for comparison
    mlx_output_nchw = np.array(mlx_output).transpose(0, 3, 1, 2)
    pt_output_np = pt_output.numpy()

    print(f"\nPyTorch output[0, 0, :4, :4]:\n{pt_output_np[0, 0, :4, :4]}")
    print(f"MLX output[0, 0, :4, :4]:\n{mlx_output_nchw[0, 0, :4, :4]}")

    match = np.allclose(pt_output_np, mlx_output_nchw, atol=1e-5)
    print(f"\n✓ PixelShuffle match: {match}")
    if not match:
        diff = np.abs(pt_output_np - mlx_output_nchw).max()
        print(f"  Max difference: {diff}")

    return match


def test_einops_pixel_shuffle():
    """Understand the einops pattern for PixelShuffle."""
    print("\n" + "=" * 60)
    print("Understanding einops PixelShuffle pattern")
    print("=" * 60)

    # Create a simple test where we can trace values
    # Input: (1, 4, 2, 2) - 4 channels, 2x2 spatial
    # After shuffle: (1, 1, 4, 4) - 1 channel, 4x4 spatial

    # Channel layout in (c p1 p2) order:
    # - c=0: channels 0,1,2,3 (p1=0,1 x p2=0,1)
    # Channel 0: c=0, p1=0, p2=0 -> output position (h*2+0, w*2+0)
    # Channel 1: c=0, p1=0, p2=1 -> output position (h*2+0, w*2+1)
    # Channel 2: c=0, p1=1, p2=0 -> output position (h*2+1, w*2+0)
    # Channel 3: c=0, p1=1, p2=1 -> output position (h*2+1, w*2+1)

    test = torch.arange(16).reshape(1, 4, 2, 2).float()
    print(f"Input shape: {test.shape}")
    print(f"Input:\n{test}")

    # Apply einops pattern
    output = rearrange(test, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")

    # Verify channel 0 at position (0,0) goes to output (0,0)
    # Value at input[0, 0, 0, 0] = 0 -> should be at output[0, 0, 0, 0]
    print(f"\nInput[0,0,0,0] = {test[0,0,0,0].item()} -> Output[0,0,0,0] = {output[0,0,0,0].item()}")
    print(f"Input[0,1,0,0] = {test[0,1,0,0].item()} -> Output[0,0,0,1] = {output[0,0,0,1].item()}")
    print(f"Input[0,2,0,0] = {test[0,2,0,0].item()} -> Output[0,0,1,0] = {output[0,0,1,0].item()}")
    print(f"Input[0,3,0,0] = {test[0,3,0,0].item()} -> Output[0,0,1,1] = {output[0,0,1,1].item()}")


def test_mlx_pixel_shuffle_manual():
    """Manually implement the einops pattern in MLX and verify."""
    print("\n" + "=" * 60)
    print("Testing manual MLX PixelShuffle implementation")
    print("=" * 60)

    # Test input in NCHW: (1, 4, 2, 2)
    test_np = np.arange(16).reshape(1, 4, 2, 2).astype(np.float32)

    # PyTorch reference using einops
    pt_input = torch.from_numpy(test_np)
    pt_output = rearrange(pt_input, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)
    print(f"PyTorch output:\n{pt_output.numpy()[0, 0]}")

    # MLX implementation - NHWC format
    # Input: (B, H, W, C) -> (1, 2, 2, 4)
    mlx_input = mx.array(test_np.transpose(0, 2, 3, 1))  # NCHW -> NHWC
    b, h, w, c = mlx_input.shape
    r = 2
    c_out = c // (r * r)  # 4 // 4 = 1

    # The einops pattern "b (c p1 p2) h w -> b c (h p1) (w p2)" with p1=2, p2=2
    # means: channel index = c * p1 * p2 + p1_idx * p2 + p2_idx
    # So for NHWC (B, H, W, C*p1*p2):
    # We need to reshape (B, H, W, c*p1*p2) -> (B, H, W, c, p1, p2)
    # Then transpose to (B, H, p1, W, p2, c)
    # Then reshape to (B, H*p1, W*p2, c)

    # Reshape: (B, H, W, c*p1*p2) -> (B, H, W, c, p1, p2)
    x = mlx_input.reshape(b, h, w, c_out, r, r)
    print(f"After reshape to (B,H,W,c,p1,p2): {x.shape}")

    # Transpose: (B, H, W, c, p1, p2) -> (B, H, p1, W, p2, c)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    print(f"After transpose to (B,H,p1,W,p2,c): {x.shape}")

    # Reshape: (B, H*p1, W*p2, c)
    x = x.reshape(b, h * r, w * r, c_out)
    print(f"After final reshape: {x.shape}")

    # Convert to NCHW for comparison
    mlx_output_nchw = np.array(x).transpose(0, 3, 1, 2)
    print(f"MLX output:\n{mlx_output_nchw[0, 0]}")

    match = np.allclose(pt_output.numpy(), mlx_output_nchw)
    print(f"\n✓ Manual MLX PixelShuffle matches PyTorch: {match}")

    return match


def test_spatial_resampler_parity():
    """Test SpatialRationalResampler parity."""
    print("\n" + "=" * 60)
    print("Testing SpatialRationalResampler Parity")
    print("=" * 60)

    from ltx_core.model.upsampler.spatial_rational_resampler import SpatialRationalResampler as PT_SRR
    from LTX_2_MLX.model.upscaler.spatial import SpatialRationalResampler as MLX_SRR

    # Create PyTorch model
    pt_model = PT_SRR(mid_channels=4, scale=2.0)

    # Create MLX model with matching weights
    mlx_model = MLX_SRR(mid_channels=4, scale=2.0)

    # Copy weights from PyTorch to MLX
    # PyTorch conv weight: (out_C, in_C, kH, kW)
    pt_conv_weight = pt_model.conv.weight.detach().numpy()
    pt_conv_bias = pt_model.conv.bias.detach().numpy()

    # MLX conv weight: (out_C, kH, kW, in_C)
    mlx_conv_weight = pt_conv_weight.transpose(0, 2, 3, 1)
    mlx_model.conv_weight = mx.array(mlx_conv_weight)
    mlx_model.conv_bias = mx.array(pt_conv_bias)

    # Also copy blur kernel if it exists
    pt_blur_kernel = pt_model.blur_down.kernel.detach().numpy()
    mlx_model.blur_down.kernel = mx.array(pt_blur_kernel)

    # Create test input: (B, C, F, H, W) = (1, 4, 2, 4, 4)
    np.random.seed(42)
    test_np = np.random.randn(1, 4, 2, 4, 4).astype(np.float32)

    # PyTorch forward
    pt_input = torch.from_numpy(test_np)
    with torch.no_grad():
        pt_output = pt_model(pt_input)
    print(f"PyTorch input shape: {pt_input.shape}")
    print(f"PyTorch output shape: {pt_output.shape}")

    # MLX forward (same NCFHW format)
    mlx_input = mx.array(test_np)
    mlx_output = mlx_model(mlx_input)
    print(f"MLX input shape: {mlx_input.shape}")
    print(f"MLX output shape: {mlx_output.shape}")

    pt_output_np = pt_output.numpy()
    mlx_output_np = np.array(mlx_output)

    print(f"\nPyTorch output sample [0,0,0,:4,:4]:\n{pt_output_np[0,0,0,:4,:4]}")
    print(f"MLX output sample [0,0,0,:4,:4]:\n{mlx_output_np[0,0,0,:4,:4]}")

    match = np.allclose(pt_output_np, mlx_output_np, atol=1e-4)
    print(f"\n✓ SpatialRationalResampler match: {match}")
    if not match:
        diff = np.abs(pt_output_np - mlx_output_np)
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean()}")

    return match


def test_full_upsampler_parity():
    """Test full LatentUpsampler parity with loaded weights."""
    print("\n" + "=" * 60)
    print("Testing Full LatentUpsampler Parity (with loaded weights)")
    print("=" * 60)

    from ltx_core.model.upsampler.model import LatentUpsampler as PT_Upsampler
    from LTX_2_MLX.model.upscaler.spatial import SpatialUpscaler as MLX_Upsampler, load_spatial_upscaler_weights
    from safetensors import safe_open

    weights_path = "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"

    # Load weights to check config
    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"Weight keys sample: {keys[:10]}")

        # Check initial_conv weight shape to determine dims and mid_channels
        initial_conv_weight = f.get_tensor("initial_conv.weight")
        print(f"initial_conv.weight shape: {initial_conv_weight.shape}")
        # Shape: (mid_channels, in_channels, kT, kH, kW) for 3D conv
        # or (mid_channels, in_channels, kH, kW) for 2D conv

        if len(initial_conv_weight.shape) == 5:
            dims = 3
            mid_channels = initial_conv_weight.shape[0]
            in_channels = initial_conv_weight.shape[1]
        else:
            dims = 2
            mid_channels = initial_conv_weight.shape[0]
            in_channels = initial_conv_weight.shape[1]

        print(f"Detected: dims={dims}, mid_channels={mid_channels}, in_channels={in_channels}")

    # Create PyTorch model
    pt_model = PT_Upsampler(
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
        dims=dims,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=True,
    )

    # Load PyTorch weights
    state_dict = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    pt_model.load_state_dict(state_dict)
    pt_model.eval()
    print("Loaded PyTorch weights")

    # Create and load MLX model
    mlx_model = MLX_Upsampler(
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
    )
    load_spatial_upscaler_weights(mlx_model, weights_path)
    print("Loaded MLX weights")

    # Create test input: (B, C, F, H, W)
    np.random.seed(42)
    test_np = np.random.randn(1, 128, 3, 4, 4).astype(np.float32) * 0.1  # Small values

    # PyTorch forward
    pt_input = torch.from_numpy(test_np)
    with torch.no_grad():
        pt_output = pt_model(pt_input)
    print(f"\nPyTorch input shape: {pt_input.shape}")
    print(f"PyTorch output shape: {pt_output.shape}")

    # MLX forward
    mlx_input = mx.array(test_np)
    mlx_output = mlx_model(mlx_input)
    mx.eval(mlx_output)
    print(f"MLX input shape: {mlx_input.shape}")
    print(f"MLX output shape: {mlx_output.shape}")

    pt_output_np = pt_output.numpy()
    mlx_output_np = np.array(mlx_output)

    print(f"\nPyTorch output stats: mean={pt_output_np.mean():.6f}, std={pt_output_np.std():.6f}")
    print(f"MLX output stats: mean={mlx_output_np.mean():.6f}, std={mlx_output_np.std():.6f}")

    print(f"\nPyTorch output sample [0,0,0,:4,:4]:\n{pt_output_np[0,0,0,:4,:4]}")
    print(f"MLX output sample [0,0,0,:4,:4]:\n{mlx_output_np[0,0,0,:4,:4]}")

    match = np.allclose(pt_output_np, mlx_output_np, atol=1e-3)
    print(f"\n✓ Full Upsampler match: {match}")
    if not match:
        diff = np.abs(pt_output_np - mlx_output_np)
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean()}")

    return match


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPATIAL UPSCALER PARITY TESTS")
    print("=" * 60)

    # Test 1: Understand einops pattern
    test_einops_pixel_shuffle()

    # Test 2: Manual MLX implementation
    test2_pass = test_mlx_pixel_shuffle_manual()

    # Test 3: PixelShuffle parity
    test3_pass = test_pixel_shuffle_parity()

    # Test 4: SpatialRationalResampler parity
    test4_pass = test_spatial_resampler_parity()

    # Test 5: Full upsampler parity
    test5_pass = test_full_upsampler_parity()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Manual MLX PixelShuffle: {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print(f"PixelShuffle parity: {'✓ PASS' if test3_pass else '✗ FAIL'}")
    print(f"SpatialRationalResampler parity: {'✓ PASS' if test4_pass else '✗ FAIL'}")
    print(f"Full Upsampler parity: {'✓ PASS' if test5_pass else '✗ FAIL'}")
