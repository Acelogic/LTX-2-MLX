"""Test depth-to-space comparison between MLX and PyTorch."""

import numpy as np

# Test our MLX implementation vs PyTorch
def test_depth_to_space_3d():
    """Test 3D depth-to-space (pixel shuffle) matches PyTorch."""

    # Create test input: (B, C*ft*fh*fw, T, H, W) with factor (2,2,2)
    # So C=4, factor=(2,2,2), packed channels = 4*2*2*2 = 32
    B, T, H, W = 1, 2, 4, 4
    ft, fh, fw = 2, 2, 2
    C = 4
    C_packed = C * ft * fh * fw  # 32

    # Create sequential data so we can trace the shuffle
    x_np = np.arange(B * C_packed * T * H * W).reshape(B, C_packed, T, H, W).astype(np.float32)

    print(f"Input shape: {x_np.shape}")
    print(f"Factor: ({ft}, {fh}, {fw})")
    print(f"Expected output shape: ({B}, {C}, {T*ft}, {H*fh}, {W*fw})")

    # PyTorch reference implementation
    import torch
    x_pt = torch.from_numpy(x_np)

    # PyTorch's depth_to_space for 3D (custom implementation matching their convention)
    def pytorch_depth_to_space_3d(x, factor):
        ft, fh, fw = factor
        b, c_packed, t, h, w = x.shape
        c = c_packed // (ft * fh * fw)

        # PyTorch convention: reshape then permute
        x = x.view(b, c, ft, fh, fw, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)  # (B, C, T, ft, H, fh, W, fw)
        x = x.contiguous().view(b, c, t * ft, h * fh, w * fw)
        return x

    y_pt = pytorch_depth_to_space_3d(x_pt, (ft, fh, fw))
    print(f"PyTorch output shape: {y_pt.shape}")

    # Our MLX implementation
    import mlx.core as mx
    x_mlx = mx.array(x_np)

    def mlx_depth_to_space_3d(x, factor):
        ft, fh, fw = factor
        b, c_packed, t, h, w = x.shape
        c = c_packed // (ft * fh * fw)

        # Current implementation
        x = x.reshape(b, c, ft, fh, fw, t, h, w)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (B, C, T, ft, H, fh, W, fw)
        x = x.reshape(b, c, t * ft, h * fh, w * fw)
        return x

    y_mlx = mlx_depth_to_space_3d(x_mlx, (ft, fh, fw))
    print(f"MLX output shape: {y_mlx.shape}")

    # Compare
    y_pt_np = y_pt.numpy()
    y_mlx_np = np.array(y_mlx)

    if np.allclose(y_pt_np, y_mlx_np):
        print("✓ DepthToSpace3D matches PyTorch!")
    else:
        print("✗ DepthToSpace3D DIFFERS from PyTorch!")
        # Find first difference
        diff = np.abs(y_pt_np - y_mlx_np)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Max diff at {max_diff_idx}: PyTorch={y_pt_np[max_diff_idx]}, MLX={y_mlx_np[max_diff_idx]}")

        # Print a slice to visualize
        print("\nPyTorch output [0,0,0,:4,:4]:")
        print(y_pt_np[0, 0, 0, :4, :4])
        print("\nMLX output [0,0,0,:4,:4]:")
        print(y_mlx_np[0, 0, 0, :4, :4])
        return False

    return True


def test_conv3d_simple():
    """Test our Conv3d implementation vs PyTorch."""
    import torch
    import torch.nn as nn
    import mlx.core as mx

    # Create a simple test case
    B, C_in, T, H, W = 1, 4, 3, 8, 8
    C_out = 8
    kernel_size = 3

    # Create input
    x_np = np.random.randn(B, C_in, T, H, W).astype(np.float32)

    # PyTorch Conv3d
    pt_conv = nn.Conv3d(C_in, C_out, kernel_size, padding=1, bias=True)
    with torch.no_grad():
        weight_np = pt_conv.weight.numpy()
        bias_np = pt_conv.bias.numpy()

    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        y_pt = pt_conv(x_pt)

    print(f"\nConv3d test:")
    print(f"  Input: {x_np.shape}")
    print(f"  Weight: {weight_np.shape}")
    print(f"  PyTorch output: {y_pt.shape}")

    # Our MLX Conv3d implementation
    import sys
    sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-MLX')
    from LTX_2_MLX.model.video_vae.simple_decoder import Conv3dSimple

    mlx_conv = Conv3dSimple(C_in, C_out, kernel_size=kernel_size, padding=1)
    mlx_conv.weight = mx.array(weight_np)
    mlx_conv.bias = mx.array(bias_np)

    x_mlx = mx.array(x_np)
    y_mlx = mlx_conv(x_mlx, causal=False)  # Non-causal to match PyTorch's symmetric padding

    print(f"  MLX output: {y_mlx.shape}")

    # Compare
    y_pt_np = y_pt.detach().numpy()
    y_mlx_np = np.array(y_mlx)

    max_diff = np.max(np.abs(y_pt_np - y_mlx_np))
    mean_diff = np.mean(np.abs(y_pt_np - y_mlx_np))

    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("✓ Conv3d matches PyTorch!")
        return True
    else:
        print("✗ Conv3d DIFFERS from PyTorch!")
        # Show sample outputs
        print(f"\nPyTorch output [0,0,1,3:6,3:6]:\n{y_pt_np[0,0,1,3:6,3:6]}")
        print(f"\nMLX output [0,0,1,3:6,3:6]:\n{y_mlx_np[0,0,1,3:6,3:6]}")
        return False


def test_full_upsample_block():
    """Test full DepthToSpaceUpsample3d block."""
    import torch
    import torch.nn as nn
    import mlx.core as mx

    print("\n=== Testing Full Upsample Block ===")

    B, C_in, T, H, W = 1, 8, 2, 4, 4
    factor = (2, 2, 2)
    ft, fh, fw = factor
    C_out_conv = C_in * ft * fh * fw  # 64
    C_out = C_in  # After depth-to-space: 8

    x_np = np.random.randn(B, C_in, T, H, W).astype(np.float32)

    # PyTorch version
    pt_conv = nn.Conv3d(C_in, C_out_conv, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        weight_np = pt_conv.weight.numpy()
        bias_np = pt_conv.bias.numpy()

    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        conv_out_pt = pt_conv(x_pt)

    # PyTorch depth-to-space
    def pytorch_depth_to_space_3d(x, factor):
        ft, fh, fw = factor
        b, c_packed, t, h, w = x.shape
        c = c_packed // (ft * fh * fw)
        x = x.view(b, c, ft, fh, fw, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.contiguous().view(b, c, t * ft, h * fh, w * fw)
        return x

    y_pt = pytorch_depth_to_space_3d(conv_out_pt, factor)
    print(f"PyTorch upsample output shape: {y_pt.shape}")

    # MLX version
    import sys
    sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-MLX')
    from LTX_2_MLX.model.video_vae.simple_decoder import DepthToSpaceUpsample3d

    mlx_block = DepthToSpaceUpsample3d(C_in, factor=factor)
    mlx_block.conv.weight = mx.array(weight_np)
    mlx_block.conv.bias = mx.array(bias_np)

    x_mlx = mx.array(x_np)
    y_mlx = mlx_block(x_mlx, causal=False)

    print(f"MLX upsample output shape: {y_mlx.shape}")

    # Compare
    y_pt_np = y_pt.detach().numpy()
    y_mlx_np = np.array(y_mlx)

    max_diff = np.max(np.abs(y_pt_np - y_mlx_np))
    mean_diff = np.mean(np.abs(y_pt_np - y_mlx_np))

    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Full upsample block matches PyTorch!")
        return True
    else:
        print("✗ Full upsample block DIFFERS from PyTorch!")
        return False


def test_unpatchify_exact():
    """Test unpatchify with exact decoder parameters."""
    import torch
    import torch.nn.functional as F
    import mlx.core as mx
    import sys
    sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-MLX')
    from LTX_2_MLX.model.video_vae.ops import unpatchify

    print("\n=== Testing Unpatchify (patch_size_hw=4, patch_size_t=1) ===")

    # Exact parameters from decoder: (B, 48, T, H, W) -> (B, 3, T, H*4, W*4)
    B, T, H, W = 1, 4, 16, 16
    patch_size_hw = 4
    patch_size_t = 1
    C_packed = 3 * patch_size_t * patch_size_hw * patch_size_hw  # 48
    C = 3

    # Create sequential data
    x_np = np.arange(B * C_packed * T * H * W).reshape(B, C_packed, T, H, W).astype(np.float32)

    print(f"Input shape: {x_np.shape}")
    print(f"Expected output: ({B}, {C}, {T}, {H*4}, {W*4})")

    # PyTorch pixel_shuffle (only spatial, since patch_size_t=1)
    x_pt = torch.from_numpy(x_np)
    # Reshape for 2D pixel_shuffle: (B, C*16, T, H, W) -> (B*T, C*16, H, W)
    x_pt_2d = x_pt.permute(0, 2, 1, 3, 4).reshape(B * T, C_packed, H, W)
    y_pt_2d = F.pixel_shuffle(x_pt_2d, patch_size_hw)  # (B*T, C, H*4, W*4)
    y_pt = y_pt_2d.reshape(B, T, C, H * 4, W * 4).permute(0, 2, 1, 3, 4)  # (B, C, T, H*4, W*4)

    print(f"PyTorch output shape: {y_pt.shape}")

    # MLX unpatchify
    x_mlx = mx.array(x_np)
    y_mlx = unpatchify(x_mlx, patch_size_hw=patch_size_hw, patch_size_t=patch_size_t)
    mx.eval(y_mlx)

    print(f"MLX output shape: {y_mlx.shape}")

    # Compare
    y_pt_np = y_pt.numpy()
    y_mlx_np = np.array(y_mlx)

    if np.allclose(y_pt_np, y_mlx_np):
        print("✓ Unpatchify matches PyTorch pixel_shuffle!")
    else:
        print("✗ Unpatchify DIFFERS from PyTorch!")
        diff = np.abs(y_pt_np - y_mlx_np)
        print(f"Max diff: {np.max(diff):.6f}")
        print(f"Mean diff: {np.mean(diff):.6f}")

        # Show first difference
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"First big diff at {max_idx}: PT={y_pt_np[max_idx]}, MLX={y_mlx_np[max_idx]}")

        # Show sample 4x4 patches
        print("\nPyTorch output [0,0,0,:8,:8]:")
        print(y_pt_np[0, 0, 0, :8, :8])
        print("\nMLX output [0,0,0,:8,:8]:")
        print(y_mlx_np[0, 0, 0, :8, :8])
        return False

    return True


def test_video_output_conversion():
    """Test the final video tensor to uint8 conversion."""
    import mlx.core as mx

    print("\n=== Testing Video Output Conversion ===")

    # Simulate decoder output: values in [-1, 1]
    # Create gradient pattern
    B, C, T, H, W = 1, 3, 4, 64, 64
    video = np.linspace(-1, 1, B * C * T * H * W).reshape(B, C, T, H, W).astype(np.float32)

    video_mlx = mx.array(video)

    # Our conversion code
    video_out = mx.clip((video_mlx + 1) / 2, 0, 1) * 255
    video_out = video_out.astype(mx.uint8)
    # Rearrange: (B, C, T, H, W) -> (T, H, W, C)
    video_out = video_out[0]  # Remove batch
    video_out = video_out.transpose(1, 2, 3, 0)

    print(f"Output shape: {video_out.shape}")
    print(f"Output dtype: {video_out.dtype}")
    print(f"Output range: [{np.array(video_out).min()}, {np.array(video_out).max()}]")

    # Check for banding/quantization issues
    unique_vals = len(np.unique(np.array(video_out)))
    print(f"Unique values: {unique_vals}")

    if video_out.shape == (T, H, W, C):
        print("✓ Output shape correct!")
    else:
        print("✗ Output shape wrong!")
        return False

    return True


def debug_decoder_pipeline():
    """Debug by checking intermediate outputs at each stage."""
    import mlx.core as mx
    import sys
    sys.path.insert(0, '/Users/mcruz/Developer/LTX-2-MLX')

    print("\n=== Debugging Decoder Pipeline ===")

    # Load latent
    latent_path = "/tmp/test_video_latent.npz"
    try:
        data = np.load(latent_path)
        latent = data['latent']
        print(f"Loaded latent: {latent.shape}")
    except FileNotFoundError:
        print("No saved latent found, creating dummy")
        latent = np.random.randn(1, 128, 5, 16, 16).astype(np.float32)

    from LTX_2_MLX.model.video_vae.simple_decoder import SimpleVideoDecoder, load_vae_decoder_weights

    # Load decoder
    decoder = SimpleVideoDecoder()
    weights_path = "/Users/mcruz/Developer/LTX-2-MLX/weights/ltx-2/ltx-2-19b-dev.safetensors"

    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return

    load_vae_decoder_weights(decoder, weights_path)

    latent_mlx = mx.array(latent)

    # Run through decoder with intermediate checkpoints
    x = latent_mlx

    # Denormalize
    x = x * decoder.std_of_means[None, :, None, None, None]
    x = x + decoder.mean_of_means[None, :, None, None, None]
    mx.eval(x)
    print(f"After denorm: range=[{np.array(x).min():.3f}, {np.array(x).max():.3f}]")

    # Conv in
    x = decoder.conv_in(x, causal=True)
    mx.eval(x)
    print(f"After conv_in: shape={x.shape}, range=[{np.array(x).min():.3f}, {np.array(x).max():.3f}]")

    # First res block group
    x = decoder.up_blocks_0(x, causal=True)
    mx.eval(x)
    print(f"After up_blocks_0: shape={x.shape}, range=[{np.array(x).min():.3f}, {np.array(x).max():.3f}]")

    # First upsample
    x = decoder.up_blocks_1(x, causal=True)
    mx.eval(x)
    print(f"After up_blocks_1 (upsample): shape={x.shape}, range=[{np.array(x).min():.3f}, {np.array(x).max():.3f}]")

    # Check for NaN/Inf
    x_np = np.array(x)
    if np.isnan(x_np).any():
        print("  WARNING: NaN detected!")
    if np.isinf(x_np).any():
        print("  WARNING: Inf detected!")

    # Continue through rest of decoder...
    x = decoder.up_blocks_2(x, causal=True)
    mx.eval(x)
    print(f"After up_blocks_2: shape={x.shape}")

    x = decoder.up_blocks_3(x, causal=True)
    mx.eval(x)
    print(f"After up_blocks_3 (upsample): shape={x.shape}")

    x = decoder.up_blocks_4(x, causal=True)
    mx.eval(x)
    print(f"After up_blocks_4: shape={x.shape}")

    x = decoder.up_blocks_5(x, causal=True)
    mx.eval(x)
    print(f"After up_blocks_5 (upsample): shape={x.shape}")

    x = decoder.up_blocks_6(x, causal=True)
    mx.eval(x)
    print(f"After up_blocks_6: shape={x.shape}")

    # Final norm
    from LTX_2_MLX.model.video_vae.simple_decoder import _pixel_norm
    x = _pixel_norm(x)
    scale = 1 + decoder.last_scale_shift_table[0][None, :, None, None, None]
    shift = decoder.last_scale_shift_table[1][None, :, None, None, None]
    x = x * scale + shift
    import mlx.nn
    x = mlx.nn.silu(x)
    mx.eval(x)
    print(f"After final norm: shape={x.shape}, range=[{np.array(x).min():.3f}, {np.array(x).max():.3f}]")

    # Conv out
    x = decoder.conv_out(x, causal=True)
    mx.eval(x)
    print(f"After conv_out: shape={x.shape}, range=[{np.array(x).min():.3f}, {np.array(x).max():.3f}]")

    # Save pre-unpatchify output for analysis
    np.save("/tmp/pre_unpatchify.npy", np.array(x))
    print("Saved pre-unpatchify output to /tmp/pre_unpatchify.npy")

    # Unpatchify
    from LTX_2_MLX.model.video_vae.ops import unpatchify
    x = unpatchify(x, patch_size_hw=4, patch_size_t=1)
    mx.eval(x)
    print(f"After unpatchify: shape={x.shape}, range=[{np.array(x).min():.3f}, {np.array(x).max():.3f}]")

    # Save final output
    np.save("/tmp/post_unpatchify.npy", np.array(x))
    print("Saved post-unpatchify output to /tmp/post_unpatchify.npy")


import os

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Depth-to-Space 3D")
    print("=" * 60)
    test_depth_to_space_3d()

    print("\n" + "=" * 60)
    print("Testing Conv3d Simple")
    print("=" * 60)
    test_conv3d_simple()

    print("\n" + "=" * 60)
    print("Testing Full Upsample Block")
    print("=" * 60)
    test_full_upsample_block()

    print("\n" + "=" * 60)
    print("Testing Unpatchify Exact Parameters")
    print("=" * 60)
    test_unpatchify_exact()

    print("\n" + "=" * 60)
    print("Testing Video Output Conversion")
    print("=" * 60)
    test_video_output_conversion()

    print("\n" + "=" * 60)
    print("Debugging Decoder Pipeline")
    print("=" * 60)
    debug_decoder_pipeline()
