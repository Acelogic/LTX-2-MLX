"""Video VAE operations: patchify, unpatchify, normalization."""

import mlx.core as mx
import mlx.nn as nn


def patchify(x: mx.array, patch_size_hw: int, patch_size_t: int = 1) -> mx.array:
    """
    Space-to-depth: rearrange spatial dimensions into channels.

    Divides image into patch_size x patch_size blocks and moves pixels
    from each block into separate channels.

    Args:
        x: Input tensor (4D or 5D).
        patch_size_hw: Spatial patch size for height and width.
        patch_size_t: Temporal patch size for frames (default 1).

    Returns:
        Patchified tensor with increased channels and reduced spatial dims.

    Example:
        5D: (B, C, F, H, W) -> (B, C*p_t*p_h*p_w, F/p_t, H/p_h, W/p_w)
        (B, 3, 33, 512, 512) with patch_size_hw=4 -> (B, 48, 33, 128, 128)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.ndim == 4:
        # 4D: (B, C, H, W) -> (B, C*r*q, H/q, W/r)
        b, c, h, w = x.shape
        q = patch_size_hw
        r = patch_size_hw

        # Reshape: (B, C, H, W) -> (B, C, H/q, q, W/r, r)
        x = x.reshape(b, c, h // q, q, w // r, r)
        # Transpose: (B, C, H/q, q, W/r, r) -> (B, C, r, q, H/q, W/r)
        x = x.transpose(0, 1, 5, 3, 2, 4)
        # Reshape: (B, C, r, q, H/q, W/r) -> (B, C*r*q, H/q, W/r)
        x = x.reshape(b, c * r * q, h // q, w // r)

    elif x.ndim == 5:
        # 5D: (B, C, F, H, W) -> (B, C*p*r*q, F/p, H/q, W/r)
        b, c, f, h, w = x.shape
        p = patch_size_t
        q = patch_size_hw
        r = patch_size_hw

        # Reshape: (B, C, F, H, W) -> (B, C, F/p, p, H/q, q, W/r, r)
        x = x.reshape(b, c, f // p, p, h // q, q, w // r, r)
        # Transpose: -> (B, C, p, r, q, F/p, H/q, W/r)
        x = x.transpose(0, 1, 3, 7, 5, 2, 4, 6)
        # Reshape: -> (B, C*p*r*q, F/p, H/q, W/r)
        x = x.reshape(b, c * p * r * q, f // p, h // q, w // r)

    else:
        raise ValueError(f"Invalid input shape: {x.shape}, expected 4D or 5D")

    return x


def unpatchify(x: mx.array, patch_size_hw: int, patch_size_t: int = 1) -> mx.array:
    """
    Depth-to-space: rearrange channels back into spatial dimensions.

    Inverse of patchify - moves pixels from channels back into
    patch_size x patch_size blocks.

    Args:
        x: Input tensor (4D or 5D).
        patch_size_hw: Spatial patch size for height and width.
        patch_size_t: Temporal patch size for frames (default 1).

    Returns:
        Unpatchified tensor with reduced channels and increased spatial dims.

    Example:
        5D: (B, C*p_t*p_h*p_w, F, H, W) -> (B, C, F*p_t, H*p_h, W*p_w)
        (B, 48, 33, 128, 128) with patch_size_hw=4 -> (B, 3, 33, 512, 512)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.ndim == 4:
        # 4D: (B, C*r*q, H, W) -> (B, C, H*q, W*r)
        b, c_packed, h, w = x.shape
        q = patch_size_hw
        r = patch_size_hw
        c = c_packed // (r * q)

        # Reshape: (B, C*r*q, H, W) -> (B, C, r, q, H, W)
        x = x.reshape(b, c, r, q, h, w)
        # Transpose: (B, C, r, q, H, W) -> (B, C, H, q, W, r)
        x = x.transpose(0, 1, 4, 3, 5, 2)
        # Reshape: (B, C, H, q, W, r) -> (B, C, H*q, W*r)
        x = x.reshape(b, c, h * q, w * r)

    elif x.ndim == 5:
        # 5D: (B, C*p*r*q, F, H, W) -> (B, C, F*p, H*q, W*r)
        b, c_packed, f, h, w = x.shape
        p = patch_size_t
        q = patch_size_hw
        r = patch_size_hw
        c = c_packed // (p * r * q)

        # Reshape: (B, C*p*r*q, F, H, W) -> (B, C, p, r, q, F, H, W)
        x = x.reshape(b, c, p, r, q, f, h, w)
        # Transpose: -> (B, C, F, p, H, q, W, r)
        x = x.transpose(0, 1, 5, 2, 6, 4, 7, 3)
        # Reshape: -> (B, C, F*p, H*q, W*r)
        x = x.reshape(b, c, f * p, h * q, w * r)

    else:
        raise ValueError(f"Invalid input shape: {x.shape}, expected 4D or 5D")

    return x


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing/denormalizing latent representations.

    These statistics are computed over the entire dataset and stored in
    the model's checkpoint under the VAE state_dict.
    """

    def __init__(self, latent_channels: int = 128):
        """
        Initialize per-channel statistics.

        Args:
            latent_channels: Number of latent channels.
        """
        super().__init__()

        # Initialize buffers (will be loaded from checkpoint)
        # Using underscores instead of hyphens for attribute names
        self.std_of_means = mx.zeros((latent_channels,))
        self.mean_of_means = mx.zeros((latent_channels,))
        self.mean_of_stds = mx.zeros((latent_channels,))
        self.mean_of_stds_over_std_of_means = mx.zeros((latent_channels,))
        self.channel = mx.zeros((latent_channels,))

    def un_normalize(self, x: mx.array) -> mx.array:
        """
        Denormalize latent representation.

        Args:
            x: Normalized latent tensor of shape (B, C, F, H, W).

        Returns:
            Denormalized tensor.
        """
        # Reshape stats for broadcasting: (C,) -> (1, C, 1, 1, 1)
        std = self.std_of_means.reshape(1, -1, 1, 1, 1)
        mean = self.mean_of_means.reshape(1, -1, 1, 1, 1)
        return x * std + mean

    def normalize(self, x: mx.array) -> mx.array:
        """
        Normalize latent representation.

        Args:
            x: Raw latent tensor of shape (B, C, F, H, W).

        Returns:
            Normalized tensor.
        """
        # Reshape stats for broadcasting: (C,) -> (1, C, 1, 1, 1)
        std = self.std_of_means.reshape(1, -1, 1, 1, 1)
        mean = self.mean_of_means.reshape(1, -1, 1, 1, 1)
        return (x - mean) / std

    def load_from_dict(self, state_dict: dict, prefix: str = "") -> None:
        """
        Load statistics from a state dict.

        Handles the hyphenated keys from PyTorch checkpoints.

        Args:
            state_dict: State dictionary containing the statistics.
            prefix: Prefix for the keys in the state dict.
        """
        # Map hyphenated PyTorch names to our underscore names
        key_map = {
            "std-of-means": "std_of_means",
            "mean-of-means": "mean_of_means",
            "mean-of-stds": "mean_of_stds",
            "mean-of-stds_over_std-of-means": "mean_of_stds_over_std_of_means",
            "channel": "channel",
        }

        for pt_key, mlx_attr in key_map.items():
            full_key = f"{prefix}{pt_key}" if prefix else pt_key
            if full_key in state_dict:
                setattr(self, mlx_attr, mx.array(state_dict[full_key]))


def pixel_shuffle_3d(x: mx.array, upscale_factor: int) -> mx.array:
    """
    Pixel shuffle for 3D tensors (depth-to-space).

    Rearranges elements in a tensor of shape (B, C*r², F, H, W) to
    (B, C, F, H*r, W*r), where r is the upscale factor.

    Args:
        x: Input tensor of shape (B, C*r², F, H, W).
        upscale_factor: Factor to upscale spatial dimensions.

    Returns:
        Upscaled tensor of shape (B, C, F, H*r, W*r).
    """
    r = upscale_factor
    b, c_packed, f, h, w = x.shape
    c = c_packed // (r * r)

    # Reshape: (B, C*r², F, H, W) -> (B, C, r, r, F, H, W)
    x = x.reshape(b, c, r, r, f, h, w)
    # Transpose: -> (B, C, F, H, r, W, r)
    x = x.transpose(0, 1, 4, 5, 2, 6, 3)
    # Reshape: -> (B, C, F, H*r, W*r)
    x = x.reshape(b, c, f, h * r, w * r)

    return x


def pixel_unshuffle_3d(x: mx.array, downscale_factor: int) -> mx.array:
    """
    Pixel unshuffle for 3D tensors (space-to-depth).

    Rearranges elements in a tensor of shape (B, C, F, H, W) to
    (B, C*r², F, H/r, W/r), where r is the downscale factor.

    Args:
        x: Input tensor of shape (B, C, F, H, W).
        downscale_factor: Factor to downscale spatial dimensions.

    Returns:
        Downscaled tensor of shape (B, C*r², F, H/r, W/r).
    """
    r = downscale_factor
    b, c, f, h, w = x.shape

    # Reshape: (B, C, F, H, W) -> (B, C, F, H/r, r, W/r, r)
    x = x.reshape(b, c, f, h // r, r, w // r, r)
    # Transpose: -> (B, C, r, r, F, H/r, W/r)
    x = x.transpose(0, 1, 4, 6, 2, 3, 5)
    # Reshape: -> (B, C*r², F, H/r, W/r)
    x = x.reshape(b, c * r * r, f, h // r, w // r)

    return x
