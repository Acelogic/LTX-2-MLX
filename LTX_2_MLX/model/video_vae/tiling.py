"""Tiled VAE decoding for memory-efficient high-resolution video generation."""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import mlx.core as mx


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> mx.array:
    """
    Generate a 1D trapezoidal blending mask with linear ramps.

    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.

    Returns:
        A 1D tensor of shape (length,) with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = mx.ones((length,))

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = mx.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        # Replace first ramp_left elements
        mask_before = fade_in
        mask_after = mask[ramp_left:]
        mask = mx.concatenate([mask_before, mask_after])

    if ramp_right > 0:
        fade_out = mx.linspace(1.0, 0.0, ramp_right + 2)[1:-1]
        # Replace last ramp_right elements
        mask_before = mask[:-ramp_right]
        mask = mx.concatenate([mask_before, fade_out])

    return mx.clip(mask, 0, 1)


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap.

    Args:
        tile_size_in_pixels: Size of each tile in pixels. Must be at least 64 and divisible by 32.
        tile_overlap_in_pixels: Overlap between tiles in pixels. Must be divisible by 32.
    """

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles with optional overlap.

    Args:
        tile_size_in_frames: Number of frames in each tile. Must be at least 16 and divisible by 8.
        tile_overlap_in_frames: Number of overlapping frames between consecutive tiles.
    """

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap.

    Attributes:
        spatial_config: Configuration for splitting spatial dimensions into tiles.
        temporal_config: Configuration for splitting temporal dimension into tiles.
    """

    spatial_config: Optional[SpatialTilingConfig] = None
    temporal_config: Optional[TemporalTilingConfig] = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
        )


@dataclass
class TileSpec:
    """Specification for a single tile."""

    # Input coordinates (slice of latent to decode)
    in_t_start: int
    in_t_end: int
    in_h_start: int
    in_h_end: int
    in_w_start: int
    in_w_end: int

    # Output coordinates (where to place decoded pixels)
    out_t_start: int
    out_t_end: int
    out_h_start: int
    out_h_end: int
    out_w_start: int
    out_w_end: int

    # Ramp sizes for blending
    ramp_t_left: int
    ramp_t_right: int
    ramp_h_left: int
    ramp_h_right: int
    ramp_w_left: int
    ramp_w_right: int


def generate_tile_specs(
    latent_shape: Tuple[int, int, int, int, int],
    tiling_config: TilingConfig,
    scale_factors: Tuple[int, int, int] = (8, 32, 32),
) -> List[TileSpec]:
    """
    Generate tile specifications for tiled decoding.

    Args:
        latent_shape: Shape of latent tensor (B, C, T, H, W).
        tiling_config: Tiling configuration.
        scale_factors: Upscaling factors (temporal, height, width).

    Returns:
        List of TileSpec objects.
    """
    _, _, t, h, w = latent_shape
    scale_t, scale_h, scale_w = scale_factors

    tiles = []

    # Calculate latent tile sizes
    if tiling_config.spatial_config:
        spatial_cfg = tiling_config.spatial_config
        tile_h_latent = spatial_cfg.tile_size_in_pixels // scale_h
        tile_w_latent = spatial_cfg.tile_size_in_pixels // scale_w
        overlap_h_latent = spatial_cfg.tile_overlap_in_pixels // scale_h
        overlap_w_latent = spatial_cfg.tile_overlap_in_pixels // scale_w
    else:
        tile_h_latent = h
        tile_w_latent = w
        overlap_h_latent = 0
        overlap_w_latent = 0

    if tiling_config.temporal_config:
        temporal_cfg = tiling_config.temporal_config
        tile_t_latent = temporal_cfg.tile_size_in_frames // scale_t
        overlap_t_latent = temporal_cfg.tile_overlap_in_frames // scale_t
    else:
        tile_t_latent = t
        overlap_t_latent = 0

    # Generate tile coordinates
    def gen_tiles_1d(length: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
        """Generate (start, end, ramp_left, ramp_right) for each tile."""
        if length <= tile_size:
            return [(0, length, 0, 0)]

        tiles_1d = []
        stride = tile_size - overlap
        pos = 0
        while pos < length:
            end = min(pos + tile_size, length)
            start = max(0, end - tile_size)

            # Ramps for blending
            ramp_left = overlap if start > 0 else 0
            ramp_right = overlap if end < length else 0

            tiles_1d.append((start, end, ramp_left, ramp_right))

            if end >= length:
                break
            pos += stride

        return tiles_1d

    t_tiles = gen_tiles_1d(t, tile_t_latent, overlap_t_latent)
    h_tiles = gen_tiles_1d(h, tile_h_latent, overlap_h_latent)
    w_tiles = gen_tiles_1d(w, tile_w_latent, overlap_w_latent)

    # Generate all tile combinations
    for t_start, t_end, ramp_t_l, ramp_t_r in t_tiles:
        for h_start, h_end, ramp_h_l, ramp_h_r in h_tiles:
            for w_start, w_end, ramp_w_l, ramp_w_r in w_tiles:
                # Convert latent coordinates to pixel coordinates
                out_t_start = t_start * scale_t if t_start > 0 else 0
                out_t_end = (t_end - 1) * scale_t + 1 if t_end > 1 else 1
                out_h_start = h_start * scale_h
                out_h_end = h_end * scale_h
                out_w_start = w_start * scale_w
                out_w_end = w_end * scale_w

                tiles.append(TileSpec(
                    in_t_start=t_start, in_t_end=t_end,
                    in_h_start=h_start, in_h_end=h_end,
                    in_w_start=w_start, in_w_end=w_end,
                    out_t_start=out_t_start, out_t_end=out_t_end,
                    out_h_start=out_h_start, out_h_end=out_h_end,
                    out_w_start=out_w_start, out_w_end=out_w_end,
                    ramp_t_left=ramp_t_l * scale_t, ramp_t_right=ramp_t_r * scale_t,
                    ramp_h_left=ramp_h_l * scale_h, ramp_h_right=ramp_h_r * scale_h,
                    ramp_w_left=ramp_w_l * scale_w, ramp_w_right=ramp_w_r * scale_w,
                ))

    return tiles


def decode_tiled(
    latent: mx.array,
    decoder_fn,
    tiling_config: TilingConfig,
    timestep: Optional[float] = 0.05,
    show_progress: bool = True,
    key: Optional[mx.array] = None,
) -> Iterator[mx.array]:
    """
    Decode a latent tensor using tiled processing.

    Splits the latent tensor into tiles, decodes each tile individually,
    and yields video chunks as they become available.

    Args:
        latent: Input latent tensor (B, C, T, H, W).
        decoder_fn: Function to decode a latent tile, takes (latent, timestep).
        tiling_config: Tiling configuration.
        timestep: Timestep for decoder conditioning.
        show_progress: Whether to show progress bar.
        key: Optional random key for deterministic decoding (reserved for future use).

    Yields:
        Decoded video chunks (B, 3, T_chunk, H, W).
    """
    b, c, t, h, w = latent.shape

    # Generate tile specifications
    tiles = generate_tile_specs(latent.shape, tiling_config)

    if show_progress:
        try:
            from tqdm import tqdm
            tiles = list(tqdm(tiles, desc="Tiled decode", ncols=80))
        except ImportError:
            pass

    # Calculate output shape
    scale_t, scale_h, scale_w = 8, 32, 32
    out_t = (t - 1) * scale_t + 1
    out_h = h * scale_h
    out_w = w * scale_w

    # Initialize output buffer and weight accumulator
    output = mx.zeros((b, 3, out_t, out_h, out_w))
    weights = mx.zeros((b, 1, out_t, out_h, out_w))

    for tile_spec in tiles:
        # Extract tile from latent
        tile_latent = latent[
            :, :,
            tile_spec.in_t_start:tile_spec.in_t_end,
            tile_spec.in_h_start:tile_spec.in_h_end,
            tile_spec.in_w_start:tile_spec.in_w_end,
        ]

        # Decode tile
        decoded_tile = decoder_fn(tile_latent, timestep=timestep)
        mx.eval(decoded_tile)

        # Generate blending mask
        tile_t = tile_spec.out_t_end - tile_spec.out_t_start
        tile_h = tile_spec.out_h_end - tile_spec.out_h_start
        tile_w = tile_spec.out_w_end - tile_spec.out_w_start

        mask_t = compute_trapezoidal_mask_1d(
            tile_t, tile_spec.ramp_t_left, tile_spec.ramp_t_right,
            left_starts_from_0=(tile_spec.out_t_start == 0)
        )
        mask_h = compute_trapezoidal_mask_1d(tile_h, tile_spec.ramp_h_left, tile_spec.ramp_h_right)
        mask_w = compute_trapezoidal_mask_1d(tile_w, tile_spec.ramp_w_left, tile_spec.ramp_w_right)

        # Create 5D mask by outer product
        mask = mask_t[None, None, :, None, None] * mask_h[None, None, None, :, None] * mask_w[None, None, None, None, :]

        # Accumulate into output buffer
        # We need to handle the accumulation carefully since MLX doesn't have in-place ops
        out_slice = (
            slice(None), slice(None),
            slice(tile_spec.out_t_start, tile_spec.out_t_end),
            slice(tile_spec.out_h_start, tile_spec.out_h_end),
            slice(tile_spec.out_w_start, tile_spec.out_w_end),
        )

        # Create update arrays
        decoded_actual = decoded_tile[:, :, :tile_t, :tile_h, :tile_w]
        output_update = decoded_actual * mask
        weight_update = mask

        # For simplicity, we'll create a sparse update
        # This is less efficient but clearer
        temp_output = mx.zeros_like(output)
        temp_weights = mx.zeros_like(weights)

        # We'll use a different approach: process tiles in temporal order
        # and yield chunks as they complete

    # Since MLX doesn't have efficient scatter operations, we'll use a different approach
    # Process tiles and yield the final blended result
    output = mx.zeros((b, 3, out_t, out_h, out_w))
    weights = mx.zeros((1, 1, out_t, out_h, out_w))

    for tile_spec in tiles:
        # Extract and decode tile
        tile_latent = latent[
            :, :,
            tile_spec.in_t_start:tile_spec.in_t_end,
            tile_spec.in_h_start:tile_spec.in_h_end,
            tile_spec.in_w_start:tile_spec.in_w_end,
        ]

        decoded_tile = decoder_fn(tile_latent, timestep=timestep)
        mx.eval(decoded_tile)

        # Get actual decoded dimensions
        _, _, dt, dh, dw = decoded_tile.shape
        tile_t = min(dt, tile_spec.out_t_end - tile_spec.out_t_start)
        tile_h = min(dh, tile_spec.out_h_end - tile_spec.out_h_start)
        tile_w = min(dw, tile_spec.out_w_end - tile_spec.out_w_start)

        # Generate blending mask
        mask_t = compute_trapezoidal_mask_1d(
            tile_t, min(tile_spec.ramp_t_left, tile_t), min(tile_spec.ramp_t_right, tile_t),
            left_starts_from_0=(tile_spec.out_t_start == 0)
        )
        mask_h = compute_trapezoidal_mask_1d(
            tile_h, min(tile_spec.ramp_h_left, tile_h), min(tile_spec.ramp_h_right, tile_h)
        )
        mask_w = compute_trapezoidal_mask_1d(
            tile_w, min(tile_spec.ramp_w_left, tile_w), min(tile_spec.ramp_w_right, tile_w)
        )

        # Create 5D mask
        mask = mask_t[None, None, :, None, None] * mask_h[None, None, None, :, None] * mask_w[None, None, None, None, :]

        # Slice decoded tile to actual size
        decoded_slice = decoded_tile[:, :, :tile_t, :tile_h, :tile_w]

        # Update output and weights using slicing
        # This is inefficient but necessary without scatter operations
        out_t_slice = slice(tile_spec.out_t_start, tile_spec.out_t_start + tile_t)
        out_h_slice = slice(tile_spec.out_h_start, tile_spec.out_h_start + tile_h)
        out_w_slice = slice(tile_spec.out_w_start, tile_spec.out_w_start + tile_w)

        # Extract current values
        current_output = output[:, :, out_t_slice, out_h_slice, out_w_slice]
        current_weights = weights[:, :, out_t_slice, out_h_slice, out_w_slice]

        # Add weighted contribution
        new_output = current_output + decoded_slice * mask
        new_weights = current_weights + mask

        # Reconstruct output array (this is expensive but necessary)
        # Build slices for before and after
        output = _update_slice_5d(output, new_output, out_t_slice, out_h_slice, out_w_slice)
        weights = _update_slice_5d(weights, new_weights, out_t_slice, out_h_slice, out_w_slice)

    # Normalize by weights
    output = output / mx.maximum(weights, 1e-8)

    yield output


def _update_slice_5d(
    arr: mx.array,
    update: mx.array,
    t_slice: slice,
    h_slice: slice,
    w_slice: slice,
) -> mx.array:
    """Update a 5D array at given slices. Returns new array."""
    # This is a workaround for lack of in-place operations in MLX
    # We concatenate the before/update/after segments

    b, c, t_total, h_total, w_total = arr.shape

    t_start = t_slice.start or 0
    t_end = t_slice.stop or t_total
    h_start = h_slice.start or 0
    h_end = h_slice.stop or h_total
    w_start = w_slice.start or 0
    w_end = w_slice.stop or w_total

    # Build along width dimension first
    w_before = arr[:, :, t_start:t_end, h_start:h_end, :w_start] if w_start > 0 else None
    w_after = arr[:, :, t_start:t_end, h_start:h_end, w_end:] if w_end < w_total else None

    # Concatenate width segments
    if w_before is not None and w_after is not None:
        row_updated = mx.concatenate([w_before, update, w_after], axis=4)
    elif w_before is not None:
        row_updated = mx.concatenate([w_before, update], axis=4)
    elif w_after is not None:
        row_updated = mx.concatenate([update, w_after], axis=4)
    else:
        row_updated = update

    # Build along height dimension
    h_before = arr[:, :, t_start:t_end, :h_start, :] if h_start > 0 else None
    h_after = arr[:, :, t_start:t_end, h_end:, :] if h_end < h_total else None

    if h_before is not None and h_after is not None:
        plane_updated = mx.concatenate([h_before, row_updated, h_after], axis=3)
    elif h_before is not None:
        plane_updated = mx.concatenate([h_before, row_updated], axis=3)
    elif h_after is not None:
        plane_updated = mx.concatenate([row_updated, h_after], axis=3)
    else:
        plane_updated = row_updated

    # Build along time dimension
    t_before = arr[:, :, :t_start, :, :] if t_start > 0 else None
    t_after = arr[:, :, t_end:, :, :] if t_end < t_total else None

    if t_before is not None and t_after is not None:
        result = mx.concatenate([t_before, plane_updated, t_after], axis=2)
    elif t_before is not None:
        result = mx.concatenate([t_before, plane_updated], axis=2)
    elif t_after is not None:
        result = mx.concatenate([plane_updated, t_after], axis=2)
    else:
        result = plane_updated

    return result
