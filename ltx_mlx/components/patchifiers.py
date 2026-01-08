"""Patchification strategies for LTX-2 latent tensors."""

import math
from typing import Optional, Protocol, Tuple

import mlx.core as mx

from ltx_mlx.types import (
    AudioLatentShape,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)


class PatchifierProtocol(Protocol):
    """Protocol for patchification strategies."""

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        """Return the patch size (temporal, height, width)."""
        ...

    def patchify(self, latents: mx.array) -> mx.array:
        """Convert latents to patch sequence."""
        ...

    def unpatchify(
        self,
        latents: mx.array,
        output_shape: VideoLatentShape | AudioLatentShape,
    ) -> mx.array:
        """Convert patch sequence back to latents."""
        ...


class VideoLatentPatchifier:
    """
    Patchifier for video latent tensors.

    Converts 5D video latents (B, C, F, H, W) to 3D patch sequences (B, N, D)
    where N is the number of patches and D is the patch embedding dimension.
    """

    def __init__(self, patch_size: int):
        """
        Initialize the video patchifier.

        Args:
            patch_size: Spatial patch size (applied to both height and width).
                       Temporal patch size is always 1.
        """
        # Patch sizes: (temporal, height, width)
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        """Return the patch size tuple (temporal, height, width)."""
        return self._patch_size

    def get_token_count(self, tgt_shape: VideoLatentShape) -> int:
        """
        Calculate the number of tokens/patches for a given shape.

        Args:
            tgt_shape: Target video latent shape.

        Returns:
            Number of tokens in the patchified sequence.
        """
        spatial_temporal_size = tgt_shape.frames * tgt_shape.height * tgt_shape.width
        patch_volume = math.prod(self._patch_size)
        return spatial_temporal_size // patch_volume

    def patchify(self, latents: mx.array) -> mx.array:
        """
        Convert video latents to patch sequence.

        Transforms (B, C, F, H, W) -> (B, N, D) where:
        - N = (F/p1) * (H/p2) * (W/p3) is the number of patches
        - D = C * p1 * p2 * p3 is the patch embedding dimension

        Args:
            latents: Video latents of shape (B, C, F, H, W).

        Returns:
            Patch sequence of shape (B, N, D).
        """
        b, c, f, h, w = latents.shape
        p1, p2, p3 = self._patch_size

        # Reshape: (B, C, F, H, W) -> (B, C, F/p1, p1, H/p2, p2, W/p3, p3)
        latents = latents.reshape(b, c, f // p1, p1, h // p2, p2, w // p3, p3)

        # Transpose to group spatial dims together: (B, F/p1, H/p2, W/p3, C, p1, p2, p3)
        latents = latents.transpose(0, 2, 4, 6, 1, 3, 5, 7)

        # Flatten to (B, N, D) where N = (F/p1)*(H/p2)*(W/p3), D = C*p1*p2*p3
        n_patches = (f // p1) * (h // p2) * (w // p3)
        patch_dim = c * p1 * p2 * p3
        latents = latents.reshape(b, n_patches, patch_dim)

        return latents

    def unpatchify(
        self,
        latents: mx.array,
        output_shape: VideoLatentShape,
    ) -> mx.array:
        """
        Convert patch sequence back to video latents.

        Transforms (B, N, D) -> (B, C, F, H, W).

        Args:
            latents: Patch sequence of shape (B, N, D).
            output_shape: Target video latent shape.

        Returns:
            Video latents of shape (B, C, F, H, W).
        """
        assert self._patch_size[0] == 1, "Temporal patch size must be 1"

        b = latents.shape[0]
        c = output_shape.channels
        f = output_shape.frames
        h = output_shape.height
        w = output_shape.width
        p1, p2, p3 = self._patch_size

        # Grid dimensions
        grid_f = f // p1
        grid_h = h // p2
        grid_w = w // p3

        # Reshape: (B, N, D) -> (B, F/p1, H/p2, W/p3, C, p1, p2, p3)
        # Since p1 = 1, this simplifies
        latents = latents.reshape(b, grid_f, grid_h, grid_w, c, p1, p2, p3)

        # Transpose: (B, C, F/p1, p1, H/p2, p2, W/p3, p3)
        latents = latents.transpose(0, 4, 1, 5, 2, 6, 3, 7)

        # Reshape to final: (B, C, F, H, W)
        latents = latents.reshape(b, c, f, h, w)

        return latents

    def get_patch_grid_bounds(
        self,
        output_shape: VideoLatentShape,
    ) -> mx.array:
        """
        Return the per-dimension bounds [start, end) for every patch.

        The bounds are expressed in the original video grid coordinates:
        frame, height, and width.

        Args:
            output_shape: Video grid description.

        Returns:
            Tensor of shape (batch, 3, num_patches, 2) containing start/end
            bounds for each patch in each dimension.
        """
        frames = output_shape.frames
        height = output_shape.height
        width = output_shape.width
        batch_size = output_shape.batch

        # Generate grid coordinates for each dimension
        frame_coords = mx.arange(0, frames, self._patch_size[0])
        height_coords = mx.arange(0, height, self._patch_size[1])
        width_coords = mx.arange(0, width, self._patch_size[2])

        # Create meshgrid (MLX uses ij indexing by default)
        grid_f, grid_h, grid_w = mx.meshgrid(frame_coords, height_coords, width_coords, indexing="ij")

        # Stack to get start coordinates: (3, grid_f, grid_h, grid_w)
        patch_starts = mx.stack([grid_f, grid_h, grid_w], axis=0)

        # Create patch size delta: (3, 1, 1, 1)
        patch_size_delta = mx.array(self._patch_size).reshape(3, 1, 1, 1)

        # Calculate end coordinates
        patch_ends = patch_starts + patch_size_delta

        # Stack start and end: (3, grid_f, grid_h, grid_w, 2)
        latent_coords = mx.stack([patch_starts, patch_ends], axis=-1)

        # Flatten grid dimensions: (3, num_patches, 2)
        num_patches = grid_f.size
        latent_coords = latent_coords.reshape(3, num_patches, 2)

        # Broadcast to batch: (batch, 3, num_patches, 2)
        latent_coords = mx.broadcast_to(
            latent_coords[None, ...],
            (batch_size, 3, num_patches, 2),
        )

        return latent_coords


def get_pixel_coords(
    latent_coords: mx.array,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool = False,
) -> mx.array:
    """
    Map latent-space [start, end) coordinates to pixel-space equivalents.

    Scales each axis (frame, height, width) with the corresponding VAE
    downsampling factors.

    Args:
        latent_coords: Tensor of latent bounds shaped (batch, 3, num_patches, 2).
        scale_factors: SpatioTemporalScaleFactors with integer scale factors.
        causal_fix: When True, rewrites temporal axis for causal VAEs.

    Returns:
        Pixel-space coordinates with same shape as input.
    """
    # Create scale tensor: (1, 3, 1, 1)
    scale_tensor = mx.array([scale_factors.time, scale_factors.height, scale_factors.width])
    scale_tensor = scale_tensor.reshape(1, 3, 1, 1)

    # Apply per-axis scaling
    pixel_coords = latent_coords * scale_tensor

    if causal_fix:
        # VAE temporal stride for first frame is 1 instead of scale_factors.time
        # Shift and clamp to keep timestamps causal and non-negative
        temporal_coords = pixel_coords[:, 0, ...]
        temporal_coords = temporal_coords + 1 - scale_factors.time
        temporal_coords = mx.maximum(temporal_coords, 0)
        # Update the temporal dimension
        pixel_coords = mx.concatenate(
            [temporal_coords[:, None, ...], pixel_coords[:, 1:, ...]],
            axis=1,
        )

    return pixel_coords


class AudioPatchifier:
    """
    Patchifier for audio latent tensors.

    Converts 4D audio latents (B, C, T, F) to 3D patch sequences (B, T, D)
    where T is the number of time steps and D is the feature dimension.
    """

    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        """
        Initialize the audio patchifier.

        Args:
            patch_size: Number of mel bins per patch.
            sample_rate: Original waveform sampling rate.
            hop_length: Window hop length for spectrogram.
            audio_latent_downsample_factor: Ratio between spectrogram and latent frames.
            is_causal: When True, adjust timing for causal receptive fields.
            shift: Integer offset for latent indices.
        """
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        """Return the patch size tuple."""
        return self._patch_size

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        """Return the number of tokens for the given shape."""
        return tgt_shape.frames

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: mx.Dtype = mx.float32,
    ) -> mx.array:
        """
        Convert latent indices to real-time seconds.

        Args:
            start_latent: Inclusive start index.
            end_latent: Exclusive end index.
            dtype: Output dtype.

        Returns:
            Timestamps in seconds for each latent frame.
        """
        audio_latent_frame = mx.arange(start_latent, end_latent).astype(dtype)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor

        if self.is_causal:
            causal_offset = 1
            audio_mel_frame = audio_mel_frame + causal_offset - self.audio_latent_downsample_factor
            audio_mel_frame = mx.maximum(audio_mel_frame, 0)

        return audio_mel_frame * self.hop_length / self.sample_rate

    def _compute_audio_timings(
        self,
        batch_size: int,
        num_steps: int,
    ) -> mx.array:
        """
        Build timing tensor for audio latent frames.

        Args:
            batch_size: Batch size.
            num_steps: Number of latent frames.

        Returns:
            Tensor of shape (batch, 1, num_steps, 2) with start/end timestamps.
        """
        start_timings = self._get_audio_latent_time_in_sec(
            self.shift,
            num_steps + self.shift,
        )
        start_timings = mx.broadcast_to(
            start_timings[None, None, :],
            (batch_size, 1, num_steps),
        )

        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1,
            num_steps + self.shift + 1,
        )
        end_timings = mx.broadcast_to(
            end_timings[None, None, :],
            (batch_size, 1, num_steps),
        )

        return mx.stack([start_timings, end_timings], axis=-1)

    def patchify(self, audio_latents: mx.array) -> mx.array:
        """
        Convert audio latents to patch sequence.

        Transforms (B, C, T, F) -> (B, T, C*F).

        Args:
            audio_latents: Audio latents of shape (B, C, T, F).

        Returns:
            Patch sequence of shape (B, T, C*F).
        """
        b, c, t, f = audio_latents.shape

        # Transpose to (B, T, C, F) then flatten last two dims
        audio_latents = audio_latents.transpose(0, 2, 1, 3)
        audio_latents = audio_latents.reshape(b, t, c * f)

        return audio_latents

    def unpatchify(
        self,
        audio_latents: mx.array,
        output_shape: AudioLatentShape,
    ) -> mx.array:
        """
        Convert patch sequence back to audio latents.

        Transforms (B, T, C*F) -> (B, C, T, F).

        Args:
            audio_latents: Patch sequence of shape (B, T, D).
            output_shape: Target audio latent shape.

        Returns:
            Audio latents of shape (B, C, T, F).
        """
        b, t, _ = audio_latents.shape
        c = output_shape.channels
        f = output_shape.mel_bins

        # Reshape: (B, T, C*F) -> (B, T, C, F)
        audio_latents = audio_latents.reshape(b, t, c, f)

        # Transpose to (B, C, T, F)
        audio_latents = audio_latents.transpose(0, 2, 1, 3)

        return audio_latents

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape,
    ) -> mx.array:
        """
        Return temporal bounds for each audio patch.

        Args:
            output_shape: Audio grid specification.

        Returns:
            Tensor of shape (batch, 1, time_steps, 2) with timestamps.
        """
        return self._compute_audio_timings(output_shape.batch, output_shape.frames)
