#!/usr/bin/env python3
"""Generate video from text prompt using LTX-2 MLX."""

import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ltx_mlx.model.transformer import (
    LTXModel,
    LTXModelType,
    Modality,
    create_position_grid,
)
from ltx_mlx.model.video_vae import VideoDecoder, NormLayerType
from ltx_mlx.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier
from ltx_mlx.types import VideoLatentShape
from ltx_mlx.loader import load_transformer_weights
from ltx_mlx.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    decode_latent,
)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


def progress_bar(iterable, desc=None, total=None):
    """Create a progress bar wrapper."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, ncols=80)
    else:
        return _simple_progress(iterable, desc, total)


def _simple_progress(iterable, desc, total):
    """Simple progress fallback when tqdm is not available."""
    items = list(iterable)
    total = len(items) if total is None else total
    for i, item in enumerate(items):
        print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
        yield item
    print()  # newline after completion


def create_dummy_text_encoding(
    prompt: str,
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,  # Pre-projection dimension (transformer does its own projection)
) -> tuple:
    """
    Create dummy text encoding for testing.

    In production, this should be replaced with actual Gemma encoding.
    Note: Output is 3840-dim because the transformer has its own caption_projection.
    """
    # For now, use random but deterministic encoding based on prompt
    mx.random.seed(hash(prompt) % (2**31))

    # Create text embeddings in pre-projection dimension
    text_encoding = mx.random.normal(shape=(batch_size, max_tokens, embed_dim)) * 0.1
    text_mask = mx.ones((batch_size, max_tokens))

    return text_encoding, text_mask


def load_text_embedding(embedding_path: str) -> tuple:
    """
    Load pre-computed text embedding from file.

    Args:
        embedding_path: Path to .npz file with embedding and attention_mask.

    Returns:
        Tuple of (embedding, attention_mask).
    """
    data = np.load(embedding_path)
    embedding = mx.array(data["embedding"])
    mask = mx.array(data["attention_mask"])

    print(f"  Loaded embedding from {embedding_path}")
    print(f"  Shape: {embedding.shape}")

    if "prompt" in data:
        print(f"  Original prompt: {data['prompt']}")

    return embedding, mask


def load_vae_decoder(weights_path: str) -> VideoDecoder:
    """Load VAE decoder with weights."""
    print("Loading VAE decoder...")

    # LTX-2 decoder configuration
    decoder_blocks = [
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 4}),
        ("compress_time", {}),
        ("res_x", {"num_layers": 4}),
        ("compress_space", {}),
        ("res_x", {"num_layers": 4}),
        ("compress_space", {}),
    ]

    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
    )

    # TODO: Load VAE weights from file
    print("  VAE decoder created (weights not loaded yet)")

    return decoder


def load_transformer(weights_path: str, num_layers: int = 48) -> LTXModel:
    """Load transformer with weights."""
    print("Loading transformer...")

    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=num_layers,
        cross_attention_dim=4096,
        caption_channels=3840,
        positional_embedding_theta=10000.0,
    )

    # Load weights
    if weights_path and os.path.exists(weights_path):
        load_transformer_weights(model, weights_path)
    else:
        print(f"  Warning: Weights not found at {weights_path}, using random init")

    return model


def euler_step(
    latent: mx.array,
    velocity: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """Perform one Euler diffusion step."""
    # Euler step: x_next = x + (sigma_next - sigma) * velocity
    dt = sigma_next - sigma
    return latent + dt * velocity


def generate_video(
    prompt: str,
    height: int = 480,
    width: int = 704,
    num_frames: int = 97,  # 12 seconds at 8fps after VAE (97 = 1 + 12*8)
    num_steps: int = 7,  # Distilled model uses 7 steps
    cfg_scale: float = 3.0,
    seed: int = 42,
    weights_path: str = None,
    output_path: str = "output.mp4",
    use_placeholder: bool = False,
    skip_vae: bool = False,
    embedding_path: str = None,
):
    """Generate video from text prompt."""

    print(f"\n{'='*50}")
    print(f"LTX-2 MLX Video Generation")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}, {num_frames} frames")
    print(f"Steps: {num_steps}, CFG: {cfg_scale}, Seed: {seed}")
    if skip_vae:
        print(f"VAE decoding: SKIPPED")
    if embedding_path:
        print(f"Using pre-computed embedding: {embedding_path}")

    # Set seed
    mx.random.seed(seed)

    # Compute latent dimensions
    # VAE: 32x spatial, 8x temporal compression
    latent_height = height // 32
    latent_width = width // 32
    latent_frames = (num_frames - 1) // 8 + 1

    print(f"\nLatent shape: {latent_frames}x{latent_height}x{latent_width}")

    # Get text encoding
    print("\n[1/5] Encoding prompt...")
    if embedding_path:
        text_encoding, text_mask = load_text_embedding(embedding_path)
    else:
        text_encoding, text_mask = create_dummy_text_encoding(prompt)
        print("  Using dummy encoding (Gemma integration pending)")

    # Load model
    print("\n[2/5] Loading transformer...")
    if not use_placeholder and weights_path:
        model = load_transformer(weights_path, num_layers=48)
    else:
        model = None
        print("  Skipping model load (placeholder mode)")

    # Load VAE decoder
    vae_decoder = None
    if not skip_vae and weights_path:
        print("\n[3/5] Loading VAE decoder...")
        vae_decoder = SimpleVideoDecoder()
        load_vae_decoder_weights(vae_decoder, weights_path)
    elif not skip_vae:
        print("\n[3/5] Skipping VAE decoder (no weights)")
    else:
        print("\n[3/5] VAE decoder skipped by user")

    # Initialize noise
    print("\n[4/5] Initializing latent noise...")
    latent = mx.random.normal(shape=(1, 128, latent_frames, latent_height, latent_width))

    # Get sigma schedule (distilled)
    sigmas = mx.array(DISTILLED_SIGMA_VALUES[:num_steps + 1])
    print(f"  Sigma schedule: {[f'{float(s):.3f}' for s in sigmas]}")

    # Create patchifier
    patchifier = VideoLatentPatchifier(patch_size=1)

    print(f"\n[5/5] Denoising ({num_steps} steps)...")

    # Denoising loop with progress bar
    step_iterator = progress_bar(range(len(sigmas) - 1), desc="Denoising", total=num_steps)

    for i in step_iterator:
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        if model is not None and not use_placeholder:
            # === Actual model inference ===
            # Patchify latent: [B, C, F, H, W] -> [B, T, C]
            latent_patchified = patchifier.patchify(latent)
            num_tokens = latent_patchified.shape[1]

            # Create position grid with bounds
            grid = create_position_grid(1, latent_frames, latent_height, latent_width)
            grid_start = grid[..., None]
            grid_end = grid_start + 1
            positions = mx.concatenate([grid_start, grid_end], axis=-1)

            # Create modality input
            modality = Modality(
                latent=latent_patchified,
                context=text_encoding,
                context_mask=text_mask,
                timesteps=mx.array([sigma]),
                positions=positions,
                enabled=True,
            )

            # Run transformer
            velocity_patchified = model(modality)

            # Unpatchify velocity: [B, T, C] -> [B, C, F, H, W]
            output_shape = VideoLatentShape(
                batch=1,
                channels=128,
                frames=latent_frames,
                height=latent_height,
                width=latent_width,
            )
            velocity = patchifier.unpatchify(velocity_patchified, output_shape=output_shape)
        else:
            # Placeholder: random velocity
            velocity = mx.random.normal(shape=latent.shape) * 0.1

        # Euler step
        latent = euler_step(latent, velocity, sigma, sigma_next)

        # Force evaluation for memory efficiency
        mx.eval(latent)

    # Save denoised latent
    latent_path = output_path.replace('.mp4', '_latent.npz')
    print(f"\nSaving denoised latent to {latent_path}...")
    np.savez(latent_path, latent=np.array(latent))
    print(f"  Latent shape: {latent.shape}")

    # Decode with VAE or create placeholder
    if vae_decoder is not None:
        print("\nDecoding with VAE...")
        print(f"  Input latent: {latent.shape}")

        # VAE decode
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)
        print(f"  Output video: {video.shape}")

        # Convert to frames list for save_video
        frames = [np.array(video[f]) for f in range(video.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

    else:
        print("\nCreating placeholder video (VAE not loaded)...")

        # Placeholder output - create simple visualization based on latent statistics
        frames = []
        latent_np = np.array(latent[0])  # (C, F, H, W)
        latent_mean = latent_np.mean(axis=0)  # (F, H, W)
        latent_std = latent_np.std(axis=0)

        frame_iterator = progress_bar(range(num_frames), desc="Creating frames", total=num_frames)

        for f in frame_iterator:
            # Map latent frame to visualization
            lat_f = f * latent_frames // num_frames
            lat_f = min(lat_f, latent_frames - 1)

            # Get latent statistics for this frame
            lat_slice_mean = latent_mean[lat_f]  # (H, W)
            lat_slice_std = latent_std[lat_f]

            # Create frame based on latent (upscale from latent to output resolution)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Simple bilinear-ish upscale of latent visualization
            for y in range(height):
                for x in range(width):
                    lat_y = y * latent_height // height
                    lat_x = x * latent_width // width
                    lat_y = min(lat_y, latent_height - 1)
                    lat_x = min(lat_x, latent_width - 1)

                    # Use latent values to create RGB
                    val_mean = float(lat_slice_mean[lat_y, lat_x])
                    val_std = float(lat_slice_std[lat_y, lat_x])

                    # Normalize and convert to color
                    r = int(np.clip((val_mean + 2) / 4 * 255, 0, 255))
                    g = int(np.clip((val_std) / 2 * 255, 0, 255))
                    b = int(np.clip((val_mean * val_std + 1) / 2 * 128, 0, 255))

                    frame[y, x] = [r, g, b]

            frames.append(frame)

    # Save as video
    print(f"\nSaving video to {output_path}...")
    save_video(frames, output_path, fps=24)

    print(f"\nDone! Video saved to {output_path}")

    if use_placeholder or model is None:
        print("\nNote: This is a placeholder output. Full inference requires:")
        print("  1. Proper weight loading (use --weights flag)")
        print("  2. Gemma text encoder integration")

    if vae_decoder is None and not skip_vae:
        print("\nNote: VAE decoder was not loaded - output is placeholder visualization.")


def save_video(frames: list, output_path: str, fps: int = 24):
    """Save frames as video using ffmpeg."""
    import subprocess
    import tempfile
    from PIL import Image

    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as images with progress
        print("  Writing frames...")
        if HAS_TQDM:
            iterator = tqdm(enumerate(frames), desc="  Saving frames", total=len(frames), ncols=80)
        else:
            iterator = enumerate(frames)

        for i, frame in iterator:
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        # Use ffmpeg to create video
        print("\n  Encoding video...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-loglevel", "error",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Generate video with LTX-2 MLX")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=704, help="Video width")
    parser.add_argument("--frames", type=int, default=97, help="Number of frames")
    parser.add_argument("--steps", type=int, default=7, help="Denoising steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output path")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to weights"
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Use placeholder inference (skip model loading)"
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip VAE decoding (output latent visualization instead)"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Path to pre-computed text embedding (.npz)"
    )

    args = parser.parse_args()

    generate_video(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        weights_path=args.weights,
        output_path=args.output,
        use_placeholder=args.placeholder,
        skip_vae=args.skip_vae,
        embedding_path=args.embedding,
    )


if __name__ == "__main__":
    main()
