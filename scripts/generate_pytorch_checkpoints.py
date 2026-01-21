#!/usr/bin/env python3
"""
Generate PyTorch inference checkpoints for MLX parity testing.

This script runs PyTorch LTX-2 inference and saves intermediate tensors:
- Text encoder output
- Initial noisy latent
- Transformer outputs at each step (0-7)
- Final latent before VAE
- VAE decoder output pixels
- Config manifest with exact parameters

Usage:
    python scripts/generate_pytorch_checkpoints.py \
        --prompt "A golden retriever running through a meadow" \
        --height 128 --width 128 --frames 17 --steps 8 --seed 42 \
        --output-dir /tmp/pytorch_parity_checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Setup paths for PyTorch LTX-2
PYTORCH_DIR = Path("/Users/mcruz/Developer/LTX-2-Pytorch")
sys.path.insert(0, str(PYTORCH_DIR))
import mock_triton  # noqa: F401 - Must mock triton before importing PyTorch LTX

sys.path.insert(0, str(PYTORCH_DIR / "packages" / "ltx-core" / "src"))
sys.path.insert(0, str(PYTORCH_DIR / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(PYTORCH_DIR / "packages" / "ltx-trainer" / "src"))

import numpy as np
import torch


def generate_checkpoints(
    prompt: str,
    height: int = 128,
    width: int = 128,
    num_frames: int = 17,
    num_steps: int = 8,
    seed: int = 42,
    output_dir: str = "/tmp/pytorch_parity_checkpoints",
    model_path: str = "weights/ltx-2/ltx-2-19b-distilled.safetensors",
    gemma_path: str = "weights/gemma-3-12b",
    negative_prompt: str = "",
    guidance_scale: float = 1.0,
):
    """Generate PyTorch inference checkpoints for parity testing."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PYTORCH CHECKPOINT GENERATOR")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Prompt: '{prompt[:60]}...'")
    print(f"  Resolution: {height}x{width}, {num_frames} frames")
    print(f"  Steps: {num_steps}, Seed: {seed}")
    print(f"  CFG: {guidance_scale}")
    print(f"  Output: {output_dir}")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Config for manifest
    config = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "model_path": model_path,
        "gemma_path": gemma_path,
    }

    checkpoints = []

    # === Load Models ===
    print("\n" + "-" * 70)
    print("Loading Models")
    print("-" * 70)

    from ltx_trainer.model_loader import load_model

    # Load LTX components (includes text encoder with Gemma)
    # Use MPS for Apple Silicon acceleration, bfloat16 to match Gemma's default dtype
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading LTX-2 components on {device}...")
    components = load_model(
        checkpoint_path=model_path,
        device=device,
        dtype=torch.bfloat16,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=True,
        text_encoder_path=gemma_path,
    )

    transformer = components.transformer
    text_encoder = components.text_encoder
    vae_decoder = components.video_vae_decoder

    transformer.eval()
    text_encoder.eval()
    vae_decoder.eval()

    # Disable noise in VAE for deterministic comparison
    vae_decoder.decode_noise_scale = 0.0

    # === Text Encoding ===
    print("\n" + "-" * 70)
    print("Text Encoding")
    print("-" * 70)

    # PyTorch text encoder takes a string directly (handles tokenization internally)
    print(f"  Running text encoder with prompt: '{prompt[:50]}...'")
    with torch.no_grad():
        text_output = text_encoder(text=prompt, padding_side="left")
        # Extract video encoding from the structured output
        video_encoding = text_output.video_encoding
        text_mask = text_output.attention_mask

    print(f"  Video encoding shape: {video_encoding.shape}")

    # Save text encoder output
    video_encoding_np = video_encoding.detach().cpu().float().numpy()
    ckpt_path = os.path.join(output_dir, "text_encoder_video_encoding.npy")
    np.save(ckpt_path, video_encoding_np)
    checkpoints.append({
        "name": "text_encoder_video_encoding",
        "path": ckpt_path,
        "shape": list(video_encoding_np.shape),
        "min": float(video_encoding_np.min()),
        "max": float(video_encoding_np.max()),
        "mean": float(video_encoding_np.mean()),
        "std": float(video_encoding_np.std()),
    })
    print(f"  Saved: {ckpt_path}")

    # === Transformer Denoising ===
    print("\n" + "-" * 70)
    print("Transformer Denoising")
    print("-" * 70)

    # Calculate latent shape
    latent_frames = (num_frames - 1) // 8 + 1
    latent_height = height // 32
    latent_width = width // 32
    latent_channels = 128

    print(f"  Latent shape: (1, {latent_channels}, {latent_frames}, {latent_height}, {latent_width})")

    # Generate initial noise (matching shape: [B, C, T, H, W])
    torch.manual_seed(seed)  # Reset seed for reproducible noise
    initial_latent = torch.randn(
        1, latent_channels, latent_frames, latent_height, latent_width,
        dtype=torch.bfloat16,
        device=device,
    )

    # Save initial latent
    initial_np = initial_latent.detach().cpu().float().numpy()
    ckpt_path = os.path.join(output_dir, "initial_latent.npy")
    np.save(ckpt_path, initial_np)
    checkpoints.append({
        "name": "initial_latent",
        "path": ckpt_path,
        "shape": list(initial_np.shape),
        "min": float(initial_np.min()),
        "max": float(initial_np.max()),
        "mean": float(initial_np.mean()),
        "std": float(initial_np.std()),
    })
    print(f"  Saved initial latent: {ckpt_path}")

    # Get distilled sigma schedule
    # Use the same DISTILLED_SIGMA_VALUES as MLX for consistent comparison
    # These are the official distilled model sigmas
    DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    sigmas = DISTILLED_SIGMA_VALUES[:num_steps + 1]

    print(f"  Sigmas: {[f'{s:.4f}' for s in sigmas]}")

    # Patchify setup
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import VideoLatentShape, SpatioTemporalScaleFactors
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    # Wrap transformer with X0Model to get denoised (x0) outputs instead of velocity
    x0_model = X0Model(transformer)

    patchifier = VideoLatentPatchifier(patch_size=1)

    # Create output shape object
    output_shape = VideoLatentShape(
        batch=1,
        channels=latent_channels,
        frames=latent_frames,
        height=latent_height,
        width=latent_width,
    )

    # Get positions for transformer
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=output_shape,
        device=torch.device(device),
    )

    scale_factors = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    positions = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=scale_factors,
        causal_fix=True,
    )

    # Prepare for denoising loop
    latent = initial_latent.clone()

    # Scale initial noise by sigma_max
    latent = latent * sigmas[0]

    print(f"  Running {num_steps} denoising steps...")

    # Create empty perturbation config (no perturbations)
    from ltx_core.guidance.perturbations import PerturbationConfig
    perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig.empty()])

    # Save positions for debugging
    positions_np = positions.detach().cpu().float().numpy()
    ckpt_path = os.path.join(output_dir, "positions.npy")
    np.save(ckpt_path, positions_np)
    checkpoints.append({
        "name": "positions",
        "path": ckpt_path,
        "shape": list(positions_np.shape),
        "min": float(positions_np.min()),
        "max": float(positions_np.max()),
    })
    print(f"  Saved positions: {ckpt_path}")

    for step_idx in range(num_steps):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        # Patchify latent
        latent_patchified = patchifier.patchify(latent)

        # Save step 0 patchified latent for debugging
        if step_idx == 0:
            patchified_np = latent_patchified.detach().cpu().float().numpy()
            ckpt_path = os.path.join(output_dir, "step0_patchified_latent.npy")
            np.save(ckpt_path, patchified_np)
            checkpoints.append({
                "name": "step0_patchified_latent",
                "path": ckpt_path,
                "shape": list(patchified_np.shape),
                "min": float(patchified_np.min()),
                "max": float(patchified_np.max()),
                "mean": float(patchified_np.mean()),
                "std": float(patchified_np.std()),
            })
            print(f"  Saved step 0 patchified latent: {ckpt_path}")

        # Create timestep tensor (shape [B] for number of timesteps per batch)
        timesteps = torch.tensor([sigma], dtype=torch.bfloat16, device=device)

        # Create Modality object for video
        # NOTE: PyTorch LTX uses context_mask=None
        video_modality = Modality(
            latent=latent_patchified.to(torch.bfloat16),
            context=video_encoding,
            context_mask=None,  # CRITICAL: PyTorch uses None
            timesteps=timesteps,
            positions=positions.to(torch.bfloat16),
            enabled=True,
        )

        # Run X0Model to get denoised (x0) outputs
        with torch.no_grad():
            x0_video, _ = x0_model(
                video=video_modality,
                audio=None,
                perturbations=perturbations,
            )

        # Save transformer output for this step
        x0_np = x0_video.detach().cpu().float().numpy()
        ckpt_name = f"transformer_step_{step_idx:03d}"
        ckpt_path = os.path.join(output_dir, f"{ckpt_name}.npy")
        np.save(ckpt_path, x0_np)
        checkpoints.append({
            "name": ckpt_name,
            "path": ckpt_path,
            "shape": list(x0_np.shape),
            "min": float(x0_np.min()),
            "max": float(x0_np.max()),
            "mean": float(x0_np.mean()),
            "std": float(x0_np.std()),
        })
        print(f"    Step {step_idx}: sigma={sigma:.4f} -> {sigma_next:.4f}, saved {ckpt_path}")

        # Unpatchify to get denoised prediction
        denoised = patchifier.unpatchify(x0_video, output_shape=output_shape)

        # Euler step: latent = latent + (denoised - latent) * (sigma_next - sigma) / sigma
        # This is X0 prediction mode
        if sigma_next == 0:
            latent = denoised
        else:
            velocity = (latent - denoised) / sigma
            latent = latent + velocity * (sigma_next - sigma)

    # === VAE Decode ===
    print("\n" + "-" * 70)
    print("VAE Decode")
    print("-" * 70)

    # Save final latent (input to VAE)
    vae_input_np = latent.detach().cpu().float().numpy()
    ckpt_path = os.path.join(output_dir, "vae_decoder_input_latent.npy")
    np.save(ckpt_path, vae_input_np)
    checkpoints.append({
        "name": "vae_decoder_input_latent",
        "path": ckpt_path,
        "shape": list(vae_input_np.shape),
        "min": float(vae_input_np.min()),
        "max": float(vae_input_np.max()),
        "mean": float(vae_input_np.mean()),
        "std": float(vae_input_np.std()),
    })
    print(f"  Saved VAE input: {ckpt_path}")

    # Run VAE decoder
    with torch.no_grad():
        pixels = vae_decoder(latent)

    # Save output pixels
    pixels_np = pixels.detach().cpu().float().numpy()
    ckpt_path = os.path.join(output_dir, "vae_decoder_output_pixels.npy")
    np.save(ckpt_path, pixels_np)
    checkpoints.append({
        "name": "vae_decoder_output_pixels",
        "path": ckpt_path,
        "shape": list(pixels_np.shape),
        "min": float(pixels_np.min()),
        "max": float(pixels_np.max()),
        "mean": float(pixels_np.mean()),
        "std": float(pixels_np.std()),
    })
    print(f"  Saved VAE output: {ckpt_path}")

    # === Save Manifest ===
    manifest = {
        "config": config,
        "checkpoints": checkpoints,
        "sigmas": sigmas,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved manifest: {manifest_path}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nGenerated {len(checkpoints)} checkpoints in {output_dir}:")
    for ckpt in checkpoints:
        print(f"  - {ckpt['name']}: {ckpt['shape']}")

    print("\nTo run MLX comparison:")
    print(f"  python scripts/compare_inference.py --pytorch-dir {output_dir}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch checkpoints for parity testing")
    parser.add_argument("--prompt", default="A golden retriever running through a meadow",
                        help="Text prompt for generation")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--height", type=int, default=128, help="Video height")
    parser.add_argument("--width", type=int, default=128, help="Video width")
    parser.add_argument("--frames", type=int, default=17, help="Number of frames")
    parser.add_argument("--steps", type=int, default=8, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--output-dir", default="/tmp/pytorch_parity_checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--weights", default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
                        help="Path to LTX-2 weights")
    parser.add_argument("--gemma-path", default="weights/gemma-3-12b",
                        help="Path to Gemma 3 model")

    args = parser.parse_args()

    generate_checkpoints(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        model_path=args.weights,
        gemma_path=args.gemma_path,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
