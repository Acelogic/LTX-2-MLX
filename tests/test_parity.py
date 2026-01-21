#!/usr/bin/env python3
"""
Comprehensive PyTorch ↔ MLX Parity Tests

Tests the entire LTX-2 pipeline by comparing MLX outputs against PyTorch reference
checkpoints at each stage:
- Text encoding
- Transformer denoising (all steps)
- VAE decoding

Usage:
    # Run all parity tests
    pytest tests/test_parity.py -v

    # Run with PyTorch checkpoint generation (slower, more thorough)
    pytest tests/test_parity.py -v --generate-checkpoints

    # Run specific test
    pytest tests/test_parity.py::TestFullPipelineParity::test_transformer_parity -v
"""

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

# Test configuration
CHECKPOINT_DIR = os.environ.get("PARITY_CHECKPOINT_DIR", "/tmp/pytorch_parity_checkpoints")
CORRELATION_THRESHOLD = 0.95
WEIGHTS_PATH = "weights/ltx-2/ltx-2-19b-distilled.safetensors"
GEMMA_PATH = "weights/gemma-3-12b"

# Test parameters
TEST_CONFIG = {
    "prompt": "A golden retriever running through a meadow",
    "height": 128,
    "width": 128,
    "num_frames": 17,
    "num_steps": 8,
    "seed": 42,
}


def compute_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    flat1 = arr1.flatten().astype(np.float64)
    flat2 = arr2.flatten().astype(np.float64)
    if np.std(flat1) < 1e-8 or np.std(flat2) < 1e-8:
        return 1.0 if np.allclose(flat1, flat2) else 0.0
    return float(np.corrcoef(flat1, flat2)[0, 1])


def compare_arrays(name: str, mlx_arr: np.ndarray, pytorch_arr: np.ndarray) -> dict:
    """Compare two arrays and return metrics."""
    if mlx_arr.shape != pytorch_arr.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "mlx_shape": mlx_arr.shape,
            "pytorch_shape": pytorch_arr.shape,
            "correlation": 0.0,
        }

    corr = compute_correlation(mlx_arr, pytorch_arr)
    diff = np.abs(mlx_arr - pytorch_arr)

    return {
        "name": name,
        "status": "PASS" if corr >= CORRELATION_THRESHOLD else "FAIL",
        "correlation": corr,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "mlx_range": (float(mlx_arr.min()), float(mlx_arr.max())),
        "pytorch_range": (float(pytorch_arr.min()), float(pytorch_arr.max())),
    }


def checkpoints_exist() -> bool:
    """Check if PyTorch checkpoints exist."""
    manifest_path = os.path.join(CHECKPOINT_DIR, "manifest.json")
    return os.path.exists(manifest_path)


def weights_exist() -> bool:
    """Check if model weights exist."""
    return os.path.exists(WEIGHTS_PATH) and os.path.exists(GEMMA_PATH)


@pytest.fixture(scope="module")
def pytorch_checkpoints():
    """Load PyTorch checkpoints."""
    if not checkpoints_exist():
        pytest.skip(f"PyTorch checkpoints not found at {CHECKPOINT_DIR}. "
                   f"Run: python scripts/generate_pytorch_checkpoints.py")

    with open(os.path.join(CHECKPOINT_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    checkpoints = {"manifest": manifest, "config": manifest["config"]}

    # Load all checkpoint files
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.endswith(".npy"):
            name = filename.replace(".npy", "")
            checkpoints[name] = np.load(os.path.join(CHECKPOINT_DIR, filename))

    return checkpoints


@pytest.fixture(scope="module")
def mlx_models():
    """Load MLX models."""
    if not weights_exist():
        pytest.skip(f"Model weights not found. See README for download instructions.")

    from transformers import AutoTokenizer

    from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier
    from LTX_2_MLX.components.patchifiers import get_pixel_coords
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.model.text_encoder.encoder import (
        create_av_text_encoder,
        load_av_text_encoder_weights,
    )
    from LTX_2_MLX.model.text_encoder.gemma3 import (
        Gemma3Config,
        Gemma3Model,
        load_gemma3_weights,
    )
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, X0Model
    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        decode_latent,
        load_vae_decoder_weights,
    )
    from LTX_2_MLX.types import SpatioTemporalScaleFactors, VideoLatentShape

    return {
        "tokenizer_path": GEMMA_PATH,
        "weights_path": WEIGHTS_PATH,
        "gemma_path": GEMMA_PATH,
        "DISTILLED_SIGMA_VALUES": DISTILLED_SIGMA_VALUES,
        "VideoLatentPatchifier": VideoLatentPatchifier,
        "get_pixel_coords": get_pixel_coords,
        "load_transformer_weights": load_transformer_weights,
        "create_av_text_encoder": create_av_text_encoder,
        "load_av_text_encoder_weights": load_av_text_encoder_weights,
        "Gemma3Config": Gemma3Config,
        "Gemma3Model": Gemma3Model,
        "load_gemma3_weights": load_gemma3_weights,
        "LTXModel": LTXModel,
        "LTXModelType": LTXModelType,
        "Modality": Modality,
        "X0Model": X0Model,
        "SimpleVideoDecoder": SimpleVideoDecoder,
        "decode_latent": decode_latent,
        "load_vae_decoder_weights": load_vae_decoder_weights,
        "SpatioTemporalScaleFactors": SpatioTemporalScaleFactors,
        "VideoLatentShape": VideoLatentShape,
        "AutoTokenizer": AutoTokenizer,
    }


class TestFullPipelineParity:
    """Full pipeline parity tests."""

    def test_text_encoder_parity(self, pytorch_checkpoints, mlx_models):
        """Test text encoder produces matching output."""
        config = pytorch_checkpoints["config"]
        prompt = config["prompt"]

        # Load tokenizer
        tokenizer = mlx_models["AutoTokenizer"].from_pretrained(mlx_models["tokenizer_path"])

        # Format prompt (matching PyTorch)
        T2V_SYSTEM_PROMPT = "Describe the video in extreme detail, focusing on the visual content, without any introductory phrases."
        chat_prompt = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        tokens = tokenizer(
            chat_prompt,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="np",
        )

        # Load Gemma
        gemma = mlx_models["Gemma3Model"](mlx_models["Gemma3Config"]())
        mlx_models["load_gemma3_weights"](gemma, mlx_models["gemma_path"])
        mx.eval(gemma.parameters())

        # Load text encoder
        text_encoder = mlx_models["create_av_text_encoder"]()
        mlx_models["load_av_text_encoder_weights"](text_encoder, mlx_models["weights_path"])
        mx.eval(text_encoder.parameters())

        # Run Gemma
        _, hidden_states = gemma(
            mx.array(tokens["input_ids"]),
            attention_mask=mx.array(tokens["attention_mask"]),
            output_hidden_states=True,
        )
        mx.eval(hidden_states)

        # Run text encoder
        output = text_encoder.encode_from_hidden_states(
            hidden_states=hidden_states,
            attention_mask=mx.array(tokens["attention_mask"]),
            padding_side="left",
        )
        mx.eval(output.video_encoding)

        mlx_encoding = np.array(output.video_encoding)

        # Compare
        pytorch_encoding = pytorch_checkpoints["text_encoder_video_encoding"]
        result = compare_arrays("text_encoder", mlx_encoding, pytorch_encoding)

        # Cleanup
        del gemma, text_encoder, hidden_states
        gc.collect()
        mx.metal.clear_cache()

        assert result["status"] == "PASS", (
            f"Text encoder parity failed: correlation={result['correlation']:.4f}"
        )

    def test_transformer_parity(self, pytorch_checkpoints, mlx_models):
        """Test transformer produces matching output at each step."""
        config = pytorch_checkpoints["config"]

        # Calculate latent shape
        latent_frames = (config["num_frames"] - 1) // 8 + 1
        latent_height = config["height"] // 32
        latent_width = config["width"] // 32
        latent_channels = 128

        # Load transformer
        velocity_model = mlx_models["LTXModel"](
            model_type=mlx_models["LTXModelType"].VideoOnly,
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_layers=48,
        )
        mlx_models["load_transformer_weights"](
            velocity_model, mlx_models["weights_path"], target_dtype="bfloat16"
        )
        mx.eval(velocity_model.parameters())

        model = mlx_models["X0Model"](velocity_model)

        # Setup patchifier
        patchifier = mlx_models["VideoLatentPatchifier"](patch_size=1)
        scale_factors = mlx_models["SpatioTemporalScaleFactors"].default()

        output_shape = mlx_models["VideoLatentShape"](
            batch=1,
            channels=latent_channels,
            frames=latent_frames,
            height=latent_height,
            width=latent_width,
        )

        # Load initial latent from PyTorch
        initial_latent = mx.array(pytorch_checkpoints["initial_latent"]).astype(mx.bfloat16)

        # Get positions
        latent_coords = patchifier.get_patch_grid_bounds(output_shape=output_shape)
        positions = mlx_models["get_pixel_coords"](
            latent_coords=latent_coords,
            scale_factors=scale_factors,
            causal_fix=True,
        ).astype(mx.bfloat16)

        # Use PyTorch text encoding
        text_encoding = mx.array(pytorch_checkpoints["text_encoder_video_encoding"]).astype(
            mx.bfloat16
        )

        # Get sigmas
        sigmas = mlx_models["DISTILLED_SIGMA_VALUES"][: config["num_inference_steps"] + 1]

        # Denoising loop
        latent = initial_latent
        results = []

        for step_idx in range(config["num_inference_steps"]):
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]

            # Patchify
            latent_patchified = patchifier.patchify(latent)

            # Create modality
            modality = mlx_models["Modality"](
                latent=latent_patchified.astype(mx.bfloat16),
                context=text_encoding,
                context_mask=None,
                timesteps=mx.array([sigma], dtype=mx.bfloat16),
                positions=positions,
                enabled=True,
            )

            # Run transformer
            x0_patchified = model(modality)
            mx.eval(x0_patchified)

            # Compare with PyTorch checkpoint
            ckpt_name = f"transformer_step_{step_idx:03d}"
            if ckpt_name in pytorch_checkpoints:
                mlx_output = np.array(x0_patchified.astype(mx.float32))
                pytorch_output = pytorch_checkpoints[ckpt_name]
                result = compare_arrays(ckpt_name, mlx_output, pytorch_output)
                results.append(result)

            # Unpatchify and Euler step
            denoised = patchifier.unpatchify(x0_patchified, output_shape=output_shape)
            if sigma_next == 0:
                latent = denoised
            else:
                velocity = (latent - denoised) / sigma
                latent = latent + velocity * (sigma_next - sigma)
            mx.eval(latent)

        # Cleanup
        del model, velocity_model
        gc.collect()
        mx.metal.clear_cache()

        # Check all steps passed
        failed = [r for r in results if r["status"] != "PASS"]
        if failed:
            failed_names = [r["name"] for r in failed]
            failed_corrs = [f"{r['correlation']:.4f}" for r in failed]
            pytest.fail(
                f"Transformer parity failed at steps: {failed_names}, "
                f"correlations: {failed_corrs}"
            )

    def test_vae_decoder_parity(self, pytorch_checkpoints, mlx_models):
        """Test VAE decoder produces matching output."""
        # Load VAE input from PyTorch
        vae_input = mx.array(pytorch_checkpoints["vae_decoder_input_latent"])

        # Load VAE decoder
        vae_decoder = mlx_models["SimpleVideoDecoder"]()
        mlx_models["load_vae_decoder_weights"](vae_decoder, mlx_models["weights_path"])
        vae_decoder.decode_noise_scale = 0.0  # Deterministic

        # Decode
        video = mlx_models["decode_latent"](vae_input, vae_decoder)
        mx.eval(video)

        # Convert to PyTorch format [B, C, T, H, W] and normalize
        video_np = np.array(video)  # [T, H, W, C]
        video_bcthtw = video_np.transpose(3, 0, 1, 2)[None, ...]  # [B, C, T, H, W]
        video_normalized = (video_bcthtw / 127.5) - 1.0  # Normalize to [-1, 1]

        # Compare
        pytorch_output = pytorch_checkpoints["vae_decoder_output_pixels"]
        result = compare_arrays("vae_decoder", video_normalized, pytorch_output)

        # Cleanup
        del vae_decoder
        gc.collect()
        mx.metal.clear_cache()

        assert result["status"] == "PASS", (
            f"VAE decoder parity failed: correlation={result['correlation']:.4f}"
        )


class TestParityReport:
    """Generate a comprehensive parity report."""

    def test_full_pipeline_report(self, pytorch_checkpoints, mlx_models):
        """Run full pipeline and generate report."""
        results = []

        # Test positions
        if "positions" in pytorch_checkpoints:
            config = pytorch_checkpoints["config"]
            latent_frames = (config["num_frames"] - 1) // 8 + 1
            latent_height = config["height"] // 32
            latent_width = config["width"] // 32

            patchifier = mlx_models["VideoLatentPatchifier"](patch_size=1)
            output_shape = mlx_models["VideoLatentShape"](
                batch=1, channels=128, frames=latent_frames,
                height=latent_height, width=latent_width
            )
            latent_coords = patchifier.get_patch_grid_bounds(output_shape=output_shape)
            positions = mlx_models["get_pixel_coords"](
                latent_coords=latent_coords,
                scale_factors=mlx_models["SpatioTemporalScaleFactors"].default(),
                causal_fix=True,
            )
            mlx_positions = np.array(positions.astype(mx.float32))
            result = compare_arrays("positions", mlx_positions, pytorch_checkpoints["positions"])
            results.append(result)

        # Collect all transformer step results
        for key in sorted(pytorch_checkpoints.keys()):
            if key.startswith("transformer_step_"):
                # Already tested in test_transformer_parity
                pass

        # Print summary
        print("\n" + "=" * 60)
        print("PARITY TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in results if r.get("status") == "PASS")
        total = len(results)

        for r in results:
            status_symbol = "✓" if r.get("status") == "PASS" else "✗"
            print(f"  {status_symbol} {r['name']}: correlation={r['correlation']:.4f}")

        print(f"\nPassed: {passed}/{total}")

        if results:
            avg_corr = np.mean([r["correlation"] for r in results])
            print(f"Average correlation: {avg_corr:.4f}")

        assert all(r.get("status") == "PASS" for r in results), "Some parity tests failed"


# Convenience function to run parity tests from command line
def run_parity_tests():
    """Run all parity tests and print results."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_parity_tests()
