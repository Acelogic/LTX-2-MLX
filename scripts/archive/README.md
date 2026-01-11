# Archived Debug and Development Scripts

This directory contains debug and comparison scripts that were used during the development and debugging of LTX-2-MLX. These scripts were instrumental in diagnosing and fixing critical issues but are no longer needed for regular use.

## Purpose

These scripts helped identify and fix several major issues:

1. **Text Embedding Homogenization** - Scripts revealed that Gemma's self-attention was destroying differentiation between prompts
2. **Velocity vs X0 Prediction Mismatch** - Debugging showed the transformer output velocity but denoising expected X0
3. **Dark VAE Output** - Diagnostics revealed missing timestep conditioning in the VAE decoder
4. **Two-Stage Pipeline Gray Noise** - Scripts showed variance collapse requiring normalization wrapper
5. **Transformer Correctness** - Comparison scripts verified MLX implementation matches PyTorch exactly

## Script Categories

### Text Encoding Validation (8 scripts)
- `compare_text_embeddings.py` - Compare MLX vs PyTorch embeddings
- `debug_text_encoder.py` - Text encoder statistics
- `debug_embedding_stages.py` - Trace differentiation loss through layers
- `debug_text_differentiation.py` - Compare prompt embeddings
- `debug_mask.py` - Attention mask handling
- `test_with_gemma.py` - Test denoising with real Gemma embeddings

### Transformer/Diffusion Process Debugging (15 scripts)
- `debug_nan.py` - Check for NaN/Inf in full pipeline
- `debug_velocity.py` - Verify velocity vs X0 prediction
- `debug_denoising.py` - Trace denoising loop convergence
- `debug_pipeline_step.py` - Debug single pipeline step
- `debug_single_step.py` - Test single denoising step
- `debug_full_loop.py` - Full denoising loop test
- `debug_scheduler.py` - Sigma schedule validation
- `debug_latent_distribution.py` - Latent statistics
- `compare_denoising_step.py` - MLX vs PyTorch comparison
- `compare_denoising_loop.py` - Compare full loops
- `compare_full_forward.py` - Full transformer forward pass
- `compare_all_blocks.py` - Block-by-block comparison
- `compare_block_forward.py` - Single block comparison
- `bisect_divergence.py` - Binary search for divergence
- `analyze_transformer_output.py` - Output statistics

### VAE/Decoder Debugging (8 scripts)
- `debug_vae.py` - VAE decoder issues
- `debug_adaln.py` - AdaLN conditioning verification
- `validate_vae_decoder.py` - VAE decoder validation
- `validate_vae_encoder.py` - VAE encoder validation
- `test_vae_roundtrip.py` - Encoder/decoder roundtrip
- `test_latent_normalization.py` - Normalization verification
- `trace_decoder.py` - Trace decoder computation

### Pipeline and Model Variant Testing (12 scripts)
- `test_distilled_pipeline.py` - Two-stage distilled pipeline
- `test_dev_model.py` - Non-distilled dev model (25 steps)
- `test_pytorch_pipeline.py` - PyTorch reference pipeline
- `debug_twostage.py` - Two-stage pipeline debugging
- `debug_twostage_v2.py` - Two-stage v2 debugging
- `interpolate.py` - Frame interpolation
- `upscale_temporal.py` - Temporal upscaling standalone

### Verification and Comparison (7 scripts)
- `compare_pytorch.py` - MLX vs PyTorch transformer blocks
- `compare_with_pytorch.py` - Save outputs for PyTorch comparison
- `compare_pixel_coords.py` - Position encoding validation
- `verify_attention_weights.py` - Attention weight inspection

## Status

All critical issues identified by these scripts have been fixed:
- ✅ Text embedding pipeline corrected (using all 49 Gemma layers)
- ✅ Velocity prediction properly handled in denoising
- ✅ VAE decoder includes timestep conditioning and bias correction
- ✅ Two-stage pipeline uses normalization wrapper around upsampling
- ✅ RoPE encoding changed from INTERLEAVED to SPLIT
- ✅ Spatial upscaler res-block instability fixed with output normalization

## Usage

These scripts are kept for historical reference and may be useful for:
- Understanding how specific bugs were diagnosed
- Debugging similar issues in the future
- Converting to formal unit tests
- Reference for PyTorch comparison methodology

**Note**: Many of these scripts may have dependencies on older code versions or may require PyTorch installation for comparison features.
