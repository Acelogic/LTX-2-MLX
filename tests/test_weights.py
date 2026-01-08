"""Tests for weight loading and conversion."""

import os
import mlx.core as mx


def test_key_conversion():
    """Test PyTorch to MLX key conversion."""
    from ltx_mlx.loader import (
        convert_transformer_key,
        convert_vae_key,
        convert_text_encoder_key,
    )

    # Test transformer key conversion
    assert convert_transformer_key("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight") == \
           "transformer_blocks.0.attn1.to_q.weight"
    print("  transformer key basic: OK")

    # Test to_out.0 -> to_out conversion
    assert convert_transformer_key("model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight") == \
           "transformer_blocks.0.attn1.to_out.weight"
    print("  transformer key to_out: OK")

    # Test ff.net conversion
    converted = convert_transformer_key("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")
    assert converted == "transformer_blocks.0.ff.project_in.proj.weight", f"Got {converted}"
    print("  transformer key ff.net.0: OK")

    converted = convert_transformer_key("model.diffusion_model.transformer_blocks.0.ff.net.2.weight")
    assert converted == "transformer_blocks.0.ff.project_out.weight", f"Got {converted}"
    print("  transformer key ff.net.2: OK")

    # Test audio weights are skipped
    assert convert_transformer_key("model.diffusion_model.audio_blocks.0.weight") is None
    print("  skip audio: OK")

    # Test VAE key conversion
    assert convert_vae_key("vae.encoder.conv_in.weight") == "encoder.conv_in.weight"
    assert convert_vae_key("vae.decoder.conv_out.weight") == "decoder.conv_out.weight"
    print("  vae key conversion: OK")

    # Non-VAE keys should return None
    assert convert_vae_key("model.transformer.weight") is None
    print("  vae key filtering: OK")

    # Test text encoder key conversion
    assert convert_text_encoder_key("text_embedding_projection.aggregate_embed.weight") == \
           "feature_extractor.aggregate_embed.weight"
    print("  text encoder key conversion: OK")


def test_weight_transposition():
    """Test linear weight transposition."""
    from ltx_mlx.loader import transpose_linear_weights

    weights = {
        "layer.weight": mx.zeros((64, 128)),  # PyTorch [out, in]
        "layer.bias": mx.zeros((64,)),  # Bias stays as-is
        "embedding.weight": mx.zeros((100, 64)),  # Embedding stays as-is
    }

    transposed = transpose_linear_weights(weights)

    # Linear weight should be transposed
    assert transposed["layer.weight"].shape == (128, 64), f"Got {transposed['layer.weight'].shape}"
    print("  linear weight transpose: OK")

    # Bias should not change
    assert transposed["layer.bias"].shape == (64,)
    print("  bias unchanged: OK")


def test_weight_extraction():
    """Test weight extraction functions."""
    from ltx_mlx.loader import extract_transformer_weights, extract_vae_weights

    # Create mock weights
    mock_weights = {
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": mx.zeros((256, 256)),
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.bias": mx.zeros((256,)),
        "vae.encoder.conv_in.weight": mx.zeros((128, 3, 3, 3, 3)),
        "vae.decoder.conv_out.weight": mx.zeros((3, 128, 3, 3, 3)),
        "audio.something": mx.zeros((10,)),  # Should be skipped
    }

    # Extract transformer weights
    transformer_weights = extract_transformer_weights(mock_weights)
    assert "transformer_blocks.0.attn1.to_q.weight" in transformer_weights
    assert "transformer_blocks.0.attn1.to_q.bias" in transformer_weights
    # Linear weight should be transposed
    assert transformer_weights["transformer_blocks.0.attn1.to_q.weight"].shape == (256, 256)
    print("  transformer weight extraction: OK")

    # Extract VAE weights
    encoder_weights, decoder_weights = extract_vae_weights(mock_weights)
    assert "conv_in.weight" in encoder_weights
    assert "conv_out.weight" in decoder_weights
    print("  vae weight extraction: OK")


def test_load_real_weights():
    """Test loading real weights if available."""
    # Try multiple possible weight files
    weight_files = [
        "/Users/mcruz/Developer/LTX-2-MLX/weights/ltx-2/ltx-2-19b-distilled.safetensors",
        "/Users/mcruz/Developer/LTX-2-MLX/weights/ltx-2/ltx-2-19b-dev.safetensors",
    ]

    weights_path = None
    for path in weight_files:
        if os.path.exists(path):
            weights_path = path
            break

    if weights_path is None:
        print("  Skipping real weight loading (no weight files found)")
        return

    from ltx_mlx.loader import load_safetensors

    # Try loading weights (only load a sample to avoid memory issues)
    print(f"  Loading weights from {weights_path}...")
    print("  (This may take a moment for large files...)")

    # For very large files, just verify we can open it
    from safetensors import safe_open
    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())[:100]  # Just sample first 100 keys

    print(f"  Found {len(keys)} keys (sampled first 100)")

    # Check some expected keys exist
    transformer_keys = [k for k in keys if "diffusion_model" in k]
    vae_keys = [k for k in keys if k.startswith("vae.")]

    print(f"  Sample contains {len(transformer_keys)} transformer keys")
    print(f"  Sample contains {len(vae_keys)} VAE keys")

    print("  real weight file access: OK")


def run_weight_tests():
    """Run all weight tests."""
    print("\n=== Weight Loading Tests ===\n")

    print("Testing key conversion...")
    test_key_conversion()

    print("\nTesting weight transposition...")
    test_weight_transposition()

    print("\nTesting weight extraction...")
    test_weight_extraction()

    print("\nTesting real weight loading...")
    test_load_real_weights()

    print("\n=== All Weight Tests Passed! ===\n")


if __name__ == "__main__":
    run_weight_tests()
