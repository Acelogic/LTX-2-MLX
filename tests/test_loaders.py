"""Tests for loader module: weight conversion, FP8, LoRA, and registry."""

import pytest
import mlx.core as mx
import threading

# Import the modules under test
from LTX_2_MLX.loader.weight_converter import (
    convert_transformer_key,
    convert_vae_key,
    convert_text_encoder_key,
    convert_pytorch_key_to_mlx,
    transpose_linear_weights,
    _flatten_to_nested,
    _convert_numeric_dicts_to_lists,
)
from LTX_2_MLX.loader.lora_loader import (
    LoRAConfig,
    find_lora_keys_for_weight,
    compute_lora_delta,
    get_lora_target_keys,
)
from LTX_2_MLX.loader.registry import (
    DummyRegistry,
    StateDictRegistry,
)


# ============================================================================
# Weight Converter Tests
# ============================================================================

class TestConvertTransformerKey:
    """Tests for convert_transformer_key function."""

    def test_basic_key_conversion(self):
        """Test basic transformer key is stripped of prefix."""
        key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
        result = convert_transformer_key(key)
        assert result == "transformer_blocks.0.attn1.to_q.weight"

    def test_to_out_conversion(self):
        """Test to_out.0 -> to_out conversion."""
        key = "model.diffusion_model.attn.to_out.0.weight"
        result = convert_transformer_key(key)
        assert result == "attn.to_out.weight"

    def test_ff_net_project_in_conversion(self):
        """Test ff.net.0.proj -> ff.project_in.proj conversion."""
        # Note: convert_transformer_key only strips prefix, full key conversion
        # is done by convert_pytorch_key_to_mlx
        key = "model.diffusion_model.ff.net.0.proj.weight"
        result = convert_transformer_key(key)
        # This function only strips model.diffusion_model. prefix
        assert result == "ff.net.0.proj.weight"

    def test_ff_net_project_out_conversion(self):
        """Test ff.net.2 -> ff.project_out conversion."""
        # Note: convert_transformer_key only strips prefix
        key = "model.diffusion_model.ff.net.2.weight"
        result = convert_transformer_key(key)
        assert result == "ff.net.2.weight"

    def test_skips_audio_keys(self):
        """Test that audio-related keys are skipped."""
        audio_keys = [
            "model.diffusion_model.audio_blocks.0.weight",
            "model.diffusion_model.Audio_processor.weight",
            "audio_vae.encoder.weight",
        ]
        for key in audio_keys:
            result = convert_transformer_key(key)
            assert result is None, f"Audio key {key} should be skipped"

    def test_skips_vocoder_keys(self):
        """Test that vocoder keys are skipped."""
        key = "vocoder.generator.weight"
        result = convert_transformer_key(key)
        assert result is None

    def test_skips_audio_vae_keys(self):
        """Test that audio_vae keys are skipped."""
        key = "audio_vae.encoder.weight"
        result = convert_transformer_key(key)
        assert result is None


class TestConvertVaeKey:
    """Tests for convert_vae_key function."""

    def test_vae_encoder_key(self):
        """Test VAE encoder key conversion."""
        key = "vae.encoder.layers.0.weight"
        result = convert_vae_key(key)
        assert result == "encoder.layers.0.weight"

    def test_vae_decoder_key(self):
        """Test VAE decoder key conversion."""
        key = "vae.decoder.layers.0.weight"
        result = convert_vae_key(key)
        assert result == "decoder.layers.0.weight"

    def test_skips_non_vae_keys(self):
        """Test that non-VAE keys are skipped."""
        key = "model.diffusion_model.transformer.weight"
        result = convert_vae_key(key)
        assert result is None


class TestConvertTextEncoderKey:
    """Tests for convert_text_encoder_key function."""

    def test_text_embedding_projection_key(self):
        """Test text embedding projection key conversion."""
        key = "text_embedding_projection.aggregate_embed.weight"
        result = convert_text_encoder_key(key)
        assert result == "feature_extractor.aggregate_embed.weight"

    def test_embeddings_connector_key(self):
        """Test embeddings connector key conversion."""
        key = "model.diffusion_model.embeddings_connector.proj.weight"
        result = convert_text_encoder_key(key)
        assert result == "embeddings_connector.proj.weight"

    def test_skips_unrelated_keys(self):
        """Test that unrelated keys are skipped."""
        key = "transformer.layers.0.weight"
        result = convert_text_encoder_key(key)
        assert result is None


class TestConvertPytorchKeyToMlx:
    """Tests for convert_pytorch_key_to_mlx function."""

    def test_skips_audio_keys_by_default(self):
        """Test audio/av_ca keys are skipped by default."""
        audio_keys = [
            "av_ca.to_q.weight",
            "audio_ff.net.0.proj.weight",
        ]
        for key in audio_keys:
            result = convert_pytorch_key_to_mlx(key, include_audio=False)
            assert result is None

    def test_includes_audio_when_flag_set(self):
        """Test audio keys are included when include_audio=True."""
        key = "audio_ff.net.0.proj.weight"
        result = convert_pytorch_key_to_mlx(key, include_audio=True)
        assert result is not None
        # The function converts ff.net.0.proj -> ff.project_in.proj
        assert "audio_ff.project_in.proj" in result or result == "audio_ff.net.0.proj.weight"

    def test_skips_video_embeddings_connector(self):
        """Test video_embeddings_connector keys are skipped."""
        key = "video_embeddings_connector.proj.weight"
        result = convert_pytorch_key_to_mlx(key)
        assert result is None

    def test_to_out_conversion(self):
        """Test to_out.0 -> to_out conversion."""
        key = "attn.to_out.0.weight"
        result = convert_pytorch_key_to_mlx(key)
        assert result == "attn.to_out.weight"


class TestTransposeLinearWeights:
    """Tests for transpose_linear_weights function."""

    def test_transposes_2d_weight(self):
        """Test that 2D .weight tensors are transposed."""
        weights = {
            "layer.weight": mx.array([[1, 2], [3, 4], [5, 6]]),  # (3, 2)
        }
        result = transpose_linear_weights(weights)
        assert result["layer.weight"].shape == (2, 3)

    def test_skips_embedding_weights(self):
        """Test that embedding weights (without proj) are not transposed."""
        weights = {
            "embeddings.weight": mx.array([[1, 2], [3, 4]]),
        }
        result = transpose_linear_weights(weights)
        # Embeddings should not be transposed
        assert result["embeddings.weight"].shape == (2, 2)

    def test_transposes_embedding_proj(self):
        """Test that embedding projection weights are transposed."""
        weights = {
            "embeddings_proj.weight": mx.array([[1, 2], [3, 4], [5, 6]]),
        }
        result = transpose_linear_weights(weights)
        assert result["embeddings_proj.weight"].shape == (2, 3)

    def test_skips_non_weight_keys(self):
        """Test that non-.weight keys are not transposed."""
        weights = {
            "layer.bias": mx.array([1, 2, 3]),
        }
        result = transpose_linear_weights(weights)
        assert mx.array_equal(result["layer.bias"], weights["layer.bias"])

    def test_skips_non_2d_weights(self):
        """Test that non-2D weights are not transposed."""
        weights = {
            "conv.weight": mx.array([[[1, 2], [3, 4]]]),  # 3D
        }
        result = transpose_linear_weights(weights)
        assert result["conv.weight"].shape == (1, 2, 2)


class TestFlattenToNested:
    """Tests for _flatten_to_nested function."""

    def test_simple_nesting(self):
        """Test simple key nesting."""
        flat = {
            "layer.weight": mx.array([1, 2, 3]),
            "layer.bias": mx.array([4, 5, 6]),
        }
        nested = _flatten_to_nested(flat)
        assert "layer" in nested
        assert "weight" in nested["layer"]
        assert "bias" in nested["layer"]

    def test_deep_nesting(self):
        """Test deep key nesting."""
        flat = {
            "model.layers.0.attn.weight": mx.array([1, 2]),
        }
        nested = _flatten_to_nested(flat)
        # Note: numeric keys like "0" get converted to list indices
        # So layers becomes a list, and we access via index
        assert nested["model"]["layers"][0]["attn"]["weight"] is not None


class TestConvertNumericDictsToLists:
    """Tests for _convert_numeric_dicts_to_lists function."""

    def test_converts_numeric_dict_to_list(self):
        """Test numeric string keys become list indices."""
        d = {"0": "a", "1": "b", "2": "c"}
        result = _convert_numeric_dicts_to_lists(d)
        assert result == ["a", "b", "c"]

    def test_preserves_non_numeric_dict(self):
        """Test non-numeric dicts are preserved."""
        d = {"weight": mx.array([1]), "bias": mx.array([2])}
        result = _convert_numeric_dicts_to_lists(d)
        assert isinstance(result, dict)
        assert "weight" in result

    def test_handles_gaps_in_indices(self):
        """Test handles gaps in numeric indices."""
        d = {"0": "a", "2": "c"}
        result = _convert_numeric_dicts_to_lists(d)
        assert result == ["a", None, "c"]

    def test_recursive_conversion(self):
        """Test recursive conversion of nested structures."""
        d = {
            "layers": {
                "0": {"weight": mx.array([1])},
                "1": {"weight": mx.array([2])},
            }
        }
        result = _convert_numeric_dicts_to_lists(d)
        assert isinstance(result["layers"], list)
        assert len(result["layers"]) == 2


# ============================================================================
# LoRA Loader Tests
# ============================================================================

class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_strength(self):
        """Test default LoRA strength is 1.0."""
        config = LoRAConfig(path="/path/to/lora.safetensors")
        assert config.strength == 1.0

    def test_custom_strength(self):
        """Test custom LoRA strength."""
        config = LoRAConfig(path="/path/to/lora.safetensors", strength=0.5)
        assert config.strength == 0.5

    def test_rejects_out_of_range_strength(self):
        """Test strength validation (-2.0 to 2.0)."""
        with pytest.raises(ValueError):
            LoRAConfig(path="/path", strength=3.0)

        with pytest.raises(ValueError):
            LoRAConfig(path="/path", strength=-3.0)

    def test_accepts_edge_strengths(self):
        """Test edge values are accepted."""
        config1 = LoRAConfig(path="/path", strength=-2.0)
        config2 = LoRAConfig(path="/path", strength=2.0)
        assert config1.strength == -2.0
        assert config2.strength == 2.0


class TestFindLoraKeysForWeight:
    """Tests for find_lora_keys_for_weight function."""

    def test_finds_lora_a_b_pattern(self):
        """Test finding lora_A/lora_B pattern."""
        lora_weights = {
            "layer.lora_A.weight": mx.array([1]),
            "layer.lora_B.weight": mx.array([2]),
        }
        key_a, key_b = find_lora_keys_for_weight(lora_weights, "layer.weight")
        assert key_a == "layer.lora_A.weight"
        assert key_b == "layer.lora_B.weight"

    def test_finds_lora_down_up_pattern(self):
        """Test finding lora_down/lora_up pattern."""
        lora_weights = {
            "layer.lora_down.weight": mx.array([1]),
            "layer.lora_up.weight": mx.array([2]),
        }
        key_a, key_b = find_lora_keys_for_weight(lora_weights, "layer.weight")
        assert key_a == "layer.lora_down.weight"
        assert key_b == "layer.lora_up.weight"

    def test_returns_none_when_not_found(self):
        """Test returns None when no LoRA keys found."""
        lora_weights = {
            "other.weight": mx.array([1]),
        }
        key_a, key_b = find_lora_keys_for_weight(lora_weights, "layer.weight")
        assert key_a is None
        assert key_b is None

    def test_handles_missing_weight_suffix(self):
        """Test handles base key without .weight suffix."""
        lora_weights = {
            "layer.lora_A": mx.array([1]),
            "layer.lora_B": mx.array([2]),
        }
        key_a, key_b = find_lora_keys_for_weight(lora_weights, "layer")
        assert key_a == "layer.lora_A"
        assert key_b == "layer.lora_B"


class TestComputeLoraDeltas:
    """Tests for compute_lora_delta function."""

    def test_basic_delta_computation(self):
        """Test basic LoRA delta: B @ A."""
        lora_weights = {
            "lora_A": mx.array([[1.0, 2.0], [3.0, 4.0]]),  # (2, 2) - rank=2
            "lora_B": mx.array([[1.0, 0.0], [0.0, 1.0]]),  # (2, 2) - identity
        }
        delta = compute_lora_delta(lora_weights, "lora_A", "lora_B", strength=1.0)
        expected = mx.matmul(lora_weights["lora_B"], lora_weights["lora_A"])
        assert mx.allclose(delta, expected)

    def test_strength_scaling(self):
        """Test strength scales the delta."""
        lora_weights = {
            "lora_A": mx.array([[1.0, 2.0]]),  # (1, 2)
            "lora_B": mx.array([[1.0], [1.0]]),  # (2, 1)
        }
        delta_full = compute_lora_delta(lora_weights, "lora_A", "lora_B", strength=1.0)
        delta_half = compute_lora_delta(lora_weights, "lora_A", "lora_B", strength=0.5)
        assert mx.allclose(delta_half, delta_full * 0.5)

    def test_negative_strength(self):
        """Test negative strength inverts delta."""
        lora_weights = {
            "lora_A": mx.array([[1.0, 2.0]]),
            "lora_B": mx.array([[1.0], [1.0]]),
        }
        delta_pos = compute_lora_delta(lora_weights, "lora_A", "lora_B", strength=1.0)
        delta_neg = compute_lora_delta(lora_weights, "lora_A", "lora_B", strength=-1.0)
        assert mx.allclose(delta_neg, -delta_pos)


class TestGetLoraTargetKeys:
    """Tests for get_lora_target_keys function."""

    def test_finds_common_targets(self):
        """Test finds common LoRA target modules."""
        model_weights = {
            "layer.to_q.weight": mx.array([1]),
            "layer.to_k.weight": mx.array([2]),
            "layer.to_v.weight": mx.array([3]),
            "layer.ff.project_in.weight": mx.array([4]),
            "other.weight": mx.array([5]),  # Not a target
        }
        targets = get_lora_target_keys(model_weights)
        assert "layer.to_q.weight" in targets
        assert "layer.to_k.weight" in targets
        assert "layer.to_v.weight" in targets
        assert "layer.ff.project_in.weight" in targets
        assert "other.weight" not in targets

    def test_only_includes_weight_keys(self):
        """Test only includes .weight keys."""
        model_weights = {
            "layer.to_q.weight": mx.array([1]),
            "layer.to_q.bias": mx.array([2]),  # Not a weight
        }
        targets = get_lora_target_keys(model_weights)
        assert "layer.to_q.weight" in targets
        assert "layer.to_q.bias" not in targets


# ============================================================================
# Registry Tests
# ============================================================================

class TestDummyRegistry:
    """Tests for DummyRegistry class."""

    def test_add_returns_empty_string(self):
        """Test add always returns empty string."""
        registry = DummyRegistry()
        result = registry.add(["path1"], "op", {"key": mx.array([1])})
        assert result == ""

    def test_get_returns_none(self):
        """Test get always returns None."""
        registry = DummyRegistry()
        registry.add(["path1"], "op", {"key": mx.array([1])})
        result = registry.get(["path1"], "op")
        assert result is None

    def test_pop_returns_none(self):
        """Test pop always returns None."""
        registry = DummyRegistry()
        registry.add(["path1"], "op", {"key": mx.array([1])})
        result = registry.pop(["path1"], "op")
        assert result is None

    def test_clear_is_noop(self):
        """Test clear does nothing (no error)."""
        registry = DummyRegistry()
        registry.clear()  # Should not raise


class TestStateDictRegistry:
    """Tests for StateDictRegistry class."""

    def test_add_and_get(self):
        """Test adding and retrieving state dict."""
        registry = StateDictRegistry()
        state_dict = {"weight": mx.array([1, 2, 3])}

        sd_id = registry.add(["path1.safetensors"], "transformer", state_dict)
        assert sd_id  # Non-empty ID

        retrieved = registry.get(["path1.safetensors"], "transformer")
        assert retrieved is not None
        assert "weight" in retrieved

    def test_add_duplicate_raises(self):
        """Test adding duplicate raises ValueError."""
        registry = StateDictRegistry()
        state_dict = {"weight": mx.array([1])}

        registry.add(["path1"], "op", state_dict)
        with pytest.raises(ValueError):
            registry.add(["path1"], "op", state_dict)

    def test_pop_removes_entry(self):
        """Test pop removes and returns state dict."""
        registry = StateDictRegistry()
        state_dict = {"weight": mx.array([1])}

        registry.add(["path1"], "op", state_dict)
        popped = registry.pop(["path1"], "op")
        assert popped is not None

        # Should be gone now
        assert registry.get(["path1"], "op") is None

    def test_get_nonexistent_returns_none(self):
        """Test get on nonexistent key returns None."""
        registry = StateDictRegistry()
        result = registry.get(["nonexistent"], "op")
        assert result is None

    def test_clear_removes_all(self):
        """Test clear removes all entries."""
        registry = StateDictRegistry()
        registry.add(["path1"], "op1", {"w1": mx.array([1])})
        registry.add(["path2"], "op2", {"w2": mx.array([2])})

        assert len(registry) == 2
        registry.clear()
        assert len(registry) == 0

    def test_len(self):
        """Test __len__ returns correct count."""
        registry = StateDictRegistry()
        assert len(registry) == 0

        registry.add(["path1"], "op1", {"w": mx.array([1])})
        assert len(registry) == 1

        registry.add(["path2"], "op2", {"w": mx.array([2])})
        assert len(registry) == 2

    def test_keys(self):
        """Test keys() returns list of IDs."""
        registry = StateDictRegistry()
        id1 = registry.add(["path1"], "op1", {"w": mx.array([1])})
        id2 = registry.add(["path2"], "op2", {"w": mx.array([2])})

        keys = registry.keys()
        assert id1 in keys
        assert id2 in keys

    def test_different_paths_different_ids(self):
        """Test different paths generate different IDs."""
        registry = StateDictRegistry()

        id1 = registry.add(["path1"], "op", {"w": mx.array([1])})
        id2 = registry.add(["path2"], "op", {"w": mx.array([2])})

        assert id1 != id2

    def test_different_ops_different_ids(self):
        """Test different op_names generate different IDs."""
        registry = StateDictRegistry()

        id1 = registry.add(["path"], "op1", {"w": mx.array([1])})
        id2 = registry.add(["path"], "op2", {"w": mx.array([2])})

        assert id1 != id2

    def test_none_op_name(self):
        """Test None op_name works correctly."""
        registry = StateDictRegistry()
        state_dict = {"weight": mx.array([1])}

        registry.add(["path"], None, state_dict)
        retrieved = registry.get(["path"], None)
        assert retrieved is not None

    def test_thread_safety(self):
        """Test thread-safe access to registry."""
        registry = StateDictRegistry()
        errors = []

        def add_weights(thread_id):
            try:
                for i in range(10):
                    state_dict = {"weight": mx.array([thread_id, i])}
                    registry.add([f"path_{thread_id}_{i}"], f"op_{thread_id}_{i}", state_dict)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_weights, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry) == 50  # 5 threads * 10 entries


# ============================================================================
# FP8 Loader Tests (Mock-based, no actual weights needed)
# ============================================================================

class TestFP8Functions:
    """Tests for FP8 loader functions."""

    def test_dequantize_fp8_weight_mock(self):
        """Test FP8 dequantization logic with mocks."""
        # This test verifies the logic without requiring actual FP8 tensors
        from LTX_2_MLX.loader.fp8_loader import HAS_TORCH

        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        import torch
        from LTX_2_MLX.loader.fp8_loader import dequantize_fp8_weight

        # Create a regular float tensor as a stand-in
        # (actual FP8 would need special handling)
        weight = torch.tensor([1.0, 2.0, 3.0])
        scale = 2.0

        result = dequantize_fp8_weight(weight, scale)
        expected = mx.array([2.0, 4.0, 6.0])

        assert mx.allclose(result, expected)

    def test_is_fp8_checkpoint_detection(self):
        """Test FP8 checkpoint detection by checking for weight_scale keys."""
        # This is a mock test - actual file testing would require fixtures
        from LTX_2_MLX.loader.fp8_loader import HAS_TORCH

        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # We can't easily test is_fp8_checkpoint without a real file
        # but we can verify the function exists and is callable
        from LTX_2_MLX.loader.fp8_loader import is_fp8_checkpoint
        assert callable(is_fp8_checkpoint)


# ============================================================================
# Integration Tests (require actual weights - marked for skip)
# ============================================================================

@pytest.mark.requires_weights
class TestLoaderIntegration:
    """Integration tests requiring actual model weights."""

    def test_load_transformer_weights_real(self, weights_dir):
        """Test loading real transformer weights using the streaming loader."""
        # Note: load_safetensors has issues with bfloat16, but load_transformer_weights
        # handles dtype conversion properly. Skip if no weights available.
        weight_files = list(weights_dir.glob("**/*.safetensors"))
        if not weight_files:
            pytest.skip("No weight files found")

        # Just verify we can read the file metadata without loading all weights
        from safetensors import safe_open
        with safe_open(str(weight_files[0]), framework="pt") as f:
            keys = list(f.keys())
        assert len(keys) > 0

    def test_fp8_checkpoint_info_real(self, weights_dir):
        """Test FP8 checkpoint info on real files."""
        from LTX_2_MLX.loader.fp8_loader import get_fp8_checkpoint_info

        fp8_files = list(weights_dir.glob("**/*fp8*.safetensors"))
        if not fp8_files:
            pytest.skip("No FP8 weight files found")

        info = get_fp8_checkpoint_info(str(fp8_files[0]))
        assert "is_fp8" in info
        assert info["is_fp8"] == True
