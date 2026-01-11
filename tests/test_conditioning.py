"""Unit tests for LTX-2 conditioning logic.

Tests conditioning items (keyframe and latent-index) without requiring model weights.

Run with: pytest tests/test_conditioning.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.components.patchifiers import VideoLatentPatchifier
from LTX_2_MLX.conditioning.keyframe import VideoConditionByKeyframeIndex
from LTX_2_MLX.conditioning.latent import VideoConditionByLatentIndex, ConditioningError
from LTX_2_MLX.conditioning.tools import VideoLatentTools
from LTX_2_MLX.types import LatentState, VideoLatentShape


class TestVideoConditionByLatentIndex:
    """Test VideoConditionByLatentIndex conditioning."""

    def test_basic_conditioning(self):
        """Test basic latent index conditioning."""
        # Create target shape
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        # Create tools
        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        # Create initial state
        initial_state = tools.create_initial_state(dtype=mx.float32)

        # Create conditioning latent (3 frames starting at index 0)
        cond_latent = mx.random.normal((1, 128, 3, 8, 8))
        conditioning = VideoConditionByLatentIndex(
            latent=cond_latent,
            strength=0.5,
            latent_idx=0,
        )

        # Apply conditioning
        conditioned_state = conditioning.apply_to(initial_state, tools)

        # Verify shape is preserved
        assert conditioned_state.latent.shape == initial_state.latent.shape
        assert conditioned_state.denoise_mask.shape == initial_state.denoise_mask.shape
        assert conditioned_state.positions.shape == initial_state.positions.shape

    def test_conditioning_strength(self):
        """Test that conditioning strength affects denoise mask correctly."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)
        cond_latent = mx.random.normal((1, 128, 1, 8, 8))

        # Test different strengths
        for strength in [0.0, 0.5, 1.0]:
            conditioning = VideoConditionByLatentIndex(
                latent=cond_latent,
                strength=strength,
                latent_idx=0,
            )

            conditioned_state = conditioning.apply_to(initial_state, tools)

            # Get the conditioning region mask
            tokens = patchifier.patchify(cond_latent)
            num_tokens = tokens.shape[1]

            # Check denoise mask values in conditioned region
            mask_values = np.array(conditioned_state.denoise_mask[0, :num_tokens, 0])
            expected_mask = 1.0 - strength

            np.testing.assert_allclose(
                mask_values,
                expected_mask,
                rtol=1e-6,
                err_msg=f"Denoise mask should be {expected_mask} for strength={strength}",
            )

    def test_conditioning_at_different_indices(self):
        """Test conditioning at different latent frame indices."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)
        cond_latent = mx.random.normal((1, 128, 1, 8, 8))

        # Test conditioning at different indices
        for latent_idx in [0, 4, 8]:
            conditioning = VideoConditionByLatentIndex(
                latent=cond_latent,
                strength=0.5,
                latent_idx=latent_idx,
            )

            # Should not raise error
            conditioned_state = conditioning.apply_to(initial_state, tools)
            assert conditioned_state.latent.shape == initial_state.latent.shape

    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises ConditioningError."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)

        # Create conditioning with wrong spatial dimensions
        cond_latent = mx.random.normal((1, 128, 1, 16, 16))  # Wrong H, W
        conditioning = VideoConditionByLatentIndex(
            latent=cond_latent,
            strength=0.5,
            latent_idx=0,
        )

        # Should raise ConditioningError
        with pytest.raises(ConditioningError, match="Cannot apply image conditioning"):
            conditioning.apply_to(initial_state, tools)

    def test_bounds_checking(self):
        """Test that out-of-bounds conditioning raises ValueError."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)

        # Create conditioning that exceeds bounds
        cond_latent = mx.random.normal((1, 128, 5, 8, 8))
        conditioning = VideoConditionByLatentIndex(
            latent=cond_latent,
            strength=0.5,
            latent_idx=6,  # 6 + 5 = 11 > 9 frames
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Conditioning tokens exceed latent sequence length"):
            conditioning.apply_to(initial_state, tools)

    def test_clean_latent_replacement(self):
        """Test that clean_latent is also updated with conditioning."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)
        cond_latent = mx.random.normal((1, 128, 1, 8, 8))
        conditioning = VideoConditionByLatentIndex(
            latent=cond_latent,
            strength=0.5,
            latent_idx=0,
        )

        conditioned_state = conditioning.apply_to(initial_state, tools)

        # clean_latent should be updated
        assert conditioned_state.clean_latent.shape == initial_state.clean_latent.shape

        # The conditioned region in latent and clean_latent should match
        tokens = patchifier.patchify(cond_latent)
        num_tokens = tokens.shape[1]

        np.testing.assert_allclose(
            np.array(conditioned_state.latent[0, :num_tokens]),
            np.array(conditioned_state.clean_latent[0, :num_tokens]),
            rtol=1e-6,
            err_msg="Conditioned region should match in latent and clean_latent",
        )


class TestVideoConditionByKeyframeIndex:
    """Test VideoConditionByKeyframeIndex conditioning."""

    def test_basic_keyframe_conditioning(self):
        """Test basic keyframe conditioning."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)

        # Create keyframe (single frame)
        keyframe = mx.random.normal((1, 128, 1, 8, 8))
        conditioning = VideoConditionByKeyframeIndex(
            keyframes=keyframe,
            frame_idx=0,
            strength=0.5,
        )

        # Apply conditioning
        conditioned_state = conditioning.apply_to(initial_state, tools)

        # Keyframe conditioning appends tokens, so output should be larger
        assert conditioned_state.latent.shape[1] > initial_state.latent.shape[1]
        assert conditioned_state.denoise_mask.shape[1] > initial_state.denoise_mask.shape[1]

    def test_keyframe_strength(self):
        """Test that keyframe strength affects denoise mask."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)
        keyframe = mx.random.normal((1, 128, 1, 8, 8))

        # Test different strengths
        for strength in [0.0, 0.5, 1.0]:
            conditioning = VideoConditionByKeyframeIndex(
                keyframes=keyframe,
                frame_idx=0,
                strength=strength,
            )

            conditioned_state = conditioning.apply_to(initial_state, tools)

            # Get the appended keyframe tokens
            initial_num_tokens = initial_state.latent.shape[1]
            keyframe_mask = conditioned_state.denoise_mask[0, initial_num_tokens:, 0]

            expected_mask = 1.0 - strength
            np.testing.assert_allclose(
                np.array(keyframe_mask),
                expected_mask,
                rtol=1e-6,
                err_msg=f"Keyframe denoise mask should be {expected_mask} for strength={strength}",
            )

    def test_keyframe_at_different_indices(self):
        """Test keyframe conditioning at different frame indices."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)
        keyframe = mx.random.normal((1, 128, 1, 8, 8))

        # Test conditioning at different frame indices
        for frame_idx in [0, 4, 8]:
            conditioning = VideoConditionByKeyframeIndex(
                keyframes=keyframe,
                frame_idx=frame_idx,
                strength=0.5,
            )

            # Should not raise error
            conditioned_state = conditioning.apply_to(initial_state, tools)
            assert conditioned_state.latent.shape[1] > initial_state.latent.shape[1]

    def test_keyframe_positions(self):
        """Test that keyframe positions are correctly offset."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        initial_state = tools.create_initial_state(dtype=mx.float32)
        keyframe = mx.random.normal((1, 128, 1, 8, 8))

        frame_idx = 4
        conditioning = VideoConditionByKeyframeIndex(
            keyframes=keyframe,
            frame_idx=frame_idx,
            strength=0.5,
        )

        conditioned_state = conditioning.apply_to(initial_state, tools)

        # Check that positions tensor has been extended
        assert conditioned_state.positions.shape[2] > initial_state.positions.shape[2]

    def test_multiple_keyframes(self):
        """Test conditioning with multiple keyframes applied sequentially."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        state = tools.create_initial_state(dtype=mx.float32)

        # Apply two keyframes
        keyframe1 = mx.random.normal((1, 128, 1, 8, 8))
        keyframe2 = mx.random.normal((1, 128, 1, 8, 8))

        conditioning1 = VideoConditionByKeyframeIndex(
            keyframes=keyframe1,
            frame_idx=0,
            strength=0.5,
        )
        conditioning2 = VideoConditionByKeyframeIndex(
            keyframes=keyframe2,
            frame_idx=8,
            strength=0.5,
        )

        state = conditioning1.apply_to(state, tools)
        state = conditioning2.apply_to(state, tools)

        # Should have tokens from both keyframes
        initial_num_tokens = tools.create_initial_state(dtype=mx.float32).latent.shape[1]
        keyframe_tokens = patchifier.patchify(keyframe1).shape[1]

        expected_tokens = initial_num_tokens + 2 * keyframe_tokens
        assert state.latent.shape[1] == expected_tokens


class TestVideoLatentTools:
    """Test VideoLatentTools utility class."""

    def test_create_initial_state(self):
        """Test creating initial latent state."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        state = tools.create_initial_state(dtype=mx.float32)

        # Verify shapes
        assert state.latent.dtype == mx.float32
        assert state.denoise_mask.dtype == mx.float32
        assert state.positions.dtype == mx.float32

        # Latent should be patchified
        expected_tokens = patchifier.get_token_count(target_shape)
        assert state.latent.shape[1] == expected_tokens

    def test_create_initial_state_with_custom_latent(self):
        """Test creating state with custom initial latent."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        # Create custom latent
        custom_latent = mx.random.normal((1, 128, 9, 8, 8))

        state = tools.create_initial_state(dtype=mx.float32, initial_latent=custom_latent)

        # Should use custom latent
        assert state.latent.shape[1] == patchifier.get_token_count(target_shape)

    def test_clear_conditioning(self):
        """Test clearing conditioning from state."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        # Create state and add keyframe
        state = tools.create_initial_state(dtype=mx.float32)
        keyframe = mx.random.normal((1, 128, 1, 8, 8))
        conditioning = VideoConditionByKeyframeIndex(
            keyframes=keyframe,
            frame_idx=0,
            strength=0.5,
        )
        conditioned_state = conditioning.apply_to(state, tools)

        # Clear conditioning (should remove appended tokens)
        cleared_state = tools.clear_conditioning(conditioned_state)

        # Should have original token count
        expected_tokens = patchifier.get_token_count(target_shape)
        assert cleared_state.latent.shape[1] == expected_tokens

    def test_unpatchify(self):
        """Test unpatchifying latent state."""
        target_shape = VideoLatentShape(
            batch=1,
            channels=128,
            frames=9,
            height=8,
            width=8,
        )

        patchifier = VideoLatentPatchifier(patch_size=1)
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=target_shape,
            fps=24.0,
        )

        state = tools.create_initial_state(dtype=mx.float32)
        unpatchified = tools.unpatchify(state)

        # Should have spatial shape
        assert unpatchified.latent.shape == target_shape.to_tuple()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
