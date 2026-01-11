
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from LTX_2_MLX.model.transformer import (
    LTXModel,
    LTXAVModel,
    LTXModelType,
    LTXRopeType,
    BasicAVTransformerBlock,
    BasicTransformerBlock,
)

class TestParityStructure(unittest.TestCase):
    def test_ltx_model_defaults(self):
        """Verify LTXModel defaults match PyTorch parity."""
        model = LTXModel(model_type=LTXModelType.VideoOnly, num_layers=2) # 2 layers for speed
        
        # Check RoPE default
        self.assertEqual(model.rope_type, LTXRopeType.SPLIT, "Default RoPE type must be SPLIT for current weights")
        
        # Check Block type
        self.assertIsInstance(model.transformer_blocks[0], BasicAVTransformerBlock, "Must use BasicAVTransformerBlock")
        
        # Check Audio components not present/none for VideoOnly
        self.assertFalse(hasattr(model, 'audio_patchify_proj'), "VideoOnly model should not have audio_patchify_proj")

    def test_ltx_av_model_alias(self):
        """Verify LTXAVModel alias works and defaults."""
        model = LTXAVModel(model_type=LTXModelType.AudioVideo, num_layers=2)
        
        # Check RoPE default
        self.assertEqual(model.rope_type, LTXRopeType.SPLIT)
        
        # Check Audio components present
        self.assertTrue(hasattr(model, 'audio_patchify_proj'))
        
    def test_block_optimizations(self):
        """Verify BasicAVTransformerBlock has optimizations."""
        block = BasicAVTransformerBlock(idx=0)
        # Check if cross_attn_scale is attribute
        self.assertTrue(hasattr(block, 'cross_attn_scale'))
        
        # Check methods
        # Note: can't easily check for @mx.compile without inspecting, but we can check it runs
        
if __name__ == '__main__':
    unittest.main()
