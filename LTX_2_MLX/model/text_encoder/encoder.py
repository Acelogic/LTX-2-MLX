"""Video Gemma Text Encoder for LTX-2."""

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .connector import Embeddings1DConnector
from .feature_extractor import GemmaFeaturesExtractorProjLinear


@dataclass
class VideoGemmaEncoderOutput:
    """Output from the video Gemma encoder."""

    video_encoding: mx.array  # Shape: [B, T, D]
    attention_mask: mx.array  # Shape: [B, T]


@dataclass
class AudioVideoGemmaEncoderOutput:
    """Output from the audio+video Gemma encoder.

    Contains separate encodings for video and audio modalities.
    Mirrors PyTorch's AVGemmaEncoderOutput from ltx_core/text_encoders/gemma/encoders/av_encoder.py.
    """

    video_encoding: mx.array  # Shape: [B, T, D] - encoding for video modality
    audio_encoding: mx.array  # Shape: [B, T, D] - encoding for audio modality
    attention_mask: mx.array  # Shape: [B, T]


class CaptionProjection(nn.Module):
    """
    Project text embeddings from Gemma dimension to transformer cross-attention dimension.

    This is a 2-layer MLP that converts from 3840 (Gemma) to 4096 (transformer).

    DEPRECATED: This class is kept for backwards compatibility but is no longer used
    in the text encoder. Caption projection is now handled by the transformer model's
    PixArtAlphaTextProjection, matching PyTorch LTX-2 architecture.
    """

    def __init__(
        self,
        in_features: int = 3840,
        hidden_features: int = 4096,
        out_features: int = 4096,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_features)
        self.linear_2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        """Project embeddings."""
        x = self.linear_1(x)
        x = nn.silu(x)
        x = self.linear_2(x)
        return x


class VideoGemmaTextEncoderModel(nn.Module):
    """
    Video Gemma Text Encoder Model.

    This model processes text prompts through:
    1. Gemma language model (external, via mlx-lm)
    2. Feature extractor (projects multi-layer hidden states)
    3. Embeddings connector (1D transformer refinement)

    The output is 3840-dim embeddings suitable for the diffusion transformer,
    which applies its own caption projection (3840 → 4096) for cross-attention.

    Note: The Gemma model itself is loaded separately via mlx-lm.
    This class handles the LTX-2 specific projection layers.

    Note: Caption projection (3840 → 4096) is handled by the transformer model,
    not this text encoder. This matches PyTorch LTX-2 architecture where the
    text encoder returns 3840-dim features and the diffusion model contains
    the caption_projection layer.
    """

    def __init__(
        self,
        feature_extractor: Optional[GemmaFeaturesExtractorProjLinear] = None,
        embeddings_connector: Optional[Embeddings1DConnector] = None,
    ):
        """
        Initialize text encoder.

        Args:
            feature_extractor: Gemma feature extractor for hidden state projection.
            embeddings_connector: 1D connector for sequence refinement.
        """
        super().__init__()

        self.feature_extractor = feature_extractor or GemmaFeaturesExtractorProjLinear()
        self.embeddings_connector = embeddings_connector or Embeddings1DConnector()

    def _convert_to_additive_mask(
        self,
        attention_mask: mx.array,
        dtype: mx.Dtype = mx.float32,
    ) -> mx.array:
        """
        Convert binary attention mask to additive mask for softmax.

        Args:
            attention_mask: Binary mask where 1 = attend, 0 = don't attend.
            dtype: Output dtype.

        Returns:
            Additive mask where 0 = attend, large negative = don't attend.
        """
        # (mask - 1) makes 1 -> 0, 0 -> -1
        # Multiply by large value to get 0 or -large_value
        # Use dtype-appropriate values matching transformer's attention mask scaling
        if dtype == mx.float16:
            large_value = 65504.0  # finfo(fp16).max - matches transformer
        elif dtype == mx.bfloat16:
            large_value = 3.38e38  # finfo(bfloat16).max
        else:
            large_value = 3.40e38  # finfo(fp32).max
        additive_mask = (attention_mask.astype(dtype) - 1) * large_value
        # Reshape for attention: [B, 1, 1, T]
        additive_mask = additive_mask.reshape(
            attention_mask.shape[0], 1, 1, attention_mask.shape[-1]
        )
        return additive_mask

    def encode_from_hidden_states(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> VideoGemmaEncoderOutput:
        """
        Encode text from pre-computed Gemma hidden states.

        This is the main entry point when using mlx-lm to run Gemma
        and extracting hidden states from all layers.

        Args:
            hidden_states: List of hidden states from each Gemma layer.
                Each tensor has shape [B, T, 3840].
            attention_mask: Binary attention mask [B, T].
            padding_side: Side where padding was applied.

        Returns:
            VideoGemmaEncoderOutput with encoded text (3840 dim).
            Note: Caption projection to 4096 is handled by the transformer.
        """
        # Extract features from hidden states
        encoded = self.feature_extractor.extract_from_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            padding_side=padding_side,
        )

        # Convert mask to additive format
        connector_mask = self._convert_to_additive_mask(attention_mask, encoded.dtype)

        # Process through connector
        encoded, output_mask = self.embeddings_connector(encoded, connector_mask)

        # Note: Caption projection (3840 → 4096) is handled by the transformer model,
        # not here. This matches PyTorch where text encoder returns 3840-dim features.

        # Convert mask back to binary for output
        # output_mask is additive: 0 = attend, negative = don't attend
        binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)

        # Apply mask to zero out padded positions
        encoded = encoded * binary_mask[:, :, None]

        return VideoGemmaEncoderOutput(
            video_encoding=encoded,
            attention_mask=binary_mask,
        )

    def encode_projected(
        self,
        projected_features: mx.array,
        attention_mask: mx.array,
    ) -> VideoGemmaEncoderOutput:
        """
        Encode from already-projected features.

        Use this when feature extraction has already been done.

        Args:
            projected_features: Pre-projected features [B, T, 3840].
            attention_mask: Binary attention mask [B, T].

        Returns:
            VideoGemmaEncoderOutput with encoded text (3840 dim).
            Note: Caption projection to 4096 is handled by the transformer.
        """
        # Convert mask to additive format
        connector_mask = self._convert_to_additive_mask(
            attention_mask, projected_features.dtype
        )

        # Process through connector
        encoded, output_mask = self.embeddings_connector(
            projected_features, connector_mask
        )

        # Note: Caption projection (3840 → 4096) is handled by the transformer model

        # Convert mask back to binary
        binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)

        # Apply mask
        encoded = encoded * binary_mask[:, :, None]

        return VideoGemmaEncoderOutput(
            video_encoding=encoded,
            attention_mask=binary_mask,
        )

    def __call__(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> VideoGemmaEncoderOutput:
        """
        Forward pass.

        Args:
            hidden_states: List of Gemma hidden states from all layers.
            attention_mask: Binary attention mask.
            padding_side: Side where padding was applied.

        Returns:
            VideoGemmaEncoderOutput with encoded text.
        """
        return self.encode_from_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            padding_side=padding_side,
        )


class AudioVideoGemmaTextEncoderModel(nn.Module):
    """
    Audio+Video Gemma Text Encoder Model.

    This model processes text prompts through:
    1. Gemma language model (external, via mlx-lm)
    2. Feature extractor (projects multi-layer hidden states)
    3. Two separate embeddings connectors: one for video, one for audio

    Mirrors PyTorch's AVGemmaTextEncoderModel from ltx_core/text_encoders/gemma/encoders/av_encoder.py.
    """

    def __init__(
        self,
        feature_extractor: Optional[GemmaFeaturesExtractorProjLinear] = None,
        embeddings_connector: Optional[Embeddings1DConnector] = None,
        audio_embeddings_connector: Optional[Embeddings1DConnector] = None,
    ):
        """
        Initialize audio+video text encoder.

        Args:
            feature_extractor: Gemma feature extractor for hidden state projection.
            embeddings_connector: 1D connector for video sequence refinement.
            audio_embeddings_connector: 1D connector for audio sequence refinement.
        """
        super().__init__()

        self.feature_extractor = feature_extractor or GemmaFeaturesExtractorProjLinear()
        self.embeddings_connector = embeddings_connector or Embeddings1DConnector()
        self.audio_embeddings_connector = audio_embeddings_connector or Embeddings1DConnector()

    def _convert_to_additive_mask(
        self,
        attention_mask: mx.array,
        dtype: mx.Dtype = mx.float32,
    ) -> mx.array:
        """Convert binary attention mask to additive mask for softmax."""
        # Use dtype-appropriate values matching transformer's attention mask scaling
        if dtype == mx.float16:
            large_value = 65504.0  # finfo(fp16).max - matches transformer
        elif dtype == mx.bfloat16:
            large_value = 3.38e38  # finfo(bfloat16).max
        else:
            large_value = 3.40e38  # finfo(fp32).max
        additive_mask = (attention_mask.astype(dtype) - 1) * large_value
        additive_mask = additive_mask.reshape(
            attention_mask.shape[0], 1, 1, attention_mask.shape[-1]
        )
        return additive_mask

    def encode_from_hidden_states(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> AudioVideoGemmaEncoderOutput:
        """
        Encode text from pre-computed Gemma hidden states.

        Args:
            hidden_states: List of hidden states from each Gemma layer.
            attention_mask: Binary attention mask [B, T].
            padding_side: Side where padding was applied.

        Returns:
            AudioVideoGemmaEncoderOutput with separate video and audio encodings.
        """
        # Extract features from hidden states
        encoded_input = self.feature_extractor.extract_from_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            padding_side=padding_side,
        )

        # Convert mask to additive format
        connector_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        # Process through video connector
        video_encoded, output_mask = self.embeddings_connector(encoded_input, connector_mask)

        # Convert mask back to binary
        binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)
        binary_mask_expanded = binary_mask[:, :, None]

        # Apply mask to video encoding
        video_encoded = video_encoded * binary_mask_expanded

        # Process through audio connector (uses same input features)
        audio_encoded, _ = self.audio_embeddings_connector(encoded_input, connector_mask)
        # Note: audio uses same mask

        return AudioVideoGemmaEncoderOutput(
            video_encoding=video_encoded,
            audio_encoding=audio_encoded,
            attention_mask=binary_mask,
        )

    def __call__(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> AudioVideoGemmaEncoderOutput:
        """Forward pass."""
        return self.encode_from_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            padding_side=padding_side,
        )


def create_text_encoder(
    hidden_dim: int = 3840,
    num_gemma_layers: int = 49,
    connector_heads: int = 30,
    connector_head_dim: int = 128,
    connector_layers: int = 2,
    num_registers: int = 128,
) -> VideoGemmaTextEncoderModel:
    """
    Create a text encoder with default LTX-2 configuration.

    Args:
        hidden_dim: Gemma hidden dimension (3840).
        num_gemma_layers: Number of Gemma layers (49).
        connector_heads: Number of attention heads in connector (30).
        connector_head_dim: Head dimension in connector (128).
        connector_layers: Number of transformer layers in connector (2).
        num_registers: Number of learnable registers (128).

    Returns:
        Configured VideoGemmaTextEncoderModel.
        Note: Output is 3840-dim. Caption projection (3840 → 4096) is handled
        by the transformer model, matching PyTorch LTX-2 architecture.
    """
    feature_extractor = GemmaFeaturesExtractorProjLinear(
        hidden_dim=hidden_dim,
        num_layers=num_gemma_layers,
    )

    embeddings_connector = Embeddings1DConnector(
        attention_head_dim=connector_head_dim,
        num_attention_heads=connector_heads,
        num_layers=connector_layers,
        num_learnable_registers=num_registers,
    )

    return VideoGemmaTextEncoderModel(
        feature_extractor=feature_extractor,
        embeddings_connector=embeddings_connector,
    )


def load_text_encoder_weights(
    encoder: VideoGemmaTextEncoderModel,
    weights_path: str,
) -> None:
    """
    Load text encoder weights from PyTorch safetensors file.

    This loads:
    - Feature extractor (aggregate_embed)
    - Embeddings connector (transformer blocks + learnable registers)

    Note: Caption projection weights (model.diffusion_model.caption_projection.*)
    are NOT loaded here. They are loaded into the transformer model instead,
    matching PyTorch LTX-2 architecture where caption_projection is part of
    the diffusion model, not the text encoder.

    Args:
        encoder: VideoGemmaTextEncoderModel instance.
        weights_path: Path to safetensors file.
    """
    from safetensors import safe_open
    import torch

    print(f"Loading text encoder weights from {weights_path}...")

    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        # Load feature extractor (aggregate_embed)
        fe_key = "text_embedding_projection.aggregate_embed.weight"
        if fe_key in f.keys():
            tensor = f.get_tensor(fe_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            # MLX Linear stores weight as (out, in), same as PyTorch - no transpose needed
            encoder.feature_extractor.aggregate_embed.weight = mx.array(tensor.numpy())
            loaded_count += 1

        # Load embeddings connector
        # Uses video_embeddings_connector prefix for video-only encoder
        connector_prefix = "model.diffusion_model.video_embeddings_connector."

        # Learnable registers
        reg_key = f"{connector_prefix}learnable_registers"
        if reg_key in f.keys():
            tensor = f.get_tensor(reg_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            encoder.embeddings_connector.learnable_registers = mx.array(tensor.numpy())
            loaded_count += 1

        # Transformer blocks
        for block_idx in range(2):
            block = encoder.embeddings_connector.transformer_1d_blocks[block_idx]
            block_prefix = f"{connector_prefix}transformer_1d_blocks.{block_idx}."

            # Attention weights
            attn_mapping = {
                "attn1.to_q.weight": ("attn1", "to_q", "weight"),
                "attn1.to_q.bias": ("attn1", "to_q", "bias"),
                "attn1.to_k.weight": ("attn1", "to_k", "weight"),
                "attn1.to_k.bias": ("attn1", "to_k", "bias"),
                "attn1.to_v.weight": ("attn1", "to_v", "weight"),
                "attn1.to_v.bias": ("attn1", "to_v", "bias"),
                "attn1.to_out.0.weight": ("attn1", "to_out", "weight"),
                "attn1.to_out.0.bias": ("attn1", "to_out", "bias"),
                "attn1.q_norm.weight": ("attn1", "q_norm", "weight"),
                "attn1.k_norm.weight": ("attn1", "k_norm", "weight"),
            }

            for pt_suffix, (attn_name, layer_name, param_name) in attn_mapping.items():
                pt_key = f"{block_prefix}{pt_suffix}"
                if pt_key in f.keys():
                    tensor = f.get_tensor(pt_key)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                    value = mx.array(tensor.numpy())

                    # MLX Linear stores (out, in) same as PyTorch - no transpose needed

                    attn = getattr(block, attn_name)
                    layer = getattr(attn, layer_name)
                    setattr(layer, param_name, value)
                    loaded_count += 1

            # Feed-forward weights
            ff_mapping = {
                "ff.net.0.proj.weight": ("project_in", "proj", "weight"),
                "ff.net.0.proj.bias": ("project_in", "proj", "bias"),
                "ff.net.2.weight": ("project_out", None, "weight"),
                "ff.net.2.bias": ("project_out", None, "bias"),
            }

            for pt_suffix, (layer1_name, layer2_name, param_name) in ff_mapping.items():
                pt_key = f"{block_prefix}{pt_suffix}"
                if pt_key in f.keys():
                    tensor = f.get_tensor(pt_key)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                    value = mx.array(tensor.numpy())

                    # MLX Linear stores (out, in) same as PyTorch - no transpose needed

                    layer1 = getattr(block.ff, layer1_name)
                    if layer2_name:
                        layer = getattr(layer1, layer2_name)
                    else:
                        layer = layer1
                    setattr(layer, param_name, value)
                    loaded_count += 1

        # Note: Caption projection weights (model.diffusion_model.caption_projection.*)
        # are loaded into the transformer model, not here.

    print(f"  Loaded {loaded_count} weight tensors")


def create_av_text_encoder(
    hidden_dim: int = 3840,
    num_gemma_layers: int = 49,
    connector_heads: int = 30,
    connector_head_dim: int = 128,
    connector_layers: int = 2,
    num_registers: int = 128,
) -> AudioVideoGemmaTextEncoderModel:
    """
    Create an audio+video text encoder with default LTX-2 configuration.

    Args:
        hidden_dim: Gemma hidden dimension (3840).
        num_gemma_layers: Number of Gemma layers (49).
        connector_heads: Number of attention heads in connector (30).
        connector_head_dim: Head dimension in connector (128).
        connector_layers: Number of transformer layers in connector (2).
        num_registers: Number of learnable registers (128).

    Returns:
        Configured AudioVideoGemmaTextEncoderModel.
    """
    feature_extractor = GemmaFeaturesExtractorProjLinear(
        hidden_dim=hidden_dim,
        num_layers=num_gemma_layers,
    )

    embeddings_connector = Embeddings1DConnector(
        attention_head_dim=connector_head_dim,
        num_attention_heads=connector_heads,
        num_layers=connector_layers,
        num_learnable_registers=num_registers,
    )

    audio_embeddings_connector = Embeddings1DConnector(
        attention_head_dim=connector_head_dim,
        num_attention_heads=connector_heads,
        num_layers=connector_layers,
        num_learnable_registers=num_registers,
    )

    return AudioVideoGemmaTextEncoderModel(
        feature_extractor=feature_extractor,
        embeddings_connector=embeddings_connector,
        audio_embeddings_connector=audio_embeddings_connector,
    )


def _load_connector_weights(
    f,
    connector: Embeddings1DConnector,
    prefix: str,
    loaded_count: int,
) -> int:
    """Load weights into an embeddings connector.

    Helper function to load weights for a single connector.

    Args:
        f: Open safetensors file handle.
        connector: The Embeddings1DConnector to load weights into.
        prefix: Weight key prefix (e.g., "model.diffusion_model.video_embeddings_connector.").
        loaded_count: Current count of loaded tensors.

    Returns:
        Updated count of loaded tensors.
    """
    import torch

    keys = list(f.keys())

    # Learnable registers
    reg_key = f"{prefix}learnable_registers"
    if reg_key in keys:
        tensor = f.get_tensor(reg_key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        connector.learnable_registers = mx.array(tensor.numpy())
        loaded_count += 1

    # Transformer blocks
    for block_idx in range(2):
        block = connector.transformer_1d_blocks[block_idx]
        block_prefix = f"{prefix}transformer_1d_blocks.{block_idx}."

        # Attention weights
        attn_mapping = {
            "attn1.to_q.weight": ("attn1", "to_q", "weight"),
            "attn1.to_q.bias": ("attn1", "to_q", "bias"),
            "attn1.to_k.weight": ("attn1", "to_k", "weight"),
            "attn1.to_k.bias": ("attn1", "to_k", "bias"),
            "attn1.to_v.weight": ("attn1", "to_v", "weight"),
            "attn1.to_v.bias": ("attn1", "to_v", "bias"),
            "attn1.to_out.0.weight": ("attn1", "to_out", "weight"),
            "attn1.to_out.0.bias": ("attn1", "to_out", "bias"),
            "attn1.q_norm.weight": ("attn1", "q_norm", "weight"),
            "attn1.k_norm.weight": ("attn1", "k_norm", "weight"),
        }

        for pt_suffix, (attn_name, layer_name, param_name) in attn_mapping.items():
            pt_key = f"{block_prefix}{pt_suffix}"
            if pt_key in keys:
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                attn = getattr(block, attn_name)
                layer = getattr(attn, layer_name)
                setattr(layer, param_name, value)
                loaded_count += 1

        # Feed-forward weights
        ff_mapping = {
            "ff.net.0.proj.weight": ("project_in", "proj", "weight"),
            "ff.net.0.proj.bias": ("project_in", "proj", "bias"),
            "ff.net.2.weight": ("project_out", None, "weight"),
            "ff.net.2.bias": ("project_out", None, "bias"),
        }

        for pt_suffix, (layer1_name, layer2_name, param_name) in ff_mapping.items():
            pt_key = f"{block_prefix}{pt_suffix}"
            if pt_key in keys:
                tensor = f.get_tensor(pt_key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                value = mx.array(tensor.numpy())

                layer1 = getattr(block.ff, layer1_name)
                if layer2_name:
                    layer = getattr(layer1, layer2_name)
                else:
                    layer = layer1
                setattr(layer, param_name, value)
                loaded_count += 1

    return loaded_count


def load_av_text_encoder_weights(
    encoder: AudioVideoGemmaTextEncoderModel,
    weights_path: str,
) -> None:
    """
    Load audio+video text encoder weights from PyTorch safetensors file.

    This loads:
    - Feature extractor (aggregate_embed)
    - Video embeddings connector
    - Audio embeddings connector

    Args:
        encoder: AudioVideoGemmaTextEncoderModel instance.
        weights_path: Path to safetensors file.
    """
    from safetensors import safe_open
    import torch

    print(f"Loading AV text encoder weights from {weights_path}...")

    loaded_count = 0

    with safe_open(weights_path, framework="pt") as f:
        # Load feature extractor (aggregate_embed)
        fe_key = "text_embedding_projection.aggregate_embed.weight"
        if fe_key in f.keys():
            tensor = f.get_tensor(fe_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            encoder.feature_extractor.aggregate_embed.weight = mx.array(tensor.numpy())
            loaded_count += 1

        # Load video embeddings connector
        video_prefix = "model.diffusion_model.video_embeddings_connector."
        loaded_count = _load_connector_weights(f, encoder.embeddings_connector, video_prefix, loaded_count)

        # Load audio embeddings connector
        audio_prefix = "model.diffusion_model.audio_embeddings_connector."
        loaded_count = _load_connector_weights(f, encoder.audio_embeddings_connector, audio_prefix, loaded_count)

    print(f"  Loaded {loaded_count} AV text encoder weight tensors")
