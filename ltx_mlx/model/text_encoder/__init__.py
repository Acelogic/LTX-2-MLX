"""Text encoder components for LTX-2 (Gemma connector)."""

from .connector import BasicTransformerBlock1D, Embeddings1DConnector
from .encoder import (
    CaptionProjection,
    VideoGemmaEncoderOutput,
    VideoGemmaTextEncoderModel,
    create_text_encoder,
    load_text_encoder_weights,
)
from .feature_extractor import (
    GemmaFeaturesExtractorProjLinear,
    norm_and_concat_padded_batch,
)

__all__ = [
    # Feature extractor
    "GemmaFeaturesExtractorProjLinear",
    "norm_and_concat_padded_batch",
    # Connector
    "BasicTransformerBlock1D",
    "Embeddings1DConnector",
    # Caption projection
    "CaptionProjection",
    # Main encoder
    "VideoGemmaTextEncoderModel",
    "VideoGemmaEncoderOutput",
    "create_text_encoder",
    "load_text_encoder_weights",
]
