"""Model ledger for coordinating model loading in LTX-2 MLX.

The ModelLedger provides a central coordinator for loading and managing all models
used in an LTX-2 pipeline, including transformer, VAE encoder/decoder, text encoder,
audio components, and upscalers.
"""

import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import mlx.core as mx

from ..loader import LoRAConfig


@dataclass
class ModelLedger:
    """
    Central coordinator for loading and building models used in LTX-2 pipelines.

    The ledger manages model instances and provides factory methods for constructing
    models on demand. Models are cached after first load to avoid redundant loading.

    Attributes:
        checkpoint_path: Path to the main LTX-2 checkpoint.
        gemma_path: Path to Gemma 3 weights directory.
        spatial_upscaler_path: Path to spatial upscaler weights (optional).
        temporal_upscaler_path: Path to temporal upscaler weights (optional).
        loras: List of LoRA configurations to apply to transformer.
        compute_dtype: Computation dtype (float32 or float16).
        use_fp8: Whether to load FP8 weights.
    """

    checkpoint_path: Optional[str] = None
    gemma_path: Optional[str] = None
    spatial_upscaler_path: Optional[str] = None
    temporal_upscaler_path: Optional[str] = None
    loras: List[LoRAConfig] = field(default_factory=list)
    compute_dtype: mx.Dtype = mx.float32
    use_fp8: bool = False

    # Cached model instances
    _transformer: Optional[Any] = field(default=None, repr=False)
    _video_encoder: Optional[Any] = field(default=None, repr=False)
    _video_decoder: Optional[Any] = field(default=None, repr=False)
    _audio_encoder: Optional[Any] = field(default=None, repr=False)
    _audio_decoder: Optional[Any] = field(default=None, repr=False)
    _vocoder: Optional[Any] = field(default=None, repr=False)
    _text_encoder: Optional[Any] = field(default=None, repr=False)
    _gemma: Optional[Any] = field(default=None, repr=False)
    _spatial_upscaler: Optional[Any] = field(default=None, repr=False)
    _temporal_upscaler: Optional[Any] = field(default=None, repr=False)

    def transformer(self, force_reload: bool = False):
        """
        Get or load the transformer model.

        Args:
            force_reload: If True, reload the model even if cached.

        Returns:
            LTXModel instance.
        """
        if self._transformer is not None and not force_reload:
            return self._transformer

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to load transformer")

        from ..model.transformer import LTXModel, LTXModelType
        from ..loader import load_transformer_weights, fuse_lora_into_weights
        from mlx.utils import tree_flatten

        print(f"Loading transformer from {self.checkpoint_path}...")
        model = LTXModel(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=128,
            out_channels=128,
            num_layers=48,
            cross_attention_dim=4096,
            caption_channels=3840,
            positional_embedding_theta=10000.0,
            compute_dtype=self.compute_dtype,
        )

        load_transformer_weights(model, self.checkpoint_path, use_fp8=self.use_fp8)

        # Apply LoRAs if any
        if self.loras:
            print(f"Applying {len(self.loras)} LoRA(s)...")
            flat_params = dict(tree_flatten(model.parameters()))
            fused = fuse_lora_into_weights(flat_params, self.loras)
            model.load_weights(list(fused.items()))
            mx.eval(model.parameters())

        self._transformer = model
        return model

    def video_encoder(self, force_reload: bool = False):
        """Get or load the video VAE encoder."""
        if self._video_encoder is not None and not force_reload:
            return self._video_encoder

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to load video encoder")

        from ..model.video_vae import SimpleVideoEncoder, load_vae_encoder_weights

        print("Loading video encoder...")
        encoder = SimpleVideoEncoder(compute_dtype=self.compute_dtype)
        load_vae_encoder_weights(encoder, self.checkpoint_path)

        self._video_encoder = encoder
        return encoder

    def video_decoder(self, force_reload: bool = False):
        """Get or load the video VAE decoder."""
        if self._video_decoder is not None and not force_reload:
            return self._video_decoder

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to load video decoder")

        from ..model.video_vae import SimpleVideoDecoder, load_vae_decoder_weights

        print("Loading video decoder...")
        decoder = SimpleVideoDecoder(compute_dtype=self.compute_dtype)
        load_vae_decoder_weights(decoder, self.checkpoint_path)

        self._video_decoder = decoder
        return decoder

    def audio_encoder(self, force_reload: bool = False):
        """Get or load the audio VAE encoder."""
        if self._audio_encoder is not None and not force_reload:
            return self._audio_encoder

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to load audio encoder")

        from ..model.audio_vae import AudioEncoder, load_audio_encoder_weights

        print("Loading audio encoder...")
        encoder = AudioEncoder(compute_dtype=self.compute_dtype)
        load_audio_encoder_weights(encoder, self.checkpoint_path)

        self._audio_encoder = encoder
        return encoder

    def audio_decoder(self, force_reload: bool = False):
        """Get or load the audio VAE decoder."""
        if self._audio_decoder is not None and not force_reload:
            return self._audio_decoder

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to load audio decoder")

        from ..model.audio_vae import AudioDecoder, load_audio_decoder_weights

        print("Loading audio decoder...")
        decoder = AudioDecoder(compute_dtype=self.compute_dtype)
        load_audio_decoder_weights(decoder, self.checkpoint_path)

        self._audio_decoder = decoder
        return decoder

    def vocoder(self, force_reload: bool = False):
        """Get or load the vocoder."""
        if self._vocoder is not None and not force_reload:
            return self._vocoder

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to load vocoder")

        from ..model.audio_vae import Vocoder, load_vocoder_weights

        print("Loading vocoder...")
        voc = Vocoder(compute_dtype=self.compute_dtype)
        load_vocoder_weights(voc, self.checkpoint_path)

        self._vocoder = voc
        return voc

    def text_encoder(self, force_reload: bool = False):
        """Get or load the text encoder (feature extractor + connector)."""
        if self._text_encoder is not None and not force_reload:
            return self._text_encoder

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to load text encoder")

        from ..model.text_encoder import create_text_encoder, load_text_encoder_weights

        print("Loading text encoder...")
        encoder = create_text_encoder()
        load_text_encoder_weights(encoder, self.checkpoint_path)

        self._text_encoder = encoder
        return encoder

    def gemma(self, force_reload: bool = False):
        """Get or load the Gemma 3 base model."""
        if self._gemma is not None and not force_reload:
            return self._gemma

        if self.gemma_path is None:
            raise ValueError("gemma_path is required to load Gemma model")

        from ..model.text_encoder.gemma3 import Gemma3Config, Gemma3Model, load_gemma3_weights

        print(f"Loading Gemma 3 from {self.gemma_path}...")
        config = Gemma3Config()
        model = Gemma3Model(config)
        load_gemma3_weights(model, self.gemma_path)

        self._gemma = model
        return model

    def spatial_upscaler(self, force_reload: bool = False):
        """Get or load the spatial upscaler."""
        if self._spatial_upscaler is not None and not force_reload:
            return self._spatial_upscaler

        if self.spatial_upscaler_path is None:
            raise ValueError("spatial_upscaler_path is required to load spatial upscaler")

        from ..model.upscaler import SpatialUpscaler, load_spatial_upscaler_weights

        print(f"Loading spatial upscaler from {self.spatial_upscaler_path}...")
        upscaler = SpatialUpscaler(compute_dtype=self.compute_dtype)
        load_spatial_upscaler_weights(upscaler, self.spatial_upscaler_path)

        self._spatial_upscaler = upscaler
        return upscaler

    def temporal_upscaler(self, force_reload: bool = False):
        """Get or load the temporal upscaler."""
        if self._temporal_upscaler is not None and not force_reload:
            return self._temporal_upscaler

        if self.temporal_upscaler_path is None:
            raise ValueError("temporal_upscaler_path is required to load temporal upscaler")

        from ..model.upscaler import TemporalUpscaler, load_temporal_upscaler_weights

        print(f"Loading temporal upscaler from {self.temporal_upscaler_path}...")
        upscaler = TemporalUpscaler(compute_dtype=self.compute_dtype)
        load_temporal_upscaler_weights(upscaler, self.temporal_upscaler_path)

        self._temporal_upscaler = upscaler
        return upscaler

    def clear_model(self, model_name: str) -> None:
        """
        Clear a specific model from cache and free memory.

        Args:
            model_name: Name of the model to clear (e.g., "transformer", "video_encoder").
        """
        attr_name = f"_{model_name}"
        if hasattr(self, attr_name):
            setattr(self, attr_name, None)
            gc.collect()
            mx.metal.clear_cache()

    def clear_all_models(self) -> None:
        """Clear all cached models and free memory."""
        self._transformer = None
        self._video_encoder = None
        self._video_decoder = None
        self._audio_encoder = None
        self._audio_decoder = None
        self._vocoder = None
        self._text_encoder = None
        self._gemma = None
        self._spatial_upscaler = None
        self._temporal_upscaler = None
        gc.collect()
        mx.metal.clear_cache()

    def with_loras(self, loras: List[LoRAConfig]) -> "ModelLedger":
        """
        Create a new ModelLedger with additional LoRAs.

        The new ledger shares no cached models with this one.

        Args:
            loras: List of LoRA configurations to add.

        Returns:
            New ModelLedger instance with combined LoRAs.
        """
        return ModelLedger(
            checkpoint_path=self.checkpoint_path,
            gemma_path=self.gemma_path,
            spatial_upscaler_path=self.spatial_upscaler_path,
            temporal_upscaler_path=self.temporal_upscaler_path,
            loras=list(self.loras) + list(loras),
            compute_dtype=self.compute_dtype,
            use_fp8=self.use_fp8,
        )


def create_model_ledger(
    checkpoint_path: str,
    gemma_path: Optional[str] = None,
    spatial_upscaler_path: Optional[str] = None,
    temporal_upscaler_path: Optional[str] = None,
    loras: Optional[List[LoRAConfig]] = None,
    use_fp16: bool = False,
    use_fp8: bool = False,
) -> ModelLedger:
    """
    Create a ModelLedger with the given configuration.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint.
        gemma_path: Path to Gemma 3 weights (optional).
        spatial_upscaler_path: Path to spatial upscaler (optional).
        temporal_upscaler_path: Path to temporal upscaler (optional).
        loras: List of LoRA configs (optional).
        use_fp16: Use FP16 computation.
        use_fp8: Load FP8 quantized weights.

    Returns:
        Configured ModelLedger instance.
    """
    compute_dtype = mx.float16 if use_fp16 else mx.float32

    return ModelLedger(
        checkpoint_path=checkpoint_path,
        gemma_path=gemma_path,
        spatial_upscaler_path=spatial_upscaler_path,
        temporal_upscaler_path=temporal_upscaler_path,
        loras=loras or [],
        compute_dtype=compute_dtype,
        use_fp8=use_fp8,
    )
