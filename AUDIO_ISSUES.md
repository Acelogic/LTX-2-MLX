# LTX-2.3 MLX Audio Issues — Investigation Log

**Date:** April 5, 2026  
**Current State:** Backup 0577 — speech works at ~60% clarity  
**Reference:** ComfyUI (PyTorch+MPS) produces clear speech with same model/prompts

---

## Symptom

- Speech audio is ~40% garbled across all pipelines (distilled, two-stage, i2v)
- Quality is best in first 2-3 seconds, degrades by 7-8 seconds
- Video quality is fine — only audio/speech affected
- Short prompts (20-30 tokens) work better than long prompts (200+ tokens)
- Lip sync tracks correctly despite garbled speech
- Ambient/environmental audio generates fine — speech specifically is weak

## Root Cause (Confirmed)

**The MLX diffusion transformer generates audio latents where speech is underrepresented relative to ambient sounds.**

Evidence chain:
1. Exported ComfyUI text embeddings → fed to MLX pipeline → still no clear speech (rules out text encoder)
2. Saved MLX mel spectrogram → fed to PyTorch vocoder → speech very quiet, mostly ambient (rules out MLX vocoder)
3. Therefore: the 48-layer MLX transformer's audio cross-attention produces weaker speech content than PyTorch

This is a numerical precision divergence across 48 transformer layers. Audio speech generation requires higher fidelity conditioning than video or ambient audio.

---

## Issues Found & Fixed

### 1. Gemma Per-Layer RoPE (FIXED — Backup 0577)
- **Problem:** Single RoPE config (theta=1M, scaling=8.0) applied to all 48 Gemma layers
- **Fix:** Two configs — 40 sliding layers (theta=10k, no scaling, 1024 window) + 8 full layers (theta=1M, scaling=8.0)
- **Impact:** Cosine similarity went from 0.05 to 0.934 at final layer. First working speech.
- **File:** `LTX_2_MLX/model/text_encoder/gemma3.py`

### 2. Gemma Boolean Attention Masks (FIXED — Backup 0577)
- **Problem:** Additive float masks caused NaN for all-padded rows
- **Fix:** Switched to boolean masks matching HuggingFace behavior
- **File:** `LTX_2_MLX/model/text_encoder/gemma3.py`

### 3. Connector Register Handling (FIXED — Backup 0578, not in 0577)
- **Problem:** MLX connector replaced padding within sequence (256 positions) instead of appending registers to extend to 1024
- **Fix:** Changed `_replace_padded_with_learnable_registers` → `_append_learnable_registers` matching ComfyUI behavior
- **Impact:** Correct sequence length but didn't fix speech quality
- **File:** `LTX_2_MLX/model/text_encoder/connector.py`

### 4. Double Precision RoPE for Connector (FIXED — Backup 0578, not in 0577)
- **Problem:** Connector used float32 for frequency grid; checkpoint specifies `frequencies_precision: float64`
- **Fix:** Added `double_precision_rope` flag, reads from checkpoint metadata
- **Impact:** Matches ComfyUI behavior but didn't fix speech quality
- **Files:** `connector.py`, `encoder.py`, `rope.py`

### 5. Padding Trim (FIXED — Backup 0578, not in 0577)
- **Problem:** Padding trimmed to next multiple of 128 instead of exact real token count
- **Fix:** Trim to exact real token count (matching ComfyUI)
- **Impact:** No audio impact
- **File:** `scripts/generate.py`

---

## Things Tried That Didn't Help

### 6. Gemma Model Swap (QAT → Standard)
- **Tried:** Replaced `gemma-3-12b-it-qat-q4_0-unquantized` with standard `gemma-3-12b-it` (bf16)
- **Result:** Audio slightly better, video slightly worse. Neither model matches ComfyUI exactly.
- **Note:** Standard model downloaded to `/Users/steveross/Documents/Development/Source Models/gemma-3-12b-it`
- ComfyUI uses `gemma_3_12B_it_fp4_mixed.safetensors` (standard model in fp4 quantization)

### 7. ComfyUI Embedding Injection (Definitive Test)
- **Tried:** Exported text embeddings from ComfyUI's `preprocess_text_embeds`, loaded into MLX pipeline
- **Result:** Ambient audio present, NO speech. Proves issue is in MLX diffusion transformer, not text encoder.
- **Tools:** `save_embeddings_hook.py`, `_load_comfyui_embeddings()` in app.py

### 8. PyTorch Vocoder on MLX Mel (Definitive Test)
- **Tried:** Saved mel spectrogram from MLX VAE decoder, ran through ComfyUI's PyTorch vocoder
- **Result:** Speech very quiet, mostly ambient. Proves mel spectrogram itself has weak speech content.
- **Conclusion:** Issue is upstream of vocoder — in the diffusion transformer's audio latent generation.

---

## Remaining Differences (Not Yet Investigated)

### Transformer-Level
- 48-layer transformer numerical divergence between MLX (Metal) and PyTorch (MPS)
- Audio cross-attention may weight text tokens differently due to floating-point arithmetic differences
- AdaLN modulation (shift/scale/gate) may accumulate differently across layers
- Audio-video cross-modal attention could have subtle differences

### Connector-Level (Backup 0577 state)
- Connector uses replace-and-sort behavior (old), not append (ComfyUI behavior)
- The append fix (backup 0578) was correct but didn't help because the bottleneck is the transformer
- `connector_positional_embedding_max_pos: [4096]` from checkpoint metadata — verify MLX reads this
- RMSNorm epsilon: ComfyUI uses 1e-5, MLX uses 1e-6 in connector attention

### Audio Token Count Off-by-One (FOUND — round vs ceil)
- ComfyUI produces 502 audio tokens, MLX produces 501 for same duration (481 frames)
- Cause: `round()` in MLX types.py vs `math.ceil()` in ComfyUI audio_vae.py
- Changing to ceil() made audio WORSE (voices super quiet) — reverted
- The model may have been trained with round() behavior, or the off-by-one matters less than expected

### Layer-by-Layer Comparison Results (Same Seed)
- Layer 0: 0.135 cosine — initial audio states already differ (noise + positions)
- Layers 1-47: gradual convergence from 0.05 to 0.945 — transformer blocks work correctly
- Final latent: 0.186 cosine — output projection diverges again
- MLX std consistently lower (0.56 vs 0.79) — MLX output is compressed/muted
- This pattern suggests the issue is in INPUTS (noise, positions) not transformer blocks

### av_ca_timestep_scale_multiplier Bug (FIXED)
- **`av_ca_timestep_scale_multiplier` was 1 instead of 1000** (checkpoint metadata value)
- This made the audio-video cross-attention gate factor 0.001 instead of 1.0 — effectively zeroing cross-modal gates
- Cross-modal attention carries speech information (lip sync → audio), so speech was weak while ambient was fine
- **Fix**: Added `av_ca_timestep_scale_multiplier=1000` to `load_av_transformer` in generate.py
- **Result**: 5-second clips now nearly perfect speech. Significant improvement.

### GPU Watchdog Crash on Audio Decode (FIXED)
- Added mx.eval() inside AMPBlock1 after each dilation iteration
- Added mx.clear_cache() between vocoder stages and BWE stages
- Added mx.eval() after STFT conv1d and conv_post
- 10-second clips now complete without kernel panic

### Duration-Dependent Amplitude Bug (ACTIVE — Next to Fix)
- 5-second clip: RMS 5535, peak 31977 — loud, healthy audio, near-perfect speech
- 10-second clip: RMS 1137, peak 9899 — **5x quieter**, speech mumbles
- The 10s clip is quiet FROM THE START (second 0: RMS 1467), not just degrading at the end
- This means the audio latent amplitude scales inversely with duration
- Something in the pipeline normalizes by number of tokens/frames
- Most likely in: noise generation, denoising step, or MultiModalGuider
- Video quality is NOT affected — only audio

### Temporal Degradation Pattern (CONFIRMED — partly explained by gate bug)
- 5 second clip: nearly perfect voice (with 1000x gate fix)
- 10 second clip: overall 5x quieter + mumbles toward end
- 20 second clip: degradation from 7-8 seconds onward
- The gate fix solved speech clarity for short clips; the amplitude bug is separate

### Potential Deep-Dive Areas
- Compare audio latents (pre-VAE-decode) between MLX and ComfyUI for same seed/prompt
- Layer-by-layer transformer output comparison (heavy instrumentation needed)
- Check if audio CFG/guidance scale is applied identically
- Verify MultiModalGuider computes modality_scale correctly for audio
- Check if audio self-attention RoPE positions match ComfyUI exactly

---

## Architecture Quick Reference

### Audio Decode Pipeline
```
Transformer output (B, T, 128) patchified
  → Unpatchify: (B, 8, T, 16)
  → Denormalize: per-channel stats (mean/std from checkpoint)
  → VAE Decoder: 2D convolutions, 3 upsample levels → (B, 2, T*4, 64) mel
  → Vocoder (BigVGAN v2): 108+ 1D convolutions, 5 upsample stages → waveform
  → BWE: mel recompute → second vocoder → resample → residual add
  → Output: (B, 2, samples) at 24kHz stereo
```

### Key Files
- Transformer: `LTX_2_MLX/model/transformer/model.py`, `transformer.py`
- Audio cross-attention: `transformer.py` lines 545-553
- Audio VAE decoder: `LTX_2_MLX/model/audio_vae/decoder.py`
- Vocoder: `LTX_2_MLX/model/audio_vae/vocoder.py`
- Audio patchifier: `LTX_2_MLX/components/patchifiers.py` (AudioPatchifier)
- Gemma text encoder: `LTX_2_MLX/model/text_encoder/gemma3.py`
- Connector: `LTX_2_MLX/model/text_encoder/connector.py`

### ComfyUI Reference Files
- AV model: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/av_model.py`
- Text encoder: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/text_encoders/lt.py`
- Connector: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/embeddings_connector.py`
- Audio VAE: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/vae/audio_vae.py`
- Vocoder: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/vocoders/vocoder.py`

### Test Tools Created
- `save_embeddings_hook.py` — Patches ComfyUI to export text embeddings
- `test_vocoder_from_mel.py` — Feeds MLX mel through PyTorch vocoder
- Exported embeddings: `/Users/steveross/Documents/ComfyUI/exported_embeddings/`
