# External Integrations

**Analysis Date:** 2026-01-23

## APIs & External Services

**None** - This is a local inference engine with no active external API dependencies in production code.

## Data Storage

**Model Weights:**
- Source: HuggingFace Hub (Lightricks/LTX-2, google/gemma-3-12b-it)
- Storage: Local filesystem only (weights directory)
- Integration point: `scripts/download_weights.py`
  - Uses `huggingface_hub` for downloading
  - Supports HF_TOKEN environment variable for authentication
- Weight format: safetensors (.safetensors files)

**Databases:**
- None - No database integration. Model inference is stateless.

**File Storage:**
- Local filesystem only
- Generated videos saved to `outputs/` directory or specified path
- Model weights cached in `weights/` directory

**Caching:**
- None - No caching service. MLX handles in-memory computation on GPU.

## Authentication & Identity

**HuggingFace Hub Access:**
- Optional authentication for downloading Gemma 3 model (requires license acceptance)
- Environment variable: `HF_TOKEN` (optional, used in `scripts/download_weights.py`)
- Authentication method: `huggingface_hub.login(token=token)`
- Location: `scripts/download_weights.py` lines 263-264
- Note: Gemma 3 requires accepting license agreement at https://huggingface.co/google/gemma-3-12b-it

**No Custom Auth:**
- No API keys required
- No user authentication system
- No rate limiting or access control

## Monitoring & Observability

**Error Tracking:**
- None - No external error tracking service

**Logging:**
- Python `logging` module available but not extensively configured
- Console output via `print()` for progress and errors
- Progress bars via `tqdm` library (optional import in `scripts/generate.py`)
- Location: `scripts/generate.py` lines 146-156

**Metrics & Performance:**
- No external metrics collection
- Local timing/profiling only via Python's built-in timing

## CI/CD & Deployment

**Hosting:**
- None - Local application, runs on user's machine
- No cloud hosting, no server deployment

**CI Pipeline:**
- None configured

**Testing Infrastructure:**
- Local pytest execution only
- Test markers support slow/integration/unit test categorization
- No remote test runners

## Environment Configuration

**Required Environment Variables:**
- `HF_TOKEN` (optional) - HuggingFace authentication token for accessing Gemma 3
  - Only needed if downloading Gemma 3 without interactive login
  - Used in: `scripts/download_weights.py` line 334

**Default Configuration:**
- All paths relative to project root
- Model weights expected in: `weights/` directory
- Generated videos saved to: `outputs/` directory (or user-specified path)
- No .env file support or dotenv integration

**Configuration Approach:**
- Command-line arguments in scripts (see `scripts/generate.py` argparse configuration)
- Hardcoded weight paths in `scripts/download_weights.py` (lines 40-105)
- No runtime configuration server

## Model Downloads

**Weight Sources:**
All weights downloaded from HuggingFace Hub via `huggingface_hub` package:

1. **LTX-2 19B Distilled** (recommended)
   - Repository: Lightricks/LTX-2
   - File: `ltx-video-2b-v0.9.5.safetensors`
   - Size: ~43GB
   - Location: `weights/ltx-2/ltx-2-19b-distilled.safetensors`

2. **LTX-2 19B Distilled (FP8)** (quantized)
   - Repository: Lightricks/LTX-2
   - File: `ltx-video-2b-v0.9.5-fp8.safetensors`
   - Size: ~27GB
   - Location: `weights/ltx-2/ltx-2-19b-distilled-fp8.safetensors`

3. **LTX-2 19B Dev** (higher quality)
   - Repository: Lightricks/LTX-2
   - File: `ltx-video-2b-v0.9.5-dev.safetensors`
   - Size: ~43GB
   - Location: `weights/ltx-2/ltx-2-19b-dev.safetensors`

4. **Spatial Upscaler 2x**
   - Repository: Lightricks/LTX-2
   - File: `ltx-video-2b-v0.9.5-spatial-upscaler-2x.safetensors`
   - Size: ~995MB
   - Location: `weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors`

5. **Temporal Upscaler 2x**
   - Repository: Lightricks/LTX-2
   - File: `ltx-video-2b-v0.9.5-temporal-upscaler-2x.safetensors`
   - Size: ~262MB
   - Location: `weights/ltx-2/ltx-2-temporal-upscaler-x2-1.0.safetensors`

6. **Distilled LoRA**
   - Repository: Lightricks/LTX-2
   - File: `ltx-video-2b-v0.9.5-distilled-lora-384.safetensors`
   - Size: ~1.5GB
   - Location: `weights/ltx-2/ltx-2-19b-distilled-lora-384.safetensors`

7. **Gemma 3 12B Text Encoder**
   - Repository: google/gemma-3-12b-it
   - Type: Full repository download
   - Size: ~25GB
   - Location: `weights/gemma-3-12b/`
   - Requires: License acceptance at https://huggingface.co/google/gemma-3-12b-it

**Download Integration:**
- Function: `download_weight()` in `scripts/download_weights.py` (lines 204-242)
- Uses: `huggingface_hub.hf_hub_download()` for single files
- Uses: `huggingface_hub.snapshot_download()` for full repositories (Gemma)
- Handles authentication: `huggingface_hub.login(token)`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## External Model References

**Text Encoder:**
- Gemma 3 12B (google/gemma-3-12b-it)
- Integration: Direct weight loading from HuggingFace
- Purpose: Text encoding for video generation prompts
- License: Requires acceptance at HuggingFace

**Base Model:**
- Lightricks LTX-2
- Metadata: https://arxiv.org/abs/2601.03233
- Source: https://github.com/Lightricks/LTX-2
- All models downloaded from: https://huggingface.co/Lightricks/LTX-2

## No External Dependencies For Inference

**Important:** Once weights are downloaded locally, the application requires NO external connectivity for video generation. All inference runs locally on Apple Silicon GPU via MLX.

---

*Integration audit: 2026-01-23*
