# Codebase Concerns

**Analysis Date:** 2026-01-23

## Tech Debt

**Large monolithic files requiring refactoring:**
- Issue: Core pipeline and model files exceed 800+ lines, reducing maintainability
- Files:
  - `LTX_2_MLX/pipelines/two_stage.py` (812 lines)
  - `LTX_2_MLX/model/transformer/model.py` (834 lines)
  - `LTX_2_MLX/model/video_vae/simple_decoder.py` (809 lines)
  - `LTX_2_MLX/pipelines/ic_lora.py` (785 lines)
- Impact: Difficult to debug, test individual functions, and follow code flow. Makes future modifications risky
- Fix approach: Break files into logical sub-modules (e.g., separate denoising loops, VAE blocks, pipeline stages into helper modules)

**Deprecated but retained code:**
- Issue: `CaptionProjection` class kept for backwards compatibility but no longer used
- Files: `LTX_2_MLX/model/text_encoder/encoder.py` (lines 34-60)
- Impact: Maintenance burden; creates confusion about correct implementation path
- Fix approach: Remove deprecated class and add migration guide in comments if needed

**Multiple try-except blocks with silent ImportError handling:**
- Issue: 12+ locations use bare `except ImportError: pass` to gracefully handle missing optional packages
- Files:
  - `LTX_2_MLX/model/video_vae/simple_decoder.py:470`
  - `LTX_2_MLX/model/video_vae/tiling.py:286-287`
  - `LTX_2_MLX/model/video_vae/simple_encoder.py:329`
  - `LTX_2_MLX/pipelines/ic_lora.py:81, 132, 186, 312`
  - `LTX_2_MLX/model/transformer/rope.py:15`
  - `LTX_2_MLX/loader/weight_converter.py:345`
  - `LTX_2_MLX/components/schedulers.py:198`
- Impact: tqdm is optional but failures to import silently degrade functionality (no progress bars). Hard to debug why progress isn't showing
- Fix approach: Centralize optional dependency handling in a module, provide clear logging when features are unavailable

## Performance Bottlenecks

**Excessive mx.eval() calls in hot paths:**
- Issue: 61 explicit `mx.eval()` calls scattered throughout codebase, particularly in VAE decoder and pipelines
- Files:
  - `LTX_2_MLX/model/video_vae/simple_decoder.py` (lines 489-522)
  - `LTX_2_MLX/pipelines/two_stage.py` (lines 650, 654, 671)
  - Multiple calls in decoder stepping/upsampling loops
- Impact: Forces computation immediately instead of letting MLX schedule lazily; reduces optimization opportunities
- Fix approach: Use mx.eval() sparingly, only where output is needed immediately (e.g., before printing progress, returning final result)

**Memory pressure during weight loading:**
- Issue: Weight converter loads all transformer weights into dictionary before model update
- Files: `LTX_2_MLX/loader/weight_converter.py` (lines 362-429)
- Impact: On constrained systems, all weights in memory simultaneously before distribution to model parameters
- Fix approach: Implement streaming weight update (write weights directly to model parameters one batch at a time)

**Spatial upscaler division without bounds checking:**
- Issue: `c // (r * r)` calculation in PixelShuffle without validation that c is divisible
- Files: `LTX_2_MLX/model/upscaler/spatial.py` (line 205)
- Impact: Silent integer truncation if c not divisible by upscale_factor^2; produces incorrect reshape
- Fix approach: Add assertion before reshape: `assert c % (r * r) == 0, f"Channels {c} not divisible by {r*r}"`

## Fragile Areas

**VAE decoder with hardcoded noise parameters:**
- Issue: Noise injection uses hardcoded `self.decode_noise_scale` without exposure to configuration
- Files: `LTX_2_MLX/model/video_vae/simple_decoder.py` (lines 513-515)
- Why fragile: Cannot tune noise schedule for different use cases; matches PyTorch behavior but undocumented
- Safe modification: Make noise_scale a constructor parameter with default matching current behavior

**Patchification logic depends on exact shape assumptions:**
- Issue: `VideoLatentPatchifier.patchify()` assumes shapes are divisible by patch sizes without validation
- Files: `LTX_2_MLX/components/patchifiers.py` (lines 74-102)
- Why fragile: Integer division in reshape will silently truncate if shape not divisible; crashes only when unpatchify fails
- Safe modification: Add validation in patchify: `assert f % p1 == 0 and h % p2 == 0 and w % p3 == 0`

**LoRA weight fusion modifies in-place without backup:**
- Issue: Two-stage pipeline stores original weights manually instead of relying on restoration
- Files: `LTX_2_MLX/pipelines/two_stage.py` (lines 660-671)
- Why fragile: Manual weight management; if fusion fails halfway, inconsistent state remains
- Safe modification: Use context manager pattern for weight restoration, validate weight structure before fusion

**Audio modality handling mixed with video in transformer:**
- Issue: Audio-video transformer blocks mix both modalities with assertions for type narrowing
- Files: `LTX_2_MLX/model/transformer/transformer.py` (lines 447, 469, 493, 558, 569)
- Why fragile: Runtime assertions with no fallback; if audio not provided when expected, crashes at inference time
- Safe modification: Pre-validate modality availability at pipeline initialization, not in forward pass

## Test Coverage Gaps

**End-to-end pipeline tests missing:**
- What's not tested: Full two-stage pipeline execution with all stages, LoRA fusion, and output validation
- Files: `tests/test_pipelines.py` has configuration tests but no integration tests for actual generation
- Risk: Regressions in full pipeline not caught until user-facing generation fails
- Priority: High - two-stage is primary use case

**Audio modality tests incomplete:**
- What's not tested: Audio generation, audio-video synchronization, audio VAE encode/decode parity
- Files: No dedicated audio pipeline tests; audio code merged into two_stage but not tested separately
- Risk: Audio generation may silently produce incorrect output; audio parity with PyTorch unknown
- Priority: High - audio is recent addition

**Weight conversion edge cases:**
- What's not tested: FP8 quantized weight loading, streaming mode correctness, partial weight loading
- Files: `tests/test_loaders.py` exists but doesn't cover FP8 dequantization or streaming behavior
- Risk: FP8 models may fail silently with wrong weight conversions
- Priority: Medium - affects model variant support

**Upscaler output validation:**
- What's not tested: Output shapes from spatial/temporal upscalers, boundary conditions, extreme resolution inputs
- Files: `tests/test_upscalers.py` has basic tests but no shape validation or extreme input tests
- Risk: Upscaler may produce misshapen output caught only at decode time
- Priority: Medium

**Conditioning application edge cases:**
- What's not tested: Image conditions at video boundaries, out-of-bounds frame indices, mismatched spatial shapes
- Files: `tests/test_conditioning.py` exists but skips boundary validation tests
- Risk: Conditioning at video edges (frame 0, last frame) may apply incorrectly
- Priority: Medium

## Known Bugs

**Integer division silence in shape calculations:**
- Symptoms: Reshapes silently truncate if shapes not exactly divisible by patch sizes
- Files:
  - `LTX_2_MLX/components/patchifiers.py` (lines 92-100 for video, similar for audio)
  - `LTX_2_MLX/model/upscaler/spatial.py` (line 209)
- Trigger: Use resolution not divisible by (patch_size * 8) or upscale factor
- Workaround: Ensure input resolutions are divisible by 64 (enforced in config validation but not in low-level functions)

**Blur downsampler unused code path:**
- Symptoms: BlurDownsample layer always returns pass-through for stride=1
- Files: `LTX_2_MLX/model/upscaler/spatial.py` (lines 259-261)
- Trigger: Blur is instantiated but stride=1 always triggers early return; manual depthwise conv code never runs
- Workaround: Currently not a bug since stride=1 is default, but code suggests unfinished implementation
- Fix approach: Remove dead code or implement depthwise blur properly

## Security Considerations

**No input validation on image paths:**
- Risk: Path traversal if image path comes from untrusted source
- Files: `LTX_2_MLX/pipelines/common.py` (line 54) checks existence but not canonicalization
- Current mitigation: PIL.Image.open() will fail on non-image files; no symlink attacks possible
- Recommendations: Use `os.path.realpath()` and verify path is within expected directory if loading from user input

**No bounds checking on video dimensions:**
- Risk: Extremely large resolution requests could allocate unbounded memory
- Files: Pipeline configs accept arbitrary height/width; no upper bounds enforced
- Current mitigation: Tested up to 1024×1536 (~128GB peak memory)
- Recommendations: Add max dimension constraints (e.g., max 2048×4096) in config validation

**Import of torch during MLX-only weight loading:**
- Risk: Depends on PyTorch being installed for weight conversion, even when not otherwise needed
- Files: `LTX_2_MLX/loader/weight_converter.py` (line 385)
- Current mitigation: PyTorch is listed in dependencies, unavoidable for safetensors loading
- Recommendations: Document that PyTorch is required for weight loading; consider alternative weight loaders

## Scaling Limits

**Two-stage pipeline memory scaling:**
- Current capacity: Tested to 1024×1536 resolution (requires ~200GB peak memory with upscaling)
- Limit: Cannot easily generate >2048×3072 without OOM on systems <256GB RAM
- Scaling path: Implement chunked upscaling (e.g., spatially tile latents, upsample tiles separately, reassemble)

**Text encoding latency:**
- Current capacity: Gemma 3 encoding takes ~5-10 seconds for 1 prompt
- Limit: Batch encoding of multiple prompts not implemented
- Scaling path: Add batch text encoding to pipelines, cache Gemma model across generations

**Frame generation latency:**
- Current capacity: Two-stage generates 97 frames in ~45s on M2 Max
- Limit: Generation not parallelizable across frames (sequential diffusion)
- Scaling path: Implement sliding-window batch generation for next-frame prediction

## Fragile Dependencies

**torch >= 2.9.1:**
- Risk: PyTorch breaking changes in minor releases
- Impact: Weight loading may fail if torch API changes (tensor.numpy() conversion)
- Migration plan: Abstract weight loading behind a facade; support multiple torch versions

**mlx >= 0.20.0:**
- Risk: MLX is under active development; API may change
- Impact: Operators like RoPE, GroupNorm may change behavior or be renamed
- Migration plan: Pin versions tightly; test against multiple MLX versions in CI

**transformers >= 4.57.3:**
- Risk: Gemma tokenizer changes
- Impact: Text encoding may produce different token sequences
- Migration plan: Use explicit tokenizer version pinning

## Missing Critical Features

**Deterministic seeding incomplete:**
- Problem: Random seed affects torch/numpy but not MLX random operations fully
- Files: `LTX_2_MLX/pipelines/two_stage.py` (line 500 uses mx.random.normal without seeding)
- Blocks: Cannot reliably reproduce generation from seed alone
- Fix: Centralize RNG initialization with mx.random.seed() calls before all sampling

**Error recovery in long-running pipelines:**
- Problem: OOM or timeout crashes lose all progress; no checkpointing between stages
- Blocks: Cannot resume generation if stage 1 completes but stage 2 fails
- Fix: Implement checkpoint saves after stage 1 completion, resume from checkpoint

**Batch generation unsupported:**
- Problem: Pipelines generate one video at a time; no batching across prompts/seeds
- Blocks: Cannot efficiently generate multiple videos in one run
- Fix: Add batch dimension to latent state, denoise multiple videos simultaneously

## Incompleteness Issues

**Audio pipeline parity claim unverified:**
- Issue: Recent audio support added but correlation with PyTorch not tested
- Files: Audio decoder, VAE, modality handling in transformer
- Gaps: No `test_audio_parity.py` equivalent to `test_parity.py`; no numeric correlation checks
- Impact: Audio output quality unknown relative to reference

**LoRA fusion stability:**
- Issue: LoRA weights loaded but stability/convergence not documented
- Files: `LTX_2_MLX/loader/__init__.py` contains LoRA logic with minimal comments
- Gaps: No tests for LoRA numerical stability or weight range validation
- Impact: LoRA may amplify/suppress generation quality unpredictably

---

*Concerns audit: 2026-01-23*
