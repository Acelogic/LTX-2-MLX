# LTX-2-MLX ANE Acceleration

## What This Is

Research project exploring Apple Neural Engine (ANE) acceleration for LTX-2-MLX video generation components. The goal is to convert viable components (VAE encoder/decoder, spatial/temporal upscalers) to Core ML format and benchmark whether running them on ANE provides any measurable speedup over the current MLX/GPU implementation.

## Core Value

Discover and implement any performance gains from ANE acceleration for the video generation pipeline.

## Requirements

### Validated

<!-- Existing capabilities from the LTX-2-MLX codebase -->

- ✓ Text-to-video generation via MLX — existing
- ✓ Two-stage pipeline with upscaling — existing
- ✓ One-stage distilled pipeline — existing
- ✓ Video VAE encoder/decoder (MLX) — existing
- ✓ Spatial upscaler 2x (MLX) — existing
- ✓ Temporal upscaler 2x (MLX) — existing
- ✓ Gemma 3 text encoder — existing
- ✓ Image-to-video conditioning — existing
- ✓ Keyframe interpolation — existing
- ✓ Audio generation support — existing

### Active

<!-- ANE acceleration research scope -->

- [ ] Research Core ML conversion feasibility for VAE decoder
- [ ] Research Core ML conversion feasibility for VAE encoder
- [ ] Research Core ML conversion feasibility for spatial upscaler
- [ ] Research Core ML conversion feasibility for temporal upscaler
- [ ] Convert at least one component to Core ML format
- [ ] Benchmark Core ML/ANE vs MLX/GPU performance
- [ ] Integrate viable ANE components into pipeline (if beneficial)

### Out of Scope

- Transformer ANE conversion — 19B parameters, too large for ANE
- Text encoder ANE conversion — 12B parameters, too large for ANE
- Core ML-only implementation — hybrid approach, MLX remains primary
- Mobile/iOS deployment — macOS only for this research

## Context

**Existing Architecture:**
- Layered pipeline architecture: Model → Components → Pipeline → Loader
- MLX-native implementation running on Apple Silicon GPU
- VAE and upscalers are separate, modular components (~50-250M params each)
- Components have clean interfaces making them candidates for swappable implementations

**ANE Considerations:**
- ANE accessed via Core ML, not MLX directly
- Core ML has practical size limits (~2-4GB models)
- Convolutions and U-Net architectures typically run well on ANE
- Hybrid approach: some components on ANE, transformer stays on GPU

**Target Components:**
| Component | Params | ANE Viability |
|-----------|--------|---------------|
| VAE Decoder | ~50M | Promising |
| VAE Encoder | ~50M | Promising |
| Spatial Upscaler | ~250M | Worth trying |
| Temporal Upscaler | ~65M | Worth trying |

## Constraints

- **Platform**: macOS with Apple Silicon (M1/M2/M3/M4) — ANE only available here
- **Framework**: Core ML for ANE access, must bridge with existing MLX pipeline
- **Compatibility**: Must not break existing MLX-only functionality
- **Success Bar**: Any measurable speedup on any component is a win

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Target VAE + upscalers only | Transformer too large for ANE | — Pending |
| Benchmark in isolation first | Prove speedup before integration complexity | — Pending |
| Hybrid MLX + Core ML approach | ANE can't run full pipeline | — Pending |

---
*Last updated: 2026-01-23 after initialization*
