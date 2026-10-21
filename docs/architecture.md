# samurai architecture and plan

## Philosophy

Correctness and reproducibility outrank model novelty, which outranks throughput.
The repository therefore separates a small auditable execution boundary from
immutable heavyweight model systems: samurai owns prompt contracts, numerical
expert fusion, mask semantics, sequence validation, receipts, and subprocess
boundaries; external model APIs and Netflix VOID own their model code, weights,
and API behavior. A new research idea enters as a pinned source or model
identity plus an executable contract, not as a copy of undocumented code.

## Ownership

| Boundary | Current owner | Contract |
| --- | --- | --- |
| User surface | `src/samurai/cli.py` | JSON results, explicit exit codes, no secret output |
| SAM 3.1 adapter | `src/samurai/sam31.py` | Absolute prompts normalized to request API; complete forward-frame mask video |
| Media/mask encoding | `src/samurai/media.py`, `src/samurai/quadmask.py` | Lossless gray yuv444p and exact VOID four-value semantics |
| Expert fusion | `src/samurai/ensemble.py` | Aligned tensors, temperature scaling, dynamic weights, support, top-k, bounded batches |
| Multimodal adapters | `scripts/aimlapi_expert.py` | One request per frame batch; normalized coordinate JSON only |
| VOID execution | `src/samurai/void.py` | Sequence validation, Pass 1 command, optional Pass 2 command |
| Verification | `scripts/check_*.py`, `scripts/tests/` | One check domain per script and stale-receipt-safe receipts |
| Historical code | `third_party/` | Read-only SAM 2/SAMURAI observation, never imported |

## Research baseline

The dated evidence is [research baseline 2026-10-21](../evidence/research-baseline-2026-10-21.json).
It records the official SAM 3 repository at `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`,
the `facebook/sam3.1` checkpoint repository, and Netflix VOID at `e3914f8f551dd4b880661991fd6b28cd1699a97a`.

Source-verified facts that shape the implementation:

- SAM 3.1 is the multiplex model. Its recommended builder is
  `build_sam3_predictor(version="sam3.1")`, followed by `handle_request`
  and `handle_stream_request`; it is not call-compatible with SAM 2's
  `build_sam2_video_predictor`, `init_state`, and `add_new_points_or_box`.
- SAM 3.1 point and box prompts use normalized coordinates. Boxes use
  normalized XYWH at the request boundary.
- The public SAM 3.1 video response exposes binary masks and per-object
  probabilities, not internal pixel logits. `samurai segment --scores-output`
  therefore emits a probability-support tensor formed by multiplying each binary
  mask by its object probability; it never claims calibrated boundary logits.
- The public predictor requires Python 3.12+, PyTorch 2.7+ (the pinned setup
  uses 2.10), and a CUDA 12.6+ device.
- VOID consumes a sequence directory containing `input_video.mp4`,
  `quadmask_0.mp4`, and `prompt.json` with a `bg` description. Quadmask values
  are `0`, `63`, `127`, and `255`.
- VOID Pass 1 is the base inpainting pass. Pass 2 consumes Pass 1 output,
  generates warped noise, and refines longer clips for temporal consistency.
- VOID's mask reasoner convention clears affected-only pixels on frame zero.

## Current status

| Claim | State | Evidence boundary |
| --- | --- | --- |
| Canonical SAM 3.1 adapter and CLI | implemented | Source and local API contract tests; no CUDA model turn |
| Primary-mask video generation | implemented/exercised against fake predictor | Real ffmpeg media round trips; no SAM checkpoint |
| VOID quadmask sequence builder | implemented/exercised | Tiny local videos; no VOID model call |
| Multi-model spatial ensemble | implemented/exercised with tensors and fake command experts | Real ffmpeg output; no external multimodal API or quality claim |
| AIMLAPI coordinate adapter | implemented | Payload/parser tests only; no live API call |
| VOID Pass 1/Pass 2 command orchestration | implemented | Path and command tests/dry run; no GPU run |
| SAM 2/SAMURAI historical tree | classified as frozen observation | Moved under `third_party/`; canonical imports rejected by check |
| SAM 3.1 live CUDA inference | proposed qualification | Requires compute host, access-approved checkpoint, artifact digest |
| VOID live GPU inference | proposed qualification | Requires pinned VOID env, base model, Pass 1/2 weights |

## Execution sequence

1. Configure the SAM 3.1 lane: `just setup-sam31`, export
   `SAMURAI_SAM31_CHECKPOINT`, authenticate to the approved Hugging Face
   checkpoint repository, and run `samurai doctor --strict` on a CUDA host.
2. Produce aligned expert tensors and run `samurai ensemble` with temperatures,
   weights, minimum support, and spatial top-k settings, or use direct SAM 3.1
   when a single anchor is sufficient.
3. Segment the primary object directly when no ensemble is needed:
   `samurai segment VIDEO --text 'person' --output black-mask.mp4`.
   Point and absolute XYXY box prompts are also supported.
4. Provide or generate an affected-region mask, then run `samurai quadmask`
   with the source video, primary mask, affected mask, background prompt, and
   destination sequence directory.
5. Configure a pinned VOID checkout and model paths from
   `configs/void.example.json`; validate first with `samurai void --dry-run`.
6. Run `samurai void` in the VOID environment. Pass 2 runs only when its
   checkpoint is configured.
7. Digest and retain the source video, mask videos, output videos, command,
   model revision/checkpoint digest, runtime, and non-claims in a dated receipt.

## Open choices

| Topic | Current position | Decision gate |
| --- | --- | --- |
| Affected-region reasoning | User supplies affected mask; no VLM/GUI path | A reviewed model must produce calibrated `127` regions and beat a baseline on real interaction cases |
| Dynamic expert weighting | Entropy confidence by default | Promote another estimator only with real interaction-case calibration and quality evidence |
| AIMLAPI model panel | MiMo V2.5, Gemini 3.8 Flash, and Dola Seed 2.0 Pro examples | A live run must pin served model identity and retain sanitized structured responses |
| SAM 3.1 FlashAttention 3 | Disabled by default for portability | Enable only after a CUDA A/B run proves correctness and speed with the pinned checkpoint |
| VOID Pass 2 | Implemented command path, optional execution | Run only when temporal-consistency metrics justify the extra GPU pass |
| Apple MPS route | Not claimed | Requires a source-reviewed SAM 3.1 MPS fork and live output equivalence |
| Similar-repository expansion | Research evidence records candidates | Add code only when it supplies a bounded executable capability |

## Delivery order

| Step | Exit evidence |
| --- | --- |
| 1. Project scaffold | `just check` and `just quality` pass; work receipt records source and non-claims |
| 2. SAM 3.1 CUDA qualification | Live doctor, one real prompt/video, complete mask video, checkpoint digest, timing |
| 3. Expert-panel qualification | At least three real model identities, aligned tensor provenance, live fused mask, and disagreement report |
| 4. VOID sequence qualification | Real primary/affected masks pass validation and produce a lossless quadmask |
| 5. VOID Pass 1 qualification | One real sequence, output video/digest, runtime and VRAM observation |
| 6. Optional Pass 2 qualification | Same-sequence temporal metric and runtime comparison; no blanket quality claim |

## Contracts to preserve

- Canonical code never imports the frozen SAM 2 tree.
- A model invocation receipt binds exact source/dependency/checkpoint identity.
- Mask values and first-frame convention are compatibility boundaries.
- Probability inputs pass through logit conversion before temperature scaling;
  no expert is treated as a binary authority.
- External multimodal text responses enter only after reproducible coordinate
  rasterization.
- Media transcodes are tests only; actual experiments never use synthetic
  media as substitute evidence.
- External repositories stay fetch-only and immutable.

## Work log

### 2026-10-21, project setup

- Classified the imported upstream SAMURAI/SAM 2 code as dated observation and
  moved it under `third_party/`.
- Added the `samurai` CLI, SAM 3.1 request adapter, lossless mask encoding,
  VOID quadmask builder, VOID sequence validator, and Pass 1/Pass 2 command
  construction.
- Added instruction, setup-contract, quality, receipt, and focused tests.
- Recorded the source research baseline and removed the inherited GitHub
  Actions workflow. No SAM 3.1 checkpoint or VOID model execution is claimed.
- Validation and worker outcomes are consolidated in
  [project setup evidence](../evidence/project-setup-2026-10-21.json).

### 2026-10-21, spatial expert triangulation

- Added weighted sigmoid fusion, per-frame entropy confidence, minimum expert
  support, spatial top-k, and pixel-budgeted dynamic batching.
- Added memory-mapped tensor experts and command experts that rasterize
  normalized boxes or polygons into probability grids.
- Added an opt-in AIMLAPI adapter using three native-multimodal examples:
  Xiaomi MiMo V2.5, Google Gemini 3.8 Flash, and ByteDance Dola Seed 2.0 Pro.
- Rebuilt the public entry point as a generic VOID+Model toolkit. Local tests
  use synthetic tensors, fake command experts, and real ffmpeg encoding; no
  external API or model-quality claim is made.
