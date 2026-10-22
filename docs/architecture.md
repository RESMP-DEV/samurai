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
| Expert calibration | `src/samurai/calibration.py` | Label-free Bernoulli Dawid-Skene EM, bounded sampling, confidence weights, generated config |
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
| Multi-model spatial ensemble | implemented/exercised with synthetic contracts and a real two-frame three-family panel | Real APIs, tensors, calibration, fusion, and VOID-reference metrics; no full-video or multi-clip quality claim |
| Label-free expert calibration | implemented/exercised with synthetic aligned tensors | Agreement metrics on synthetic panels; no real-model or live-run claim |
| AIMLAPI/OpenAI-compatible coordinate adapter | implemented/exercised through GMI | Payload/parser tests plus one two-frame GMI vision call; no AIMLAPI call |
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
| Label-free confidence weights | Dawid-Skene agreement weights from `samurai calibrate` | Promote to the default fusion weight only with a real expert panel and measured interaction-case benefit |
| AIMLAPI model panel | MiMo V2.5, Gemini 3.8 Flash, and Dola Seed 2.0 Pro examples | A live run must pin served model identity and retain sanitized structured responses |
| SAM 3.1 FlashAttention 3 | Disabled by default for portability | Enable only after a CUDA A/B run proves correctness and speed with the pinned checkpoint |
| VOID Pass 2 | Implemented command path, optional execution | Run only when temporal-consistency metrics justify the extra GPU pass |
| Apple MPS route | Not claimed | Requires a source-reviewed SAM 3.1 MPS fork and live output equivalence |
| Similar-repository expansion | Research evidence records candidates | Add code only when it supplies a bounded executable capability |
| NVFP4 low-precision lane | Source-verified candidate set recorded; native FP4 GEMM measured on SM120 at the models' own shapes, but W4A4 is required for the speedup and the FP4 weight error does not survive the mask decision boundary intact | Promote only after a real checkpoint is calibrated with trimmed activation amax, a W4A4 runtime replaces the dequantizing weight-only path, and mask agreement on real interaction cases is measured against a BF16 baseline |
| Calibration corpus for the low-precision lane | No corpus defined yet | Define the frame and prompt set, the trimmed amax rule, and the accepted error budget before any quantized artifact is published |
| Hosted SAM 3.1 lane | Meta Model API serves `sam-3.1` at `https://api.meta.ai/v1` over the OpenAI-compatible Responses API, returning boxes and binary masks per frame | Build the adapter and measure mask fidelity against the local SAM 3.1 lane on the same clips before preferring either path |
| Quantization need at all | Cloud-capability-first: quantify BF16, FP8/W8A8, and NVFP4/W4A4 against the selected target runtime; local attached GPUs are not feasibility or sizing limits | Require paired end-to-end quality, latency, memory, cost, and fixed-seed digest evidence before promoting a low-bit artifact |

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
- Fusion and calibration share one probability transform, so a weight fitted
  in one domain applies to the other.
- Latent classes stay label-free. The positive class follows the panel mean
  vote convention and is never presented as ground truth.
- Calibration requires at least three aligned tensor experts. Two label-free
  raters cannot identify skill separately from class polarity.
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

### 2026-10-21, label-free expert calibration

- Added `src/samurai/calibration.py`: a bounded Bernoulli Dawid-Skene EM
  estimator over binary latent classes, with per-expert sensitivity and
  false-positive rate, beta smoothing, a convergence tolerance, and a maximum
  iteration count.
- Expert reports enter as a sign plus a confidence weight, so a report near
  0.5 moves the latent posterior far less than a decisive report. The
  estimator reduces to classic Dawid-Skene at unit confidence.
- The latent positive class is oriented toward the panel mean vote, with each
  expert's agreement and the first expert's anchor agreement reported
  separately. No ground truth is claimed.
- Added the `samurai calibrate` subcommand with JSON output, a generated
  ensemble config, a report path, and an optional latent-probability tensor.
  The generated config preserves every path, prompt, temperature, and the
  fusion block and replaces only the weights.
  Calibration records per-frame latent/support/disagreement summaries and
  SHA-256 provenance for the source config and every expert tensor.
- Extracted the shared probability transform into
  `ensemble.expert_probabilities` so fusion and calibration cannot diverge.
- Focused tests cover latent recovery, expert ranking, pairwise summaries,
  temperature handling, iteration bounds, degenerate panels, config
  round-tripping, and fail-closed input handling. Validation lives in
  [calibration evidence](../evidence/expert-calibration-2026-10-21.json).

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
- Validation and worker outcomes are consolidated in
  [spatial expert ensemble evidence](../evidence/spatial-expert-ensemble-2026-10-21.json).

### 2026-10-21, local constituent quantization survey

- Surveyed every locally present constituent of the VOID+Model stack for a
  low-precision path: the pinned Netflix VOID checkout, the pinned Meta SAM 3
  checkout, the frozen SAM 2/SAMURAI tree, the canonical fusion and
  calibration core, and the hosted AIMLAPI experts.
- Recorded that neither upstream tree carries any quantization support. VOID's
  existing `model_cpu_offload_and_qfloat8` mode is a weight cast to
  `torch.float8_e4m3fn` with an autocast wrapper, not an FP8 tensor-core GEMM
  path. SAM 3 and the frozen SAM 2 tree contain no quantizer, no observer, and
  no `scaled_mm` call at all.
- Counted parameters from safetensors headers read over HTTP range requests
  without downloading weights. The VOID Pass 1 and CogVideoX transformers are
  5.57B parameters with 85.4% GEMM-shaped weights, the T5-XXL encoder is 2.50B
  with 100% GEMM-shaped weights, and the CogVideoX VAE is 215M with 0%
  GEMM-shaped weights because it is entirely convolutional.
- Measured NVFP4 directly on Blackwell SM120 at those models' own GEMM
  shapes. The native kernel
  `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1` was
  observed and reached 206-279 TFLOP/s against 33-46 TFLOP/s for BF16, a
  5.1x to 7.4x speedup with roughly 0.134 relative L2 error.
- Established that the speedup depends on quantizing activations as well as
  weights. A weight-only NVFP4 path dequantizes back to BF16 before the GEMM
  and measured 0.99x to 1.05x, so it buys memory but no compute.
- Established that the two kinds of calibration behave differently. The weight
  per-tensor global scale is not the error driver, because the 16-element block
  scale absorbs weight outliers; the activation per-tensor scale is the error
  driver, and one outlier batch in a calibration set inflated it enough to
  produce 0.90 relative error against 0.43 for trimmed calibration.
- Measured that the FP4 weight error does not survive the segmentation
  decision boundary intact: thresholded masks agreed on about 97% of pixels
  with IoU between 0.92 and 0.94 at realistic layer shapes.
- Recorded that `facebook/sam3.1` is a manually gated repository that this
  machine cannot read, that an ungated mirror republishes a byte-identical
  original checkpoint plus an existing NVFP4 artifact, and that this artifact
  is a storage-size quantization with a static scale rule, a null per-tensor
  error report, and a loader that dequantizes before execution.
- Evidence, per-component verdicts, and non-claims are consolidated in
  [quantization survey evidence](../evidence/quantization-survey-2026-10-21.json).
  No checkpoint was downloaded and no model was executed in this survey.

### 2026-10-21, SAM 3.1 hosted API and checkpoint access survey

- Established that Meta operates a first-party hosted API for SAM 3.1, the
  Meta Model API, with model ID `sam-3.1`, base URL `https://api.meta.ai/v1`,
  and the Responses API plus the Media segmentation endpoint. It is billed
  per segmented image or frame, USD 2.50 per 1,000 images and USD 0.20 per
  1,000 video frames, at a rate limit of 3,000 requests per minute per team.
  Client libraries exist for parsing the mask grammar, including the
  zero-dependency Python package `meta_sam_parser`.
- Established that the official `facebook/sam3.1` checkpoint repository is
  manually gated and this account is not on the authorized list, so the
  official weights remain unreadable here. Measured that the official
  checkpoint is FP32: the archive carries only `FloatStorage` and
  `ComplexFloatStorage`, with 874,365,676 elements at 4 bytes each matching
  the tensor blob total inside the 3,502,755,717-byte file.
- Surveyed third-party hosting. fal.ai lists SAM 3 only, as
  `fal-ai/sam-3/image` plus `video`, `image-rle`, and `3d-objects`, at USD
  0.005 per request, licensed commercial. Segmind lists `sam3-image` and
  `sam3-video`. Replicate has no SAM 3 or SAM 3.1 page under any tested slug.
  AWS Bedrock, Azure AI Foundry, Together, Fireworks, Novita, Wavespeed,
  DeepInfra, and Workers AI showed no SAM 3.1 listing, though several of
  those catalogs could not be enumerated without an API key.
- Confirmed the SAM 2.1 fallback is fully open: four official ungated
  Apache-2.0 repositories, `facebook/sam2.1-hiera-tiny`, `-small`,
  `-base-plus`, and `-large`, each with `model.safetensors` and a matching
  `.pt`, plus the same weights on Meta's own CDN at
  `https://dl.fbaipublicfiles.com/segment_anything_2/092824/`. The official
  SAM 2.1 safetensors are FP32.
- Confirmed the SAM License, which governs both SAM 3 and SAM 3.1, is not
  research-only and not non-commercial: it grants use, reproduction,
  distribution, and modification with derivative works, requires attribution
  only for published research, and restricts reverse engineering and
  ITAR-controlled or military end uses. SAM 2.1 is Apache-2.0.
- Recorded that pre-quantized third-party SAM 3.1 artifacts already exist, so
  first-party quantization of SAM 3.1 is not required to obtain a low-precision
  checkpoint: `unsloth/sam3.1` publishes BF16, INT8, INT8-ConvRot, and FP8
  variants with publisher-reported mask IoU of 0.9971 to 0.9984 against BF16,
  and further INT8/W4A4, NVFP4, FP8, FP16, and MLX conversions exist under
  other orgs.
- Evidence, exact file names and sizes, and non-claims are consolidated in
  [SAM access terms evidence](../evidence/sam-api-checkpoint-access-2026-10-21.json).
  No hosted endpoint was called and no checkpoint was downloaded or executed.

### 2026-10-21, hosted SAM 3.1 lane identified

- Corrected the working assumption that SAM 3.1 has no API. Meta operates the
  Meta Model API and serves SAM 3.1 as model ID `sam-3.1` at
  `https://api.meta.ai/v1`, over the same OpenAI-compatible Responses API that
  the rest of Meta Model API uses. The endpoint was confirmed live by an
  `invalid_api_key` response rather than a 404.
- Recorded the contract: a short noun phrase plus an image or a video returns,
  per frame, one box and one binary mask per matched object, in the
  special-token text grammar. Video accepts MP4 and MOV with H.264, H.265, or
  MPEG-4. Billed at USD 2.50 per 1,000 segmented images and USD 0.20 per 1,000
  segmented frames, with a 3,000 requests-per-minute-per-team limit and no
  token limit.
- Noted that the response is already a binary raster, so a hosted lane can
  enter the existing tensor-expert contract in `src/samurai/ensemble.py`
  without a checkpoint, a CUDA host, or quantization work.
- Recorded the independent correction that the VOID transformer dimensions are
  42 layers at inner dim 3072 with 5.57B parameters, read from the live
  checkpoint config rather than the local class defaults, and that VOID's
  `model_cpu_offload_and_qfloat8` mode performs no FP8 compute because the
  load dtype is undone by the immediate `weight_dtype` cast.
- Evidence, verified facts, and non-claims are consolidated in
  [SAM access terms evidence](../evidence/sam-api-checkpoint-access-2026-10-21.json).
  No hosted endpoint was called with a credential and no checkpoint was
  downloaded or executed.

### 2026-10-22, real three-family expert-panel pilot

- Inventoried the local media and research caches. Four YouTube-derived videos
  are present, but only the Irving Younger lecture is a strong person-mask
  candidate; the 2.2 GiB GPU MODE Phonon dataset is audio/ASR Parquet with 16
  source IDs, not visual masks.
- Found a stronger local visual corpus in the pinned VOID reference sample
  cache: seven aligned input/quadmask sequences, including a 60-frame
  trampoline clip with a moving person, pose changes, scale changes, and
  podium overlap.
- Ran the first real three-family panel on original trampoline frames 0 and 10:
  Grok 4.7 through `alt-vision`, GLM 5.3 Flash through `zai-vision`, and GMI's
  served DeepSeek V4 Flash Vision experiment through the OpenAI-compatible
  coordinate adapter. Each response was parsed into normalized boxes, rasterized
  by the canonical command-expert path, and retained as an aligned probability
  tensor.
- Label-free calibration converged in 12 iterations and produced nearly equal
  confidence weights. Unanimous-support fusion improved reference IoU over all
  three individual experts: 0.4620 fused versus 0.3975-0.4292 individually,
  with precision 0.4835 and recall 0.9123 against the VOID primary-mask
  reference.
- Repaired the OpenAI-compatible adapter request to send a normal User-Agent.
  A live GMI A/B proved the exact image payload receives Cloudflare HTTP 403
  error 1010 without it and succeeds with it.
- Full media, response IDs, digests, metrics, worker provenance, and non-claims
  are consolidated in
  [three-expert real-video pilot evidence](../evidence/three-expert-real-video-pilot-2026-10-22.json).
  This is a two-frame mechanism pilot, not a full-video quality claim; no SAM
  3.1 checkpoint or VOID generative model ran.

### 2026-10-22, upstream multi-model provenance audit

- Audited frozen original SAMURAI at `76ba1959` and its upstream history. The
  2024-11-20 introduction added Kalman motion scoring and confidence-filtered
  memory selection inside one SAM2 predictor. Its weighted term combines the
  model's candidate IoU with a Kalman-predicted box IoU and then selects one of
  three masks from the same decoder; it does not aggregate outputs from
  multiple segmentation models.
- Audited Netflix VOID at `e3914f8` and its initial 2026-03-30 public
  `VLM-MASK-REASONER` commit. VOID did ship a heterogeneous multi-stage
  pipeline: SAM2 primary tracking, Gemini affected-object reasoning, and an
  exclusive SAM3-or-LangSAM affected-region mask lane. Stage 4 then maps the
  primary and affected semantic channels into the exact quadmask values.
- Therefore VOID had multi-model orchestration, but not ensemble fusion. No
  upstream lane independently estimates the same primary mask and combines
  probabilities, boxes, or logits across model families. VOID inference reads
  one prepared quadmask and Pass 2 refines Pass 1 latents rather than blending
  expert masks.
- Exact source anchors, upstream commit IDs, worker provenance, and non-claims
  are consolidated in
  [upstream multi-model provenance evidence](../evidence/upstream-multi-model-provenance-2026-10-22.json).

### 2026-10-22, fusion, VOID transform, quantization, and offload study

- Compared fusion policies on the retained three-family trampoline tensors. The
  shipped unanimous-support gate reached IoU `0.4620` against the VOID reference,
  above weighted fusion `0.4188`, the top-two capability intersection `0.4443`,
  and global label-free selection `0.4029`; a per-frame reference oracle reached
  only `0.4563` on this two-frame panel. The result is a mechanism observation,
  not a general quality ranking.
- Found an immediate policy defect: an expert below `support_threshold` casts no
  support vote but still retains full weight in the average. Explicit abstention,
  active-mass renormalization, and a mass floor are the first policy changes to
  implement. Capability scalar weighting alone selected the same pixels in the
  pilot.
- Established the high-sensitivity design direction: one vote per lab lineage,
  structured expert roles, capability-weighted quorum mass, high-disagreement
  escalation, and a second-phase critic that sees prior masks and can confirm or
  modify a candidate. No critic adapter contract or routing path exists yet.
- Verified the exact samurai-to-VOID transform. VOID area-resizes and inverts the
  quadmask, then Pass 1 hardcodes `use_trimask=True`: both `0` remove and `63`
  overlap map to `1.0`, `127` affected maps near `0.498`, and keep maps to zero.
  Pass 2 hardcodes `use_trimask=False`, so all non-keep classes become one binary
  mask. Proposed temporal consensus, flow propagation, connected-component, and
  per-class morphology/feathering transforms are not implemented.
- Refreshed quantization precedents: CogVideo ships TorchAO INT8/FP8 examples;
  Diffusers exposes TorchAO hooks; SVDQuant, QVGen, Q-VDiT, and DeltaQuant cover
  W4A4, low-rank outliers, timestep/video-aware calibration, and QAT; hosted SAM
  3.1 INT8/FP8/W4A4 artifacts already exist. These are external results, not
  samurai qualifications.
- Kept the quantization order conservative: qualify the stock-loader T5 encoder
  first, validate a prequantized or hosted SAM 3.1 lane on real clips second,
  and only then build a custom VOID DiT W4A4 path with timestep-aware activation
  calibration. VOID's custom loaders bypass Diffusers hooks, its `qfloat8` mode
  immediately returns to BF16, and its convolution-heavy VAE is a poor low-bit
  GEMM candidate.
- Audited RAM/VRAM offloading at pinned VOID `e3914f8`. Model and sequential CPU
  offload are real, while `model_cpu_offload_and_qfloat8` is a cast-and-undo
  no-op. The principal runtime controls are `temporal_window_size`,
  `max_video_length`, and CFG batch size. Pass 2 moves all components to GPU
  before installing offload hooks, defaults to non-low-memory checkpoint loading,
  has no alternative offload mode, and its batch shell passes an unsupported
  `--gwf_dir`. VAE tiling/slicing helpers are never called.
- Probed both compute hosts read-only. Strix has an 8,151 MiB RTX 5070 Laptop
  with 5,306 MiB currently used; B550 has two 24,564 MiB 3090 Ti cards with
  22,430/21,636 MiB currently used. Both retain ample available host RAM. No
  checkpoint or inference was launched.
- Full findings, source anchors, refreshed external pages, live host observations,
  and non-claims are consolidated in
  [fusion and efficiency evidence](../evidence/fusion-mask-transform-quantization-offload-2026-10-22.json).

### 2026-10-22, SAM 3.1 conditioning and VOID benchmark study

- Verified the released SAM 3.1 multiplex request surface: text, points, boxes,
  object removal, and point correction are available, but direct binary-mask
  conditioning is not. The wrapper rejects `add_mask` for SAM 3.1 multiplex, and
  there is no native quadmask, boundary, or negative-space prompt. High-disagreement
  evidence must therefore enter as a point/box prompt or through a new trained
  adapter, not as a caller mask.
- Found Netflix's own lower-risk affected-region precedent: original VOID used
  the SAM 3 image model on VLM-selected affected nouns, gridified those masks,
  and wrote the `127` channel. The next conditioning design is SAM 3.1 multiplex
  for the primary anchor, the SAM 3 image lane for affected regions, and the
  existing downstream intersection rule for `63` overlap.
- Audited adaptation. The official trainer covers SAM 3 image fine-tuning and
  binary mask/presence losses, while a ready SAM 3.1 multiplex video recipe and a
  four-class semantic target do not exist. Widening SAM's one-channel semantic
  head to four quadmask classes is possible only as a larger proposed change.
- Verified official SAM benchmark surfaces: SA-Co/VEval with cgF1 and pHOTA,
  SA-Co image sets, YTVIS/HOTA tooling, and speed/compile scripts. SAM 3.1's
  publisher results include SA-V `30.5/58.7` cgF1/pHOTA and YT-Temporal-1B
  `52.9/70.7`. The SA-V and YT VEval domains are CC-BY-NC 4.0, so they are
  internal evaluation only, not client-model training data.
- Established that VOID's paper benchmark exists but the checkout does not ship a
  runnable benchmark. The paper used 75 real videos with human/VLM judging and a
  40-video paired synthetic set with PSNR, LPIPS, DreamSim, DINOv2, FVD, and a
  reference-aware VLM judge. The local benchmark should instead start with
  mask-only per-class IoU/Dice, boundary F1, flicker, and class-transition
  stability on the seven cached sequences, then generate held-out Kubric
  counterfactual pairs for full inpainting metrics.
- Defined the supervised VOID row as `rgb_full.mp4`, `rgb_removed.mp4`, exact
  four-value `mask.mp4`, metadata, and an optional Pass 2 warped-noise tensor.
  Quantization calibration adds sequence-level splits, media/prompt digests,
  timestep and CFG-branch coverage, and per-tensor/per-channel/block-16 activation
  statistics.
- Kept mixed precision deliberately non-uniform: T5 first, selective SAM
  INT8/W8A8 or FP8 second, VOID DiT FP8 before NVFP4 W4A4, and the CogVideoX VAE
  in BF16. B550's RTX 3090 Ti cards are compute capability 8.6 and have no FP8
  tensor cores; Strix is compute capability 12.0 but each pinned environment still
  needs a live kernel probe. INT8 weight-only remains a compatibility/residency
  option, not an acceleration claim.
- Full conditioning, benchmark, calibration-corpus, hardware, quality-gate, and
  non-claim details are consolidated in
  [SAM conditioning and benchmark evidence](../evidence/sam31-conditioning-void-benchmark-mixed-precision-2026-10-22.json).

### 2026-10-22, cloud-first QAT and benchmark hygiene correction

- Corrected compute planning to cloud-capability-first. Architecture, QAT,
  benchmark, and model-size choices are selected from the required numerical
  format, kernel, runtime, quality, and cost target. GPUs currently attached to
  local machines are operational observations only and are never feasibility or
  sizing ceilings. This rule is now explicit in `AGENTS.md`.
- Ranked adaptation difficulty. VOID is the strongest fine-tuning surface because
  it ships paired counterfactual-video training, exact quadmask conditioning,
  frozen VAE/T5, a trainable transformer, DeepSpeed recipes, and a separate
  Pass 2 warped-noise trainer. SAM 3 image fine-tuning is also supported, but
  SAM 3.1 multiplex has no ready video fine-tuning recipe and no direct dense
  mask-conditioning request.
- Verified that NVFP4 QAT is real ecosystem machinery rather than speculation.
  TorchAO provides `NVFP4FakeQuantizedLinear`, a quantized forward with
  dequantized/fake-quantized backward, and `QATConfig` prepare/convert steps.
  Weight-only NVFP4 QAT is unsupported: the route is W4A4. NVIDIA ModelOpt adds
  learnable-scale and low-rank outlier recipes, while QVGen provides direct
  3-4-bit video-DiT QAT precedent including CogVideoX.
- The missing piece is integration, not concept. Pinned VOID uses its own
  PyTorch 2.7.1 training environment; NVFP4 QAT requires a separate modern
  Blackwell-capable research lane with module swapping, teacher/distillation
  hooks, timestep-stratified sampling, separate CFG/pass statistics, and a
  conversion/export path. The pinned BF16 reference environment remains the
  teacher and acceptance baseline.
- The first QAT sequence is PTQ-before-QAT: establish a cloud BF16 teacher,
  measure timestep-aware NVFP4 PTQ, then train eligible transformer Linear
  layers with NVFP4 W4A4 fake quantization. Add ModelOpt-style scale learning,
  an SVDQuant/QVGen-style low-rank auxiliary, or Q-VDiT-style temporal
  distillation only after layer attribution shows which failure remains.
- Corrected benchmark-license reasoning. CC BY-NC may reduce incidental
  commercial redistribution, but it is not a train/test split and does not
  remove public test annotations. SAM's stronger rule is explicit: its
  benchmark is evaluation-only and must not be used for training. NC media also
  requires extra caution for client-facing adaptations.
- Added immutable corpus roles in `AGENTS.md`: train, calib, dev, and
  lockbox/eval are physically and cryptographically separated. No eval frame,
  mask, prompt, target, or derived tensor may enter gradient training, QAT
  calibration, expert calibration, prompt tuning, or hyperparameter search.
  Generated Kubric/HUMOTO corpora split by asset or source sequence and record
  source, license, role, and digest provenance.
- Full cloud-first QAT and anti-leakage findings are consolidated in
  [cloud-first QAT evidence](../evidence/cloud-first-qat-benchmark-hygiene-2026-10-22.json).
