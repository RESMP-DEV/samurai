# samurai agent instructions

## Purpose and scope

`samurai` is the RESMP-DEV VOID+Model execution repository. The canonical
runtime fuses aligned mask probability/logit tensors from multiple experts,
writes primary and affected masks, combines them into VOID quadmask sequences,
and invokes a separately pinned Netflix VOID checkout. SAM 3.1 is the default
anchor, not a hard-coded authority. Research execution belongs here; vendored
SAM 2 and upstream SAMURAI material under `third_party/` is frozen comparison
material, not an import target or product path.

`AGENTS.md` is the only instruction file. `CLAUDE.md` is a relative symlink to
it. Preserve the upstream fetch-only remote policy and never create GitHub
Actions workflows or publish through an external upstream.

## Architecture contracts

- Keep canonical Python in `src/samurai/`. It must not import `sam2`; SAM 3.1
  is imported only in the execution command after CUDA and dependency checks.
- Keep media encoding lossless for masks: gray frames are H.264 `yuv444p` at
  `qp=0`. Do not use chroma-subsampled mask encodings.
- Preserve the VOID quadmask values exactly: `0` remove, `63` overlap, `127`
  affected, `255` keep. The first frame clears affected-only pixels.
- Treat expert fusion as a numerical contract: probability inputs are converted
  to logits before temperature scaling; weights are normalized; dynamic weights
  are recorded per frame; and tensor dimensions/frame counts must align exactly.
- Keep spatial fusion bounded. Batch size is reduced according to frame pixels
  and `max_batch_pixels`; command experts receive only the current batch; output
  masks are written incrementally.
- Native multimodal text APIs are not mask models. A command adapter must return
  normalized box/polygon coordinates and confidence values, which samurai
  rasterizes into probability grids before fusion. Never present adapter prose
  as calibrated pixels.
- Treat SAM 3.1 and VOID as separate execution lanes. SAM 3.1 uses Python
  3.12, PyTorch 2.10/CUDA 12.6+ and Meta's SAM License; VOID pins its own
  PyTorch 2.7.1 environment in its reference checkout. Do not force them into
  one environment or silently substitute one model for the other.
- External model repositories and checkpoints are immutable inputs. Record
  exact revisions/paths in receipts; do not edit them from this repository.
- `third_party/` is dated observation only. Changes there are limited to
  provenance or policy cleanup and never repair product behavior.

## Work records and evidence

Every non-trivial state change or durable finding gets a receipt. Check
receipts are written under ignored `build/reports/<mode>/receipt.json`.
Sanitized durable evidence and work-log entries live in tracked `evidence/`
and `docs/architecture.md`. A receipt records exact source revision, dirty
state, commands, outcomes, elapsed time, scope, and non-claims. Worker reports
are evidence, never acceptance by themselves.

Claims use four states precisely: `implemented` means source exists;
`exercised` means a real command ran; `measured` means a metric was captured;
`proposed` means no executable path exists yet. Do not claim CUDA model
execution without a live SAM 3.1 or VOID run and artifact digest.

## Checks

Run `just check` for scaffold and model-boundary contracts. Run `just quality`
for Ruff, format, ty, and tests. Focused development may run a subset, but the
integrating owner reruns both after combined changes. Tests may exercise local
ffmpeg media encoding but must not download checkpoints or call external APIs.

Use `uv` for Python. The base environment contains the CLI only. Install the
SAM 3.1 lane with `just setup-sam31`; install the VOID reference environment in
its own checkout according to that pinned revision. Never use pip directly.

## Ownership

| Area | Owner | Boundary |
| --- | --- | --- |
| CLI and model adapters | `src/samurai/` | Prompt contracts, video I/O, errors |
| Expert fusion | `src/samurai/ensemble.py` | Weighted sigmoid field, dynamic weights, support, top-k, batching |
| AIMLAPI expert adapter | `scripts/aimlapi_expert.py` | Frame-batch request/response and coordinate-region contract |
| Verification | `scripts/` | Check domains, tests, receipt format |
| Runtime examples | `configs/` | Portable examples only, no private paths |
| Canonical plan/status | `docs/architecture.md` | One architecture and work log |
| Frozen references | `third_party/` | Unmodified historical evidence |
