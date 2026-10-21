# samurai

`samurai` is a SAM 3.1 and Netflix VOID execution repository. It provides a
small CLI boundary for promptable video segmentation, VOID quadmask creation,
and pinned Pass 1/Pass 2 execution. The historical SAMURAI/SAM 2 fork remains
under `third_party/` as frozen comparison material; it is not the runtime.

## Current capability

- SAM 3.1 text, point, and box prompt normalization through the official
  `build_sam3_predictor(version="sam3.1")` request API.
- Primary-mask video writing with lossless H.264 `yuv444p` at `qp=0`.
- VOID sequence creation with the exact four-value quadmask contract:
  `0` remove, `63` overlap, `127` affected, `255` keep.
- Sequence/model-path validation and Netflix VOID Pass 1 plus optional Pass 2
  command construction.
- Doctor output that distinguishes base, SAM 3.1 CUDA, and VOID readiness.

Current implementation and qualification boundaries are in
[architecture and status](docs/architecture.md). No live SAM 3.1 or VOID GPU
run is claimed by the initial scaffold.

## Setup

```sh
just setup
just check
just quality
```

The base environment is intentionally lightweight. Install the optional SAM 3.1
lane separately:

```sh
just setup-sam31
export SAMURAI_SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt
uv run samurai doctor
```

To clone the pinned Netflix VOID reference into ignored `runtime/` and build its
separate Python 3.12 environment, run `just setup-void`.

SAM 3.1 requires Python 3.12+, a CUDA 12.6+ device, and access to
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1). Its model code and
checkpoints are governed by Meta's SAM License. The lane also imports Triton,
whose current matching wheels are Linux-only, so doctor reports this Mac as not
SAM 3.1-importable even after dependency installation. Netflix VOID uses its own
PyTorch 2.7.1 environment, base CogVideoX model, and Pass 1/Pass 2 checkpoints;
configure those paths from [`configs/void.example.json`](configs/void.example.json).

## Usage

Create a primary mask with a text prompt:

```sh
uv run samurai segment input.mp4 \
  --text 'person' \
  --output black-mask.mp4
```

Use absolute coordinates for interactive prompts:

```sh
uv run samurai segment input.mp4 \
  --point 320,240 --point-label 1 \
  --box 100,120,420,500 \
  --frame-index 0 \
  --output black-mask.mp4
```

Combine primary and affected masks into a complete VOID sequence:

```sh
uv run samurai quadmask \
  --input-video input.mp4 \
  --primary-mask black-mask.mp4 \
  --affected-mask grey-mask.mp4 \
  --background-prompt 'An empty table' \
  --sequence-dir sequences/my-video
```

Validate and execute VOID:

```sh
uv run samurai void --config void.json --dry-run
uv run samurai void --config void.json
```

## Provenance and research

- Upstream SAMURAI fork: [`yangchris11/samurai`](https://github.com/yangchris11/samurai)
  at `76ba195984892b0d1e3db5d9c9f90bb62175680a`.
- Official SAM 3 code: [`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)
  pinned at `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`.
- Netflix VOID reference: [`Netflix/void-model`](https://github.com/Netflix/void-model)
  inspected at `e3914f8f551dd4b880661991fd6b28cd1699a97a`.
- Dated evidence: [research baseline](evidence/research-baseline-2026-10-21.json).

See [contributing](CONTRIBUTING.md) for checks, receipts, and evidence rules.
