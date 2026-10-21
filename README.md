# samurai, a VOID+Model toolkit

`samurai` turns video inpainting into a reproducible multi-model execution
system. It combines calibrated mask tensors from specialized segmentation
models with coordinate regions from native multimodal models, fuses them as a
weighted spatial ensemble, and converts the result into the exact four-value
quadmask consumed by Netflix VOID.

The project is model-neutral. SAM 3.1 is the default high-precision anchor, but
the ensemble boundary accepts logits or probabilities from any model that can
emit an aligned `frames × height × width` tensor. Text-only multimodal APIs are
admitted through a command adapter that returns normalized boxes or polygons;
the adapter rasterizes those regions into probability grids before fusion.

## Spatial expert triangulation

Let \(M_k(x,y,t)\) be model \(k\)'s logit tensor at pixel \((x,y)\) in frame
\(t\). Probability inputs are first converted to logits and then scaled by the
expert temperature \(T_k\). The fused field is:

\[
P(x,y,t)=\sum_k \tilde w_k(t)\,
\sigma\!\left(\frac{\logit(p_k(x,y,t))}{T_k}\right)
\]

The configured weights \(w_k\) are normalized to \(\tilde w_k\). With
`dynamic_weights: "entropy"`, samurai estimates each expert's per-frame binary
entropy over the pixel grid, converts it to a confidence factor, and renormalizes
the weights for that frame. This reduces the influence of an expert that is
uniformly uncertain without giving it a hard binary veto.

Two complementary defenses suppress hallucinated regions:

- `min_expert_support` requires support from at least N experts above
  `support_threshold`.
- `mode: "top_k"` selects the highest-scoring eligible pixels, with `top_k_ratio`
  interpreted as a pixel fraction when at most 1.0 and as an absolute pixel
  count when greater than 1.0.

The alternative `mode: "threshold"` emits every eligible pixel whose fused
probability reaches `threshold`.

## Dynamic batching

Expert inference is grouped by frame, never by an unbounded whole video. The
dynamic planner derives the safe batch size from the requested batch and the
resident-pixel budget:

Here \(B_{\text{config}}\) is the requested batch size and \(P_{\max}\) is the
configured pixel budget; samurai chooses
\(B=\min(B_{\text{config}},\lfloor P_{\max}/HW\rfloor)\), with a minimum of one
frame.

This bounds resident expert tensors while still sending multiple frames to a
multimodal API adapter when the pixel budget allows it. Tensor experts are
memory-mapped `.npy` files, command experts receive only the current frame
batch, and the output mask is written incrementally rather than retained as a
full-video tensor.

## Example expert panel

[`configs/ensemble.aimlapi.example.json`](configs/ensemble.aimlapi.example.json)
shows a SAM 3.1 anchor plus three AIMLAPI native-multimodal experts:

| Expert role | Example model | Adapter inputs | Why it is useful |
| --- | --- | --- | --- |
| Semantic interaction | `xiaomi/mimo-v2.5` | image and text through AIMLAPI; the model family is native omni-modal | Cross-modal interaction and long-video reasoning |
| Frame reasoning | `google/gemini-3-8-flash` | image and text | Fast image understanding and structured coordinate output |
| Artifact critic | `bytedance/dola-seed-2-0-pro` | image, video, and text | High-quality multimodal reasoning for artifacts and interaction boundaries |

These APIs return text, not calibrated pixels. `scripts/aimlapi_expert.py`
therefore sends a frame batch, requires normalized box/polygon JSON with
confidence values, and rasterizes the response into a probability grid. Set
`AIMLAPI_KEY` to use that adapter. No API call is required by the local test
suite.

## Pipeline

1. Produce a SAM 3.1 probability-support tensor aligned to the input video.
2. Add two or more tensor experts or command adapters in an ensemble config.
3. Fuse the experts:

```sh
uv run samurai ensemble \
  --config ensemble.json \
  --output primary-mask.mp4 \
  --report ensemble-report.json
```

4. Produce or ensemble an affected-region mask.
5. Build the VOID sequence:

```sh
uv run samurai quadmask \
  --input-video input.mp4 \
  --primary-mask primary-mask.mp4 \
  --affected-mask affected-mask.mp4 \
  --background-prompt 'An empty table' \
  --sequence-dir sequences/my-video
```

6. Validate and run the pinned Netflix VOID lane:

```sh
uv run samurai void --config void.json --dry-run
uv run samurai void --config void.json
```

## Direct SAM 3.1 segmentation

```sh
uv run samurai segment input.mp4 \
  --text 'person' \
  --output primary-mask.mp4 \
  --scores-output sam31-probabilities.npy
```

Point and absolute XYXY box prompts are also supported. The direct path uses
SAM 3.1's official request/stream API and verifies that every expected frame is
returned before writing a mask. The optional score tensor records each binary
mask multiplied by SAM's per-object probability; the public predictor does not
expose its internal pixel logits, so this is object-probability support rather
than a claim of calibrated SAM boundary logits.

## Setup

```sh
just setup
just check
just quality
```

Install the optional SAM 3.1 lane with `just setup-sam31`. Install the pinned
Netflix VOID environment with `just setup-void`.

SAM 3.1 requires a CUDA 12.6+ Linux execution host and access to
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1). Current matching
Triton wheels are Linux-only, so the Mac doctor correctly reports the full SAM 3
module as non-importable even when the distribution is installed. VOID uses its
own PyTorch 2.7.1 environment, base CogVideoX model, and Pass 1/Pass 2
checkpoints.

## Provenance and status

- Official SAM 3: `facebookresearch/sam3` at
  `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`.
- Netflix VOID: `Netflix/void-model` at
  `e3914f8f551dd4b880661991fd6b28cd1699a97a`.
- Historical SAMURAI/SAM 2 material is frozen under `third_party/` and never
  imported by canonical code.
- Current implementation and qualification boundaries live in
  [architecture and status](docs/architecture.md).
- Dated evidence starts with
  [the research baseline](evidence/research-baseline-2026-10-21.json).

No AIMLAPI, SAM 3.1 checkpoint, VOID, or CogVideoX inference claim is made by
the local tests. Live qualification requires real media, the selected API/model
identities or checkpoint digests, retained outputs, and a dated receipt.
