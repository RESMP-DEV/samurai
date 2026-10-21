# Contributing to samurai

## Workflow

1. Read `AGENTS.md` and [architecture](docs/architecture.md), then inspect
   `git status --short` and the relevant status row before editing.
2. Keep product code in `src/samurai/`, checks in `scripts/`, examples in
   `configs/`, and dated evidence in `evidence/`. Do not create parallel
   roadmap/status documents.
3. Preserve unrelated dirty work. Use a separate worktree for multi-file
   implementation and integrate only after reviewing the combined diff.
4. Python code targets 3.12 with modern typing. Use `uv`, never pip. Formatting
   and imports follow Ruff; public functions have typed signatures and explicit
   errors.
5. Keep `third_party/` immutable. If historical material must change, record
   why in the architecture work log and keep the change policy/provenance-only.

## Required checks

| Change | Required commands | Reuse rule |
| --- | --- | --- |
| Any source/docs/config change | `just check`, `just quality` | Rerun both after combined changes |
| CLI adapter only | `just quality` plus focused pytest test | Full check still gates integration |
| Mask/media semantics | `just quality` plus a real ffmpeg round-trip test | A fake predictor never proves encoding |
| SAM 3.1 execution claim | `samurai doctor --strict` and one real CUDA run | Source tests never substitute |
| VOID execution claim | Validated real sequence and output digest/duration | Dry run never substitutes |

Each check receipt is written under ignored `build/reports/`. Receipts are
removed at startup, so an interrupted run cannot leave a stale PASS. Startup
failures still write a failure receipt. Do not commit machine-generated
`build/` receipts.

## Work records

Durable findings and state changes add a tracked evidence artifact or dated
architecture work-log entry. Record:

- exact UTC/local timestamp and agent (`Codex`, worker, human);
- repository revision and dirty state;
- source URLs and upstream revisions/checkpoint digests;
- commands, exit codes, timings, and platform/runtime;
- output artifact paths and digests for real media;
- a scope sentence naming what the run does **not** prove.

Worker output is evidence. The integrating owner must inspect the diff and run
the applicable checks before accepting it.

## Model experiments

Never use synthetic media or proxy metrics as final experiment evidence. Local
fixtures are only development tests. A real qualification run must use real
input media, the pinned checkpoint, the declared device, and retained output.
Record failures and resource limits as observed; do not silently switch model,
device, precision, or pass count.
