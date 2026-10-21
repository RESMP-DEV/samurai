"""Weighted spatial expert fusion for mask-coordinate tensors."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .media import GrayVideoWriter, extract_color_frames, probe_video


class EnsembleError(RuntimeError):
    """Expert configuration, output, or tensor alignment is invalid."""


CommandRunner = Callable[[list[str], dict[str, Any]], dict[str, Any]]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    weight: float
    temperature: float
    kind: str = "tensor"
    path: Path | None = None
    representation: str = "logits"
    command: tuple[str, ...] = ()
    prompt: str | None = None
    normalized_coordinates: bool = True
    boundary_sigma: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise EnsembleError("expert name is required")
        if self.weight < 0:
            raise EnsembleError(f"expert {self.name}: weight must be nonnegative")
        if self.temperature <= 0:
            raise EnsembleError(f"expert {self.name}: temperature must be positive")
        if self.kind not in {"tensor", "command"}:
            raise EnsembleError(f"expert {self.name}: unknown kind {self.kind}")
        if self.kind == "tensor":
            if self.path is None:
                raise EnsembleError(f"expert {self.name}: tensor path is required")
            if self.representation not in {"logits", "probabilities"}:
                raise EnsembleError(f"expert {self.name}: invalid representation")
        if self.kind == "command":
            if not self.command:
                raise EnsembleError(f"expert {self.name}: command is required")
            if self.boundary_sigma < 0:
                raise EnsembleError(f"expert {self.name}: boundary_sigma is negative")


@dataclass(frozen=True)
class FusionSpec:
    mode: str = "top_k"
    threshold: float = 0.5
    top_k_ratio: float = 0.02
    min_expert_support: int = 1
    support_threshold: float = 0.5
    dynamic_weights: str = "entropy"
    batch_size: int = 8
    max_batch_pixels: int = 32_000_000

    def validate(self) -> None:
        if self.mode not in {"threshold", "top_k"}:
            raise EnsembleError(f"unknown fusion mode: {self.mode}")
        if not 0 < self.threshold < 1:
            raise EnsembleError("threshold must be between zero and one")
        if self.top_k_ratio <= 0:
            raise EnsembleError("top_k_ratio must be positive")
        if self.min_expert_support < 1:
            raise EnsembleError("min_expert_support must be at least one")
        if not 0 < self.support_threshold < 1:
            raise EnsembleError("support_threshold must be between zero and one")
        if self.dynamic_weights not in {"static", "entropy"}:
            raise EnsembleError(f"unknown dynamic_weights mode: {self.dynamic_weights}")
        if self.batch_size < 1 or self.max_batch_pixels < 1:
            raise EnsembleError("batch size and pixel budget must be positive")


@dataclass(frozen=True)
class EnsembleConfig:
    input_video: Path
    experts: tuple[ExpertSpec, ...]
    fusion: FusionSpec


def load_config(path: Path) -> EnsembleConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EnsembleError(f"cannot load ensemble config {path}: {error}") from error
    required = {"input_video", "experts", "fusion"}
    missing = sorted(required - raw.keys())
    if missing:
        raise EnsembleError(f"missing ensemble fields: {', '.join(missing)}")
    raw_experts = raw["experts"]
    if not isinstance(raw_experts, list) or len(raw_experts) < 2:
        raise EnsembleError("at least two experts are required for triangulation")
    experts: list[ExpertSpec] = []
    for item in raw_experts:
        kind = item.get("kind", "tensor")
        expert = ExpertSpec(
            name=item["name"],
            weight=float(item.get("weight", 1.0)),
            temperature=float(item.get("temperature", 1.0)),
            kind=kind,
            path=Path(item["path"]).expanduser() if item.get("path") else None,
            representation=(
                "probabilities" if kind == "command" else item.get("representation", "logits")
            ),
            command=tuple(item.get("command", [])),
            prompt=item.get("prompt"),
            normalized_coordinates=bool(item.get("normalized_coordinates", True)),
            boundary_sigma=float(item.get("boundary_sigma", 0.0)),
        )
        expert.validate()
        experts.append(expert)
    if len({expert.name for expert in experts}) != len(experts):
        raise EnsembleError("expert names must be unique")
    if sum(expert.weight for expert in experts) <= 0:
        raise EnsembleError("total expert weight must be positive")
    raw_fusion = raw["fusion"]
    fusion = FusionSpec(
        mode=raw_fusion.get("mode", "top_k"),
        threshold=float(raw_fusion.get("threshold", 0.5)),
        top_k_ratio=float(raw_fusion.get("top_k_ratio", 0.02)),
        min_expert_support=int(raw_fusion.get("min_expert_support", 1)),
        support_threshold=float(raw_fusion.get("support_threshold", 0.5)),
        dynamic_weights=raw_fusion.get("dynamic_weights", "entropy"),
        batch_size=int(raw_fusion.get("batch_size", 8)),
        max_batch_pixels=int(raw_fusion.get("max_batch_pixels", 32_000_000)),
    )
    fusion.validate()
    if fusion.min_expert_support > len(experts):
        raise EnsembleError("min_expert_support exceeds expert count")
    return EnsembleConfig(
        input_video=Path(raw["input_video"]).expanduser(),
        experts=tuple(experts),
        fusion=fusion,
    )


class TensorExpertSource:
    def __init__(self, spec: ExpertSpec):
        spec.validate()
        if spec.path is None:
            raise EnsembleError(f"expert {spec.name}: tensor path is required")
        self.spec = spec
        self._values = np.load(spec.path, mmap_mode="r")
        if self._values.ndim != 3:
            raise EnsembleError(f"expert {spec.name}: tensor must have frames,height,width shape")

    @property
    def shape(self) -> tuple[int, int, int]:
        frames, height, width = self._values.shape
        return int(frames), int(height), int(width)

    def read(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray(self._values[list(indices)], dtype=np.float64)


class CommandExpertSource:
    def __init__(
        self,
        spec: ExpertSpec,
        frame_paths: Sequence[Path],
        shape: tuple[int, int, int],
        *,
        runner: CommandRunner | None = None,
    ):
        spec.validate()
        if len(frame_paths) != shape[0]:
            raise EnsembleError(f"expert {spec.name}: frame count does not match video")
        self.spec = spec
        self.frame_paths = tuple(frame_paths)
        self._shape = shape
        self._runner = runner or self._run_command

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @staticmethod
    def _run_command(command: list[str], request: dict[str, Any]) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise EnsembleError(
                f"expert command exited {completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise EnsembleError("expert command returned invalid JSON") from error

    def _rasterize(
        self,
        regions: Sequence[dict[str, Any]],
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        normalized = self.spec.normalized_coordinates
        coordinate_limit = 1.0 if normalized else None
        for region in regions:
            confidence = float(region.get("confidence", 1.0))
            if not 0 <= confidence <= 1:
                raise EnsembleError(f"expert {self.spec.name}: region confidence must be in [0,1]")
            value = int(round(confidence * 255))
            kind = region.get("type")
            coordinates = region.get("coordinates")
            if not isinstance(coordinates, list):
                raise EnsembleError(f"expert {self.spec.name}: region coordinates must be an array")
            if coordinate_limit is not None and any(
                not 0 <= float(coordinate) <= coordinate_limit for coordinate in coordinates
            ):
                raise EnsembleError(
                    f"expert {self.spec.name}: normalized coordinates must be in [0,1]"
                )
            if kind == "box":
                if len(coordinates) != 4:
                    raise EnsembleError(f"expert {self.spec.name}: box needs four coordinates")
                if normalized:
                    x1, y1, x2, y2 = (
                        float(coordinates[0]) * width,
                        float(coordinates[1]) * height,
                        float(coordinates[2]) * width,
                        float(coordinates[3]) * height,
                    )
                else:
                    x1, y1, x2, y2 = map(float, coordinates)
                if x2 <= x1 or y2 <= y1:
                    raise EnsembleError(f"expert {self.spec.name}: box has no area")
                if not normalized and (x1 < 0 or y1 < 0 or x2 > width or y2 > height):
                    raise EnsembleError(
                        f"expert {self.spec.name}: absolute box is outside the frame"
                    )
                draw.rectangle((x1, y1, x2, y2), fill=value)
            elif kind == "polygon":
                if len(coordinates) < 3:
                    raise EnsembleError(f"expert {self.spec.name}: polygon needs three points")
                if normalized:
                    points = [(float(x) * width, float(y) * height) for x, y in coordinates]
                else:
                    points = [(float(x), float(y)) for x, y in coordinates]
                if not normalized and any(
                    x < 0 or y < 0 or x > width or y > height for x, y in points
                ):
                    raise EnsembleError(
                        f"expert {self.spec.name}: absolute polygon is outside the frame"
                    )
                draw.polygon(points, fill=value)
            else:
                raise EnsembleError(f"expert {self.spec.name}: region type must be box or polygon")
        if self.spec.boundary_sigma:
            image = image.filter(ImageFilter.GaussianBlur(self.spec.boundary_sigma))
        return np.asarray(image, dtype=np.float64) / 255.0

    def read(self, indices: Sequence[int]) -> np.ndarray:
        request = {
            "prompt": self.spec.prompt,
            "normalized_coordinates": self.spec.normalized_coordinates,
            "frames": [
                {"frame_index": index, "path": str(self.frame_paths[index])} for index in indices
            ],
        }
        response = self._runner(list(self.spec.command), request)
        regions_by_frame = response.get("regions_by_frame")
        if not isinstance(regions_by_frame, list):
            raise EnsembleError(f"expert {self.spec.name}: regions_by_frame must be an array")
        if len(regions_by_frame) != len(indices):
            raise EnsembleError(f"expert {self.spec.name}: expected {len(indices)} region records")
        for record, expected_index in zip(regions_by_frame, indices, strict=True):
            if record.get("frame_index") != expected_index:
                raise EnsembleError(
                    f"expert {self.spec.name}: region records are not in request order"
                )
        _, height, width = self._shape
        grids = [
            self._rasterize(record.get("regions", []), width=width, height=height)
            for record in regions_by_frame
        ]
        return np.stack(grids)


def _effective_weights(
    probabilities: np.ndarray, base_weights: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "static":
        return np.broadcast_to(base_weights[:, None], probabilities.shape[:2])
    epsilon = 1e-7
    safe = np.clip(probabilities, epsilon, 1.0 - epsilon)
    entropy = -np.mean(
        safe * np.log2(safe) + (1.0 - safe) * np.log2(1.0 - safe),
        axis=(2, 3),
    )
    confidence = 1.0 - entropy
    dynamic = confidence * base_weights[:, None]
    totals = np.sum(dynamic, axis=0, keepdims=True)
    return dynamic / totals


def _selected_count(ratio: float, pixel_count: int) -> int:
    if ratio <= 1.0:
        return max(1, int(round(ratio * pixel_count)))
    return max(1, min(pixel_count, int(ratio)))


def fuse_batch(
    values: np.ndarray,
    specs: Sequence[ExpertSpec],
    fusion: FusionSpec,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fuse aligned expert values into masks and report effective weights."""
    if values.ndim != 4:
        raise EnsembleError("expert batch must have experts,frames,height,width shape")
    if values.shape[0] != len(specs):
        raise EnsembleError("expert count does not match tensor count")
    if not np.isfinite(values).all():
        raise EnsembleError("expert values contain NaN or infinity")
    for index, item in enumerate(specs):
        if item.representation == "probabilities" and (
            np.min(values[index]) < 0 or np.max(values[index]) > 1
        ):
            raise EnsembleError(f"expert {item.name}: probabilities must be in [0,1]")
    probabilities = np.stack(
        [
            _sigmoid(
                (
                    _logit(values[index])
                    if specs[index].representation == "probabilities"
                    else values[index]
                )
                / specs[index].temperature
            )
            for index in range(len(specs))
        ]
    )
    base_weights = np.asarray([spec.weight for spec in specs], dtype=np.float64)
    base_weights = base_weights / np.sum(base_weights)
    weights = _effective_weights(probabilities, base_weights, fusion.dynamic_weights)
    combined = np.sum(probabilities * weights[..., None, None], axis=0, dtype=np.float64)
    support = np.sum(probabilities > fusion.support_threshold, axis=0)
    eligible = support >= fusion.min_expert_support
    masks = np.zeros(combined.shape, dtype=bool)
    for frame_index, frame_probability in enumerate(combined):
        if fusion.mode == "threshold":
            masks[frame_index] = (frame_probability >= fusion.threshold) & eligible[frame_index]
            continue
        flat_probability = frame_probability.reshape(-1)
        flat_eligible = eligible[frame_index].reshape(-1)
        candidates = np.flatnonzero(flat_eligible)
        if candidates.size:
            selected_count = min(
                _selected_count(fusion.top_k_ratio, flat_probability.size),
                candidates.size,
            )
            order = np.argsort(-flat_probability[candidates], kind="stable")[:selected_count]
            chosen = candidates[order]
            masks[frame_index].reshape(-1)[chosen] = True
    reports = [
        {
            f"weight_{spec.name}": float(weights[expert_index, frame_index])
            for expert_index, spec in enumerate(specs)
        }
        for frame_index in range(values.shape[1])
    ]
    return masks, combined, reports


def dynamic_batches(
    frame_count: int,
    shape: tuple[int, int, int],
    fusion: FusionSpec,
) -> list[tuple[int, ...]]:
    _, height, width = shape
    pixel_batch_size = max(1, fusion.max_batch_pixels // (height * width))
    size = min(fusion.batch_size, pixel_batch_size)
    return [
        tuple(range(start, min(start + size, frame_count))) for start in range(0, frame_count, size)
    ]


def run_ensemble(
    config: EnsembleConfig,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Fuse expert tensors/commands into a mask video with bounded frame batches."""
    video_info = probe_video(config.input_video)
    tensor_specs = [spec for spec in config.experts if spec.kind == "tensor"]
    command_specs = [spec for spec in config.experts if spec.kind == "command"]
    sources: dict[str, TensorExpertSource | CommandExpertSource] = {}
    reference_shape: tuple[int, int, int] | None = None
    for spec in tensor_specs:
        source = TensorExpertSource(spec)
        if reference_shape is None:
            reference_shape = source.shape
        elif source.shape != reference_shape:
            raise EnsembleError("expert tensors have mismatching frames/height/width")
        sources[spec.name] = source
    if reference_shape is None:
        reference_shape = (video_info.frames, video_info.height, video_info.width)
    if reference_shape[0] != video_info.frames:
        raise EnsembleError("expert tensor frame count does not match input video")
    if reference_shape[1:] != (video_info.height, video_info.width):
        raise EnsembleError("expert tensor dimensions do not match input video")
    with TemporaryDirectory(prefix=".samurai-ensemble-") as temporary:
        frame_paths: list[Path] = []
        if command_specs:
            frame_paths = extract_color_frames(config.input_video, Path(temporary) / "frames")
        for spec in command_specs:
            sources[spec.name] = CommandExpertSource(spec, frame_paths, reference_shape)
        batches = dynamic_batches(reference_shape[0], reference_shape, config.fusion)
        frame_reports: list[dict[str, Any] | None] = [None] * reference_shape[0]
        with GrayVideoWriter(
            output,
            width=reference_shape[2],
            height=reference_shape[1],
            fps=video_info.fps,
        ) as writer:
            for indices in batches:
                values = np.stack([sources[spec.name].read(indices) for spec in config.experts])
                masks, probabilities, weights = fuse_batch(values, config.experts, config.fusion)
                for local_index, frame_index in enumerate(indices):
                    writer.write(masks[local_index].astype(np.uint8) * 255)
                    frame_reports[frame_index] = {
                        "selected_pixels": int(np.count_nonzero(masks[local_index])),
                        "mean_probability": float(np.mean(probabilities[local_index])),
                        **weights[local_index],
                    }
    if any(report is None for report in frame_reports):
        raise EnsembleError("ensemble did not emit every frame")
    result = {
        "input_video": str(config.input_video),
        "output": str(output),
        "frames": reference_shape[0],
        "width": reference_shape[2],
        "height": reference_shape[1],
        "fps": video_info.fps,
        "experts": [spec.name for spec in config.experts],
        "fusion": {
            "mode": config.fusion.mode,
            "dynamic_weights": config.fusion.dynamic_weights,
            "top_k_ratio": config.fusion.top_k_ratio,
            "threshold": config.fusion.threshold,
            "min_expert_support": config.fusion.min_expert_support,
            "support_threshold": config.fusion.support_threshold,
        },
        "batches": [len(batch) for batch in batches],
        "frames_report": frame_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
