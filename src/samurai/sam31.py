"""SAM 3.1 request adapter and mask-video generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .media import MediaError, VideoInfo, probe_video, write_gray_video


@dataclass(frozen=True)
class SegmentationPrompt:
    """An absolute-coordinate SAM 3.1 prompt on one frame."""

    frame_index: int
    text: str | None = None
    points: tuple[tuple[float, float], ...] = ()
    point_labels: tuple[int, ...] = ()
    boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()

    def validate(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if not self.text and not self.points and not self.boxes_xyxy:
            raise ValueError("provide a text prompt, point prompt, or box prompt")
        if len(self.points) != len(self.point_labels):
            raise ValueError("every point needs one label")
        if self.points and any(label not in (0, 1) for label in self.point_labels):
            raise ValueError("point labels must be 0 or 1")
        if any(x < 0 or y < 0 for x, y in self.points):
            raise ValueError("point coordinates must be nonnegative")
        if any(x2 <= x1 or y2 <= y1 for x1, y1, x2, y2 in self.boxes_xyxy):
            raise ValueError("boxes must use x1,y1,x2,y2 with positive size")

    def request(self, video: VideoInfo) -> dict[str, Any]:
        self.validate()
        request: dict[str, Any] = {"type": "add_prompt", "frame_index": self.frame_index}
        if self.text:
            request["text"] = self.text
        if self.points:
            request["points"] = [[x / video.width, y / video.height] for x, y in self.points]
            request["point_labels"] = list(self.point_labels)
        if self.boxes_xyxy:
            request["bounding_boxes"] = [
                [
                    x1 / video.width,
                    y1 / video.height,
                    (x2 - x1) / video.width,
                    (y2 - y1) / video.height,
                ]
                for x1, y1, x2, y2 in self.boxes_xyxy
            ]
            request["bounding_box_labels"] = [True] * len(self.boxes_xyxy)
        return request


def _mask_from_outputs(outputs: dict[str, Any]) -> tuple[np.ndarray, int]:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        raise MediaError("SAM 3.1 response has no out_binary_masks")
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise MediaError(f"SAM 3.1 returned invalid mask shape {masks.shape}")
    return np.any(masks, axis=0).astype(np.uint8) * 255, int(masks.shape[0])


def _score_from_outputs(outputs: dict[str, Any]) -> np.ndarray:
    masks = np.asarray(outputs.get("out_binary_masks"), dtype=bool)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise MediaError(f"SAM 3.1 returned invalid mask shape {masks.shape}")
    probabilities = outputs.get("out_probs")
    if probabilities is None:
        object_probabilities = np.ones(masks.shape[0], dtype=np.float32)
    else:
        object_probabilities = np.asarray(probabilities, dtype=np.float32)
        if object_probabilities.shape != (masks.shape[0],):
            raise MediaError("SAM 3.1 out_probs must have one value per object")
    return np.max(masks.astype(np.float32) * object_probabilities[:, None, None], axis=0)


def segment_video(
    predictor: Any,
    video: Path,
    prompt: SegmentationPrompt,
    output: Path,
    scores_output: Path | None = None,
) -> dict[str, Any]:
    """Run one SAM 3.1 prompt through forward propagation and write a black mask video."""
    prompt.validate()
    if not video.exists():
        raise FileNotFoundError(video)
    info = probe_video(video)
    session = predictor.handle_request({"type": "start_session", "resource_path": str(video)})
    session_id = session["session_id"]
    frame_masks: dict[int, np.ndarray] = {}
    frame_scores: dict[int, np.ndarray] | None = {} if scores_output else None
    object_counts: set[int] = set()
    try:
        request = prompt.request(info)
        request["session_id"] = session_id
        predictor.handle_request(request)
        stream = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
        }
        for response in predictor.handle_stream_request(stream):
            mask, count = _mask_from_outputs(response["outputs"])
            frame_index = int(response["frame_index"])
            if frame_scores is not None:
                frame_scores[frame_index] = _score_from_outputs(response["outputs"])
            if frame_index in frame_masks:
                raise MediaError(f"duplicate SAM 3.1 output frame {frame_index}")
            frame_masks[frame_index] = mask
            object_counts.add(count)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    if set(frame_masks) != set(range(info.frames)):
        missing = sorted(set(range(info.frames)) - set(frame_masks))
        raise MediaError(f"SAM 3.1 did not return every frame; first missing={missing[:8]}")
    ordered = [frame_masks[index] for index in range(info.frames)]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_gray_video(ordered, output, fps=info.fps)
    if scores_output is not None and frame_scores is not None:
        scores = np.stack([frame_scores[index] for index in range(info.frames)])
        scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(scores_output, scores.astype(np.float32))
    return {
        "video": str(video),
        "output": str(output),
        "frames": info.frames,
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "object_counts": sorted(object_counts),
        "model_version": "sam3.1",
        "scores_output": str(scores_output) if scores_output else None,
        "scores_semantics": (
            "per-pixel object probability support; binary mask multiplied by SAM object probability"
            if scores_output
            else None
        ),
    }
