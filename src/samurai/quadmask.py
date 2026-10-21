"""Conversion of primary and affected masks into Netflix VOID quadmasks."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .media import MediaError, decode_gray_frames, probe_video, write_gray_video

REMOVE = 0
OVERLAP = 63
AFFECTED = 127
KEEP = 255


@dataclass(frozen=True)
class QuadmaskResult:
    sequence_dir: Path
    frames: int
    width: int
    height: int
    fps: float


def _combine(primary: np.ndarray, affected: np.ndarray, *, first_frame: bool) -> np.ndarray:
    primary_mask = primary < 128
    affected_mask = affected < 128
    result = np.full(primary.shape, KEEP, dtype=np.uint8)
    result[affected_mask] = AFFECTED
    result[primary_mask] = REMOVE
    result[primary_mask & affected_mask] = OVERLAP
    # Netflix's mask reasoner clears affected-only pixels on frame zero.
    if first_frame:
        result[result == AFFECTED] = KEEP
    return result


def build_sequence(
    *,
    input_video: Path,
    primary_mask: Path,
    affected_mask: Path,
    background_prompt: str,
    sequence_dir: Path,
) -> QuadmaskResult:
    """Create a complete Netflix VOID sequence directory from two mask videos."""
    background_prompt = background_prompt.strip()
    if not background_prompt:
        raise ValueError("background_prompt must describe the desired remaining scene")
    video_info = probe_video(input_video)
    primary_info = probe_video(primary_mask)
    affected_info = probe_video(affected_mask)
    expected = (video_info.width, video_info.height, video_info.frames)
    if (primary_info.width, primary_info.height, primary_info.frames) != expected:
        raise MediaError("primary mask dimensions/frames do not match input video")
    if (affected_info.width, affected_info.height, affected_info.frames) != expected:
        raise MediaError("affected mask dimensions/frames do not match input video")
    sequence_dir.mkdir(parents=True, exist_ok=True)
    destination_input = sequence_dir / "input_video.mp4"
    if input_video.resolve() != destination_input.resolve():
        shutil.copy2(input_video, destination_input)
    with (
        TemporaryDirectory(prefix=".samurai-primary-", dir=sequence_dir) as primary_temp,
        TemporaryDirectory(prefix=".samurai-affected-", dir=sequence_dir) as affected_temp,
    ):
        primary_frames = decode_gray_frames(
            primary_mask, Path(primary_temp), expected_frames=video_info.frames
        )
        affected_frames = decode_gray_frames(
            affected_mask, Path(affected_temp), expected_frames=video_info.frames
        )
        combined = [
            _combine(primary, affected, first_frame=index == 0)
            for index, (primary, affected) in enumerate(
                zip(primary_frames, affected_frames, strict=True)
            )
        ]
        write_gray_video(combined, sequence_dir / "quadmask_0.mp4", fps=video_info.fps)
    (sequence_dir / "prompt.json").write_text(
        json.dumps({"bg": background_prompt}, indent=2) + "\n", encoding="utf-8"
    )
    return QuadmaskResult(
        sequence_dir=sequence_dir,
        frames=video_info.frames,
        width=video_info.width,
        height=video_info.height,
        fps=video_info.fps,
    )
