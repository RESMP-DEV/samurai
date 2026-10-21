"""Lossless gray-video helpers shared by segmentation and VOID preparation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


class MediaError(RuntimeError):
    """A video probe, decode, or encode operation failed."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)


def probe_video(path: Path) -> VideoInfo:
    """Probe a video using ffprobe and return frame dimensions/count and FPS."""
    if not path.is_file():
        raise MediaError(f"video does not exist: {path}")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        stream = json.loads(result.stdout)["streams"][0]
        numerator, denominator = stream["r_frame_rate"].split("/", 1)
        info = VideoInfo(
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=float(numerator) / float(denominator),
            frames=int(stream["nb_read_frames"]),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MediaError(f"could not parse ffprobe output for {path}") from error
    if info.width <= 0 or info.height <= 0 or info.frames <= 0 or info.fps <= 0:
        raise MediaError(f"invalid video metadata for {path}: {info}")
    return info


def decode_gray_frames(
    video: Path,
    destination: Path,
    *,
    expected_frames: int | None = None,
) -> list[np.ndarray]:
    """Decode a video to gray arrays through ffmpeg and Pillow."""
    destination.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "format=gray",
            str(destination / "frame-%06d.png"),
        ]
    )
    files = sorted(destination.glob("frame-*.png"))
    if expected_frames is not None and len(files) != expected_frames:
        raise MediaError(f"decoded {len(files)} frames from {video}, expected {expected_frames}")
    if not files:
        raise MediaError(f"no frames decoded from {video}")
    return [np.asarray(Image.open(path).convert("L")) for path in files]


def write_gray_video(frames: list[np.ndarray], output: Path, *, fps: float) -> None:
    """Write a lossless H.264 yuv444p gray video suitable for VOID masks."""
    if not frames:
        raise MediaError("cannot write an empty mask video")
    shape = frames[0].shape
    if len(shape) != 2:
        raise MediaError(f"mask frame must be two-dimensional, got shape {shape}")
    if any(frame.shape != shape for frame in frames):
        raise MediaError("mask frames have inconsistent dimensions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".samurai-video-", dir=output.parent) as temporary:
        frame_dir = Path(temporary)
        for index, frame in enumerate(frames):
            Image.fromarray(frame.astype(np.uint8), mode="L").save(
                frame_dir / f"frame-{index:06d}.png"
            )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame-%06d.png"),
                "-c:v",
                "libx264",
                "-qp",
                "0",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "yuv444p",
                "-y",
                str(output),
            ]
        )
