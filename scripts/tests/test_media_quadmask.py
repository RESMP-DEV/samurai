from __future__ import annotations

import json

import numpy as np

from samurai.media import probe_video, write_gray_video
from samurai.quadmask import AFFECTED, KEEP, OVERLAP, REMOVE, build_sequence


def make_video(path, frames: list[np.ndarray]) -> None:
    write_gray_video(frames, path, fps=12.0)


def test_gray_video_round_trip(tmp_path):
    frames = [np.full((5, 7), value, dtype=np.uint8) for value in (0, 127, 255)]
    video = tmp_path / "video.mp4"
    make_video(video, frames)
    info = probe_video(video)
    assert (info.width, info.height, info.frames) == (7, 5, 3)


def test_build_sequence_combines_quadmask(tmp_path):
    source = tmp_path / "input.mp4"
    primary = tmp_path / "primary.mp4"
    affected = tmp_path / "affected.mp4"
    make_video(source, [np.full((4, 4), 255, dtype=np.uint8) for _ in range(2)])
    make_video(primary, [np.array([[0, 255, 255, 255]] * 4, dtype=np.uint8) for _ in range(2)])
    make_video(
        affected,
        [np.array([[255, 127, 127, 127]] * 4, dtype=np.uint8) for _ in range(2)],
    )
    result = build_sequence(
        input_video=source,
        primary_mask=primary,
        affected_mask=affected,
        background_prompt="An empty table",
        sequence_dir=tmp_path / "sequence",
    )
    assert result.frames == 2
    assert (tmp_path / "sequence/input_video.mp4").is_file()
    assert (tmp_path / "sequence/quadmask_0.mp4").is_file()
    assert json.loads((tmp_path / "sequence/prompt.json").read_text()) == {"bg": "An empty table"}


def test_quadmask_values_follow_void_contract():
    from samurai.quadmask import _combine

    primary = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    affected = np.array([[255, 0, 0, 255]], dtype=np.uint8)
    combined = _combine(primary, affected, first_frame=False)
    assert combined.tolist() == [[REMOVE, OVERLAP, AFFECTED, KEEP]]
    assert _combine(primary, affected, first_frame=True).tolist()[0][2] == KEEP
