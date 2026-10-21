from __future__ import annotations

import json
from argparse import ArgumentTypeError

import numpy as np
import pytest

from samurai.cli import _box, _point, main
from samurai.media import write_gray_video


def test_point_and_box_parsers():
    assert _point("12.5,4") == (12.5, 4.0)
    assert _box("1,2,3,5") == (1.0, 2.0, 3.0, 5.0)
    with pytest.raises(ArgumentTypeError):
        _box("3,2,1,5")


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0
    assert "samurai" in capsys.readouterr().out


def test_cli_doctor_reports_model_lanes_without_secret_values(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "aimlapi_key_configured" in report["checks"]
    assert "AIMLAPI_KEY" not in capsys.readouterr().out


def test_cli_ensemble_writes_mask_and_report(tmp_path, capsys):
    video = tmp_path / "input.mp4"
    write_gray_video(
        [np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)],
        video,
        fps=12,
    )
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    np.save(second, np.array([np.full((4, 6), -4.0), np.full((4, 6), 4.0)]))
    config = tmp_path / "ensemble.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "first", "path": str(first)},
                    {"name": "second", "path": str(second)},
                ],
                "fusion": {"mode": "top_k", "top_k_ratio": 0.25, "batch_size": 1},
            }
        )
    )
    output = tmp_path / "mask.mp4"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "ensemble",
                "--config",
                str(config),
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert json.loads(report.read_text())["frames"] == 2
    assert "frames" in json.loads(capsys.readouterr().out)
