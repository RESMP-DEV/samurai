from __future__ import annotations

from argparse import ArgumentTypeError

import pytest

from samurai.cli import _box, _point, main


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
