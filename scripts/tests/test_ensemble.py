from __future__ import annotations

import json

import numpy as np

from samurai.ensemble import (
    CommandExpertSource,
    EnsembleError,
    ExpertSpec,
    FusionSpec,
    TensorExpertSource,
    dynamic_batches,
    fuse_batch,
    load_config,
    run_ensemble,
)
from samurai.media import write_gray_video


def spec(name: str, *, weight: float = 1.0, temperature: float = 1.0) -> ExpertSpec:
    return ExpertSpec(
        name=name,
        weight=weight,
        temperature=temperature,
        kind="tensor",
        representation="logits",
    )


def test_temperature_is_applied_before_sigmoid_and_weights_normalize():
    values = np.array(
        [
            [[[2.0, -2.0], [0.0, 0.0]]],
            [[[-2.0, 2.0], [0.0, 0.0]]],
        ]
    )
    specs = [spec("a", weight=3.0, temperature=2.0), spec("b", weight=1.0, temperature=1.0)]
    masks, probabilities, reports = fuse_batch(
        values,
        specs,
        FusionSpec(mode="top_k", top_k_ratio=0.25, dynamic_weights="static"),
    )
    assert masks.shape == probabilities.shape == (1, 2, 2)
    assert np.count_nonzero(masks) == 1
    assert reports[0]["weight_a"] == 0.75
    assert reports[0]["weight_b"] == 0.25


def test_entropy_reweights_uncertain_expert_and_support_filters_outliers():
    confident = np.full((1, 2, 2), 8.0)
    uncertain = np.full((1, 2, 2), 0.0)
    hallucination = np.array([[[8.0, 8.0], [-8.0, -8.0]]])
    values = np.stack([confident, uncertain, hallucination])
    specs = [spec("anchor"), spec("uncertain"), spec("semantic")]
    masks, _probabilities, reports = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            dynamic_weights="entropy",
            min_expert_support=2,
        ),
    )
    assert reports[0]["weight_anchor"] > reports[0]["weight_uncertain"]
    assert np.array_equal(masks[0], np.array([[True, True], [False, False]]))


def test_dynamic_batches_are_bounded_by_pixels():
    fusion = FusionSpec(batch_size=8, max_batch_pixels=8)
    batches = dynamic_batches(5, (5, 2, 2), fusion)
    assert batches == [(0, 1), (2, 3), (4,)]


def test_nonfinite_and_out_of_range_probabilities_fail():
    values = np.zeros((2, 1, 2, 2))
    values[0, 0, 0, 0] = np.nan
    try:
        fuse_batch(values, [spec("a"), spec("b")], FusionSpec())
    except EnsembleError:
        pass
    else:
        raise AssertionError("nonfinite expert value was accepted")

    probability_spec = ExpertSpec(name="a", weight=1, temperature=1, representation="probabilities")
    values[0, 0, 0, 0] = 1.5
    try:
        fuse_batch(values, [probability_spec, spec("b")], FusionSpec())
    except EnsembleError:
        pass
    else:
        raise AssertionError("out-of-range probability was accepted")


def test_tensor_expert_sources_must_align(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.zeros((2, 3, 4)))
    assert TensorExpertSource(
        ExpertSpec(name="first", weight=1, temperature=1, path=first)
    ).shape == (2, 3, 4)
    np.save(second, np.zeros((3, 4)))
    try:
        TensorExpertSource(
            ExpertSpec(name="second", weight=1, temperature=1, path=second, kind="tensor")
        )
    except EnsembleError:
        pass
    else:
        raise AssertionError("invalid source was accepted")


def test_run_ensemble_writes_video_and_report(tmp_path):
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    np.save(second, np.array([np.full((4, 6), -4.0), np.full((4, 6), 4.0)]))
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "first", "path": str(first), "weight": 2.0},
                    {
                        "name": "second",
                        "path": str(second),
                        "representation": "logits",
                        "weight": 1.0,
                    },
                ],
                "fusion": {
                    "mode": "top_k",
                    "top_k_ratio": 0.25,
                    "dynamic_weights": "entropy",
                    "batch_size": 1,
                },
            }
        )
    )
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["frames"] == 2
    assert result["batches"] == [1, 1]
    assert (tmp_path / "mask.mp4").is_file()
    saved_report = json.loads((tmp_path / "report.json").read_text())
    assert saved_report["experts"] == ["first", "second"]


def test_command_expert_rasterizes_normalized_regions(tmp_path):
    image = tmp_path / "frame.png"
    from PIL import Image

    Image.new("RGB", (10, 8), "black").save(image)

    def runner(command, request):
        assert command == ["fake-expert"]
        assert request["frames"][0]["frame_index"] == 0
        return {
            "regions_by_frame": [
                {
                    "frame_index": 0,
                    "regions": [
                        {
                            "type": "box",
                            "coordinates": [0.1, 0.25, 0.6, 0.75],
                            "confidence": 0.8,
                        }
                    ],
                }
            ]
        }

    source = CommandExpertSource(
        ExpertSpec(
            name="semantic",
            weight=1,
            temperature=1,
            kind="command",
            command=("fake-expert",),
            representation="probabilities",
        ),
        [image],
        (1, 8, 10),
        runner=runner,
    )
    probabilities = source.read([0])
    assert probabilities.shape == (1, 8, 10)
    assert probabilities.max() == np.float64(0.8)
    assert probabilities[0, 0, 0] == 0
