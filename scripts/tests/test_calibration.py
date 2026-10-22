from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from samurai.calibration import (
    CalibrationError,
    CalibrationSettings,
    calibrate_config,
    calibrate_experts,
)
from samurai.cli import main
from samurai.ensemble import ExpertSpec, expert_probabilities, load_config


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(name: str, *, temperature: float = 1.0, representation: str = "logits") -> ExpertSpec:
    return ExpertSpec(
        name=name,
        weight=1.0,
        temperature=temperature,
        kind="tensor",
        representation=representation,
    )


def panel_values() -> np.ndarray:
    """Three experts over a 6x8 frame: decisive, moderate, and near-uncertain."""
    latent = np.zeros((1, 6, 8), dtype=bool)
    latent[0, :4, :4] = True
    decisive = np.where(latent, 4.0, -4.0)
    moderate = np.where(latent, 2.0, -1.0)
    uncertain = np.where(latent, 0.4, -0.2)
    return np.stack([decisive, moderate, uncertain])


def test_bernoulli_em_recovers_latent_mask_and_ranks_expert_confidence():
    specs = [spec("decisive"), spec("moderate"), spec("uncertain")]
    result = calibrate_experts(
        panel_values(),
        specs,
        CalibrationSettings(tolerance=1e-7, max_iterations=100),
    )
    assert result.converged
    assert 1 <= result.iterations <= 100
    assert np.count_nonzero(result.latent_masks[0, :4, :4]) == 16
    assert np.count_nonzero(result.latent_masks[0, 4:, :]) == 0
    assert 0.2 < result.latent_positive_fraction < 0.5
    assert 0.2 < result.latent_selected_fraction < 0.5

    metrics = {item["name"]: item for item in result.expert_metrics}
    assert metrics["decisive"]["sensitivity"] > metrics["uncertain"]["sensitivity"]
    assert metrics["decisive"]["false_positive_rate"] < metrics["uncertain"]["false_positive_rate"]
    assert (
        result.confidence_weights["decisive"]
        > result.confidence_weights["moderate"]
        > result.confidence_weights["uncertain"]
    )
    assert sum(result.confidence_weights.values()) == pytest.approx(1.0)


def test_iteration_bound_is_respected_when_convergence_is_not_reached():
    specs = [spec("decisive"), spec("moderate"), spec("uncertain")]
    bounded = calibrate_experts(
        panel_values(),
        specs,
        CalibrationSettings(tolerance=1e-12, max_iterations=1),
    )
    assert bounded.iterations == 1
    assert not bounded.converged


def test_pairwise_summaries_report_dice_brier_and_disagreement():
    result = calibrate_experts(panel_values(), [spec("a"), spec("b"), spec("c")])
    assert len(result.pairwise_metrics) == 3
    for summary in result.pairwise_metrics:
        assert {"experts", "dice", "brier", "disagreement"} <= set(summary)
        assert 0.0 <= summary["dice"] <= 1.0
        assert 0.0 <= summary["disagreement"] <= 1.0
        assert summary["brier"] >= 0.0
    with pytest.raises(CalibrationError, match="at least three experts"):
        calibrate_experts(
            np.stack([np.full((1, 4, 4), 3.0), np.full((1, 4, 4), 3.0)]),
            [spec("a"), spec("b")],
        )


def test_probability_inputs_use_logit_then_temperature_scaling():
    latent = np.zeros((1, 2, 2), dtype=bool)
    latent[0, 0, 0] = True
    votes = np.where(latent, 0.92, 0.04)
    values = np.stack([votes, votes, votes])
    specs = [
        spec("sharp", temperature=0.5, representation="probabilities"),
        spec("soft", temperature=4.0, representation="probabilities"),
        spec("neutral", temperature=2.0, representation="probabilities"),
    ]
    transformed = expert_probabilities(values, specs)
    assert transformed[0, 0, 0, 0] > transformed[1, 0, 0, 0]
    assert transformed[0, 0, 0, 1] < transformed[1, 0, 0, 1]

    result = calibrate_experts(values, specs)
    metrics = {item["name"]: item for item in result.expert_metrics}
    assert metrics["sharp"]["confidence_weight"] > metrics["soft"]["confidence_weight"]
    assert result.confidence_weights["sharp"] > result.confidence_weights["soft"]


def test_identical_and_uninformative_panels_fall_back_to_equal_weights():
    identical = calibrate_experts(
        np.stack([np.full((1, 4, 4), 3.0)] * 3),
        [spec("a"), spec("b"), spec("c")],
    )
    for weight in identical.confidence_weights.values():
        assert weight == pytest.approx(1.0 / 3)

    uninformative = calibrate_experts(
        np.zeros((3, 1, 4, 4)),
        [spec("a"), spec("b"), spec("c")],
    )
    for weight in uninformative.confidence_weights.values():
        assert weight == pytest.approx(1.0 / 3)
    assert sum(uninformative.confidence_weights.values()) == pytest.approx(1.0)


def test_reported_weights_match_the_reported_agreement():
    result = calibrate_experts(panel_values(), [spec("a"), spec("b"), spec("c")])
    metrics = {item["name"]: item for item in result.expert_metrics}
    excess = np.array([metrics[name]["soft_agreement"] - 0.5 for name in ("a", "b", "c")])
    expected = excess / excess.sum()
    for index, name in enumerate(("a", "b", "c")):
        assert metrics[name]["confidence_weight"] == pytest.approx(expected[index])
        assert result.confidence_weights[name] == pytest.approx(expected[index])


def test_positive_class_follows_panel_consensus_not_first_expert():
    latent = np.zeros((1, 4, 4), dtype=bool)
    latent[0, :2, :2] = True
    inverted = np.where(latent, -6.0, 6.0)
    decisive = np.where(latent, 6.0, -6.0)
    values = np.stack([inverted, decisive, decisive])
    result = calibrate_experts(values, [spec("inverted"), spec("correct-a"), spec("correct-b")])
    assert np.array_equal(result.latent_masks[0], latent[0])
    assert result.anchor_agreement == pytest.approx(0.0)
    assert result.confidence_weights["inverted"] == pytest.approx(0.0)
    assert result.confidence_weights["correct-a"] > 0
    assert result.confidence_weights["correct-b"] > 0


def test_report_keeps_detailed_expert_metrics_and_names():
    result = calibrate_experts(panel_values(), [spec("a"), spec("b"), spec("c")])
    report = result.report()
    assert report["experts"] == ["a", "b", "c"]
    assert [item["name"] for item in report["expert_metrics"]] == ["a", "b", "c"]
    assert {"sensitivity", "false_positive_rate", "soft_agreement"} <= set(
        report["expert_metrics"][0]
    )


def test_calibration_bounds_and_inputs_fail_closed():
    with pytest.raises(CalibrationError):
        calibrate_experts(
            np.zeros((2, 1, 2, 2)), [spec("a"), spec("b")], CalibrationSettings(tolerance=-1)
        )
    with pytest.raises(CalibrationError):
        calibrate_experts(
            np.zeros((2, 1, 2, 2)), [spec("a"), spec("b")], CalibrationSettings(max_iterations=0)
        )

    probability_spec = spec("a", representation="probabilities")
    values = np.zeros((2, 1, 2, 2))
    values[0, 0, 0, 0] = 1.5
    with pytest.raises(CalibrationError):
        calibrate_experts(values, [probability_spec, spec("b")])

    values = np.zeros((2, 1, 2, 2))
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(CalibrationError):
        calibrate_experts(values, [spec("a"), spec("b")])

    with pytest.raises(CalibrationError):
        calibrate_experts(np.zeros((1, 1, 2, 2)), [spec("a")])


def test_calibrate_cli_writes_preserving_config_and_json_report(tmp_path, capsys):
    values = panel_values()
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    third = tmp_path / "third.npy"
    np.save(first, values[0])
    np.save(second, values[1])
    np.save(third, values[2])
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"")
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(input_video),
                "experts": [
                    {"name": "first", "path": str(first), "prompt": "keep me"},
                    {"name": "second", "path": str(second), "temperature": 1.5},
                    {"name": "third", "path": str(third)},
                ],
                "fusion": {
                    "mode": "top_k",
                    "top_k_ratio": 0.25,
                    "min_expert_support": 1,
                    "batch_size": 2,
                },
            }
        )
    )
    generated_config = tmp_path / "generated.json"
    report_path = tmp_path / "calibration.json"
    latent_path = tmp_path / "latent.npy"
    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config),
                "--config-output",
                str(generated_config),
                "--report",
                str(report_path),
                "--latent-output",
                str(latent_path),
                "--max-iterations",
                "50",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["method"] == "dawid_skene_bernoulli_em"
    assert report["labels"] == "none"
    assert report["iterations"] <= 50
    assert set(report["confidence_weights"]) == {"first", "second", "third"}
    assert sum(report["confidence_weights"].values()) == pytest.approx(1.0)
    assert len(report["pairwise"]) == 3
    assert len(report["frames_report"]) == 1
    assert {
        "mean_latent_probability",
        "latent_selected_fraction",
        "mean_expert_support",
        "mean_pairwise_disagreement",
        "mean_pairwise_brier",
    } <= set(report["frames_report"][0])
    for name, path in (("first", first), ("second", second), ("third", third)):
        assert report["artifacts"]["expert_sha256"][name] == _sha256(path)
    assert report["non_claims"]

    generated = load_config(generated_config)
    original = load_config(config)
    assert generated.experts[0].prompt == "keep me"
    assert generated.experts[0].path == original.experts[0].path
    assert generated.experts[1].temperature == original.experts[1].temperature
    assert generated.fusion == original.fusion
    assert generated.experts[0].weight > generated.experts[2].weight
    assert sum(expert.weight for expert in generated.experts) == pytest.approx(1.0)
    assert generated.calibration["source_config_sha256"] == _sha256(config)
    assert generated.calibration["expert_sha256"]["first"] == _sha256(first)

    latent = np.load(latent_path)
    assert latent.shape == (1, 6, 8)
    assert float(np.count_nonzero(latent >= 0.5)) == 16


def test_calibrate_config_rejects_out_of_range_probability_tensors(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.full((1, 2, 2), 1.5))
    np.save(second, np.zeros((1, 2, 2)))
    third = tmp_path / "third.npy"
    np.save(third, np.zeros((1, 2, 2)))
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(tmp_path / "input.mp4"),
                "experts": [
                    {"name": "first", "path": str(first), "representation": "probabilities"},
                    {"name": "second", "path": str(second)},
                    {"name": "third", "path": str(third)},
                ],
                "fusion": {"mode": "top_k"},
            }
        )
    )
    with pytest.raises(CalibrationError):
        calibrate_config(config, tmp_path / "generated.json", tmp_path / "report.json")


def test_calibrate_config_rejects_mismatched_tensor_shapes(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.zeros((1, 2, 2)))
    np.save(second, np.zeros((2, 2, 2)))
    third = tmp_path / "third.npy"
    np.save(third, np.zeros((1, 2, 2)))
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(tmp_path / "input.mp4"),
                "experts": [
                    {"name": "first", "path": str(first)},
                    {"name": "second", "path": str(second)},
                    {"name": "third", "path": str(third)},
                ],
                "fusion": {"mode": "top_k"},
            }
        )
    )
    with pytest.raises(CalibrationError):
        calibrate_config(config, tmp_path / "generated.json", tmp_path / "report.json")


def test_calibrate_config_rejects_command_experts(tmp_path):
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(tmp_path / "input.mp4"),
                "experts": [
                    {"name": "first", "path": str(tmp_path / "first.npy")},
                    {"name": "second", "path": str(tmp_path / "second.npy")},
                    {"name": "adapter", "kind": "command", "command": ["fake"]},
                ],
                "fusion": {"mode": "top_k"},
            }
        )
    )
    with pytest.raises(CalibrationError):
        calibrate_config(config, tmp_path / "generated.json", tmp_path / "report.json")
