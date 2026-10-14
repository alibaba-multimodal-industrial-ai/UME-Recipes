import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "analyze_qwen2_last_token_attention.py"
SPEC = importlib.util.spec_from_file_location("attention_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stable_dataset_seed_is_stable_and_dataset_specific():
    assert MODULE.stable_dataset_seed(42, "DocVQA") == MODULE.stable_dataset_seed(42, "DocVQA")
    assert MODULE.stable_dataset_seed(42, "DocVQA") != MODULE.stable_dataset_seed(42, "ChartQA")


def test_analyze_rows_preserves_raw_mass_and_normalizes_previous_tokens():
    per_head = np.asarray(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.1, 0.3, 0.4],
            ]
        ],
        dtype=np.float32,
    )
    mean_heads, rows = MODULE.analyze_rows(
        per_head, ["special", "image", "text", "special"]
    )
    assert mean_heads.shape == (1, 4)
    assert np.isclose(rows[0]["self_mass"], 0.4)
    assert np.isclose(rows[0]["previous_mass"], 0.6)
    assert np.isclose(
        rows[0]["conditional_image_mass"]
        + rows[0]["conditional_text_mass"]
        + rows[0]["conditional_special_mass"],
        1.0,
    )


def test_analyze_rows_accepts_a_single_last_token():
    per_head = np.asarray([[[1.0], [1.0]]], dtype=np.float32)
    mean_heads, rows = MODULE.analyze_rows(per_head, ["special"])
    assert mean_heads.shape == (1, 1)
    assert rows[0]["self_mass"] == 1.0
    assert rows[0]["previous_mass"] == 0.0
    assert rows[0]["previous_entropy_normalized"] == 0.0
    for modality in ("image", "video", "text", "special"):
        assert rows[0][f"raw_{modality}_mass"] == 0.0
        assert rows[0][f"conditional_{modality}_mass"] == 0.0


def test_plot_attention_heatmap_accepts_no_previous_tokens(tmp_path):
    destination = tmp_path / "single-token.png"
    MODULE.plot_attention_heatmap(
        np.ones((4, 1), dtype=np.float32), destination, "single token"
    )
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_save_npz_via_local_stage_round_trips_and_cleans_stage(tmp_path, monkeypatch):
    stage = tmp_path / "local-stage"
    destination = tmp_path / "mounted-output" / "attention.npz"
    destination.parent.mkdir()
    monkeypatch.setenv("ATTN_LOCAL_STAGE_DIR", str(stage))
    expected = {
        "mean_heads_raw": np.arange(12, dtype=np.float32).reshape(3, 4),
        "per_head_raw": np.arange(24, dtype=np.float16).reshape(2, 3, 4),
    }
    MODULE.save_npz_via_local_stage(destination, expected)
    with np.load(destination) as actual:
        np.testing.assert_array_equal(actual["mean_heads_raw"], expected["mean_heads_raw"])
        np.testing.assert_array_equal(actual["per_head_raw"], expected["per_head_raw"])
    assert list(stage.iterdir()) == []
