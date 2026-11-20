"""Unit tests for scripts/summarize_results.py."""
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.summarize_results import (
    parse_result_filename,
    load_results,
    compute_summary,
    format_table,
    format_markdown,
    format_latex,
    main,
)


def test_parse_result_filename():
    assert parse_result_filename("cub_A_s0.json") == ("cub", "A", 0)
    assert parse_result_filename("scars_A+B_s3.json") == ("scars", "A+B", 3)
    assert parse_result_filename("aircraft_A+B_a0.25_s2.json") == ("aircraft", "A+B_a0.25", 2)
    assert parse_result_filename("cub_A+C_a0.25_s1.json") == ("cub", "A+C_a0.25", 1)
    assert parse_result_filename("scars_A+D_a0.75_s4.json") == ("scars", "A+D_a0.75", 4)
    assert parse_result_filename("invalid_name.txt") is None


def test_load_and_compute_summary(tmp_path: Path):
    # Create sample JSON result files
    sample_records = [
        {"dataset": "cub", "variant": "A", "seed": 0, "all": 0.60, "old": 0.70, "new": 0.50},
        {"dataset": "cub", "variant": "A", "seed": 1, "all": 0.64, "old": 0.72, "new": 0.56},
        {"dataset": "cub", "variant": "A+B", "seed": 0, "all": 0.80, "old": 0.85, "new": 0.75},
        {"dataset": "scars", "variant": "B", "seed": 0, "all": 0.40, "old": 0.45, "new": 0.35},
    ]

    for rec in sample_records:
        ds, var, s = rec["dataset"], rec["variant"], rec["seed"]
        data = {
            "all": rec["all"],
            "old": rec["old"],
            "new": rec["new"],
            "pred": [0, 1],
            "pred_scope": "all_rows",
            "n_empty_clusters": 0,
            "config": {
                "dataset": ds,
                "variant": var,
                "seed": s,
                "alpha": 0.5 if "A+B" in var else None,
            },
        }
        (tmp_path / f"{ds}_{var}_s{s}.json").write_text(json.dumps(data), encoding="utf-8")

    df = load_results(tmp_path)
    assert len(df) == 4

    summary = compute_summary(df, as_percent=True)
    assert len(summary) == 3

    # Check cub A values (2 seeds: 60% and 64%)
    cub_a = summary[(summary["dataset"] == "cub") & (summary["variant"] == "A")].iloc[0]
    assert cub_a["n_seeds"] == 2
    assert cub_a["all_mean"] == pytest.approx(62.0)
    assert cub_a["all_std"] == pytest.approx(np.std([60.0, 64.0], ddof=1))
    assert cub_a["all_str"] == f"62.00 ± {np.std([60.0, 64.0], ddof=1):.2f}"

    # Check 1 seed std is 0.0
    cub_ab = summary[(summary["dataset"] == "cub") & (summary["variant"] == "A+B")].iloc[0]
    assert cub_ab["n_seeds"] == 1
    assert cub_ab["all_std"] == 0.0

    # Check formatting
    txt = format_table(summary)
    assert "CUB" in txt
    assert "ACC All (%)" in txt

    md = format_markdown(summary)
    assert "| Dataset | Variant |" in md

    latex = format_latex(summary)
    assert r"\begin{tabular}" in latex


def test_cli_execution(tmp_path: Path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    data = {
        "all": 0.75,
        "old": 0.80,
        "new": 0.70,
        "pred": [0],
        "pred_scope": "all_rows",
        "n_empty_clusters": 0,
        "config": {"dataset": "aircraft", "variant": "A", "seed": 0},
    }
    (res_dir / "aircraft_A_s0.json").write_text(json.dumps(data), encoding="utf-8")

    out_csv = tmp_path / "summary.csv"
    out_md = tmp_path / "summary.md"
    detailed_csv = tmp_path / "detailed.csv"

    ret = main([
        "--results_dir", str(res_dir),
        "--out_csv", str(out_csv),
        "--out_md", str(out_md),
        "--detailed_csv", str(detailed_csv),
    ])
    assert ret == 0
    assert out_csv.exists()
    assert out_md.exists()
    assert detailed_csv.exists()
