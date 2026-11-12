from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_n3_names as prepare


def test_normalize_removes_cub_numeric_prefix_and_separators():
    assert prepare.normalize("105.Whip_poor_Will") == "whip poor will"
    assert prepare.normalize("A330-200") == "a330 200"


def test_matlab_text_handles_common_matlab_string_shapes():
    assert prepare.matlab_text("Honda Accord") == "Honda Accord"
    assert prepare.matlab_text(b"BMW M3") == "BMW M3"
    assert prepare.matlab_text(np.array(list("Audi A4"))) == "Audi A4"


def test_validate_rows_enforces_dataset_size_and_old_new_order():
    rows = [
        {
            "label": str(i),
            "class_name": f"class {i}",
            "name_norm": f"class {i}",
            "is_old": "1" if i < 100 else "0",
        }
        for i in range(200)
    ]
    prepare.validate_rows("cub", rows)
    rows[0]["is_old"] = "0"
    with pytest.raises(ValueError, match="Old classes"):
        prepare.validate_rows("cub", rows)
