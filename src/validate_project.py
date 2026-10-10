"""Validate the assignment's required dataset and evaluation invariants."""

from __future__ import annotations

import json
from pathlib import Path


def validate(audit: dict, results: dict, output_path: Path) -> dict[str, bool]:
    checks = {
        "model_2_1_exact_rows": audit["model_2_1_rows"] == 104,
        "model_2_1_exact_predictors": len(audit["model_2_1_predictors"]) == 8,
        "model_2_2_exact_rows": audit["model_2_2_rows"] == 208,
        "model_2_2_exact_predictors": len(audit["model_2_2_predictors"]) == 8,
        "chronological_holdout_model_2_1": results["model_2_1"]["holdout_match_id_range"] == [84, 104],
        "chronological_holdout_model_2_2": results["model_2_2"]["holdout_match_id_range"] == [84, 104],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Project validation failed: {', '.join(failed)}")
    return checks

