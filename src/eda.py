"""Descriptive and bivariate checks for the two modelling tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_datasets import MATCH_FEATURES, TEAM_FEATURES


def _font(size: int, bold: bool = False):
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def read(path: Path, features: list[str], target: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    y = np.array([float(row[target]) for row in rows], dtype=float)
    groups = np.array([int(row["match_id"]) for row in rows], dtype=int)
    return x, y, groups


def summarise(path: Path, features: list[str], target: str) -> dict:
    x, y, groups = read(path, features, target)
    correlations = []
    for index, feature in enumerate(features):
        correlation = float(np.corrcoef(x[:, index], y)[0, 1]) if x[:, index].std() > 0 else 0.0
        correlations.append({"feature": feature, "pearson_correlation_with_target": correlation})
    correlation_matrix = np.corrcoef(x, rowvar=False)
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append({"feature_1": features[i], "feature_2": features[j], "correlation": float(correlation_matrix[i, j])})
    return {
        "rows": int(len(y)),
        "target": target,
        "target_summary": {
            "mean": float(y.mean()),
            "standard_deviation": float(y.std(ddof=1)),
            "median": float(np.median(y)),
            "first_quartile": float(np.quantile(y, 0.25)),
            "third_quartile": float(np.quantile(y, 0.75)),
            "minimum": float(y.min()),
            "maximum": float(y.max()),
            "zero_proportion": float(np.mean(y == 0)),
            "development_mean_matches_1_to_83": float(y[groups <= 83].mean()),
            "holdout_mean_matches_84_to_104": float(y[groups >= 84].mean()),
        },
        "feature_target_correlations": sorted(correlations, key=lambda row: abs(row["pearson_correlation_with_target"]), reverse=True),
        "strongest_feature_pairs": sorted(pairs, key=lambda row: abs(row["correlation"]), reverse=True)[:8],
        "correlation_matrix_condition_number": float(np.linalg.cond(correlation_matrix)),
    }


def draw_distribution(path: Path, title: str, y: np.ndarray) -> None:
    values = list(range(int(y.min()), int(y.max()) + 1))
    counts = [int(np.sum(y == value)) for value in values]
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), title, fill="#17233c", font=_font(42, True))
    draw.text((70, 98), "Observed outcome distribution across the full tournament", fill="#4b5563", font=_font(23))
    left, top, right, bottom = 110, 170, 1120, 650
    maximum = max(counts) * 1.12
    slot = (right - left) / max(len(values), 1)
    for value, count in zip(values, counts):
        x1 = left + (value - values[0]) * slot + slot * 0.15
        x2 = x1 + slot * 0.70
        y1 = bottom - count / maximum * (bottom - top)
        draw.rounded_rectangle((x1, y1, x2, bottom), radius=8, fill="#3b82a0")
        draw.text((x1 + (x2-x1)/2 - 10, y1 - 30), str(count), fill="#111827", font=_font(19, True))
        draw.text((x1 + (x2-x1)/2 - 8, bottom + 12), str(value), fill="#374151", font=_font(20))
    draw.line((left, bottom, right, bottom), fill="#6b7280", width=2)
    draw.text(((left+right)//2-80, 700), "Outcome value", fill="#111827", font=_font(22, True))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def run(root: Path) -> dict:
    model_21_path = root / "data/processed/model_2_1_matches.csv"
    model_22_path = root / "data/processed/model_2_2_team_matches.csv"
    s21 = summarise(model_21_path, MATCH_FEATURES, "goal_difference")
    s22 = summarise(model_22_path, TEAM_FEATURES, "team_goals")
    _, y21, _ = read(model_21_path, MATCH_FEATURES, "goal_difference")
    _, y22, _ = read(model_22_path, TEAM_FEATURES, "team_goals")
    draw_distribution(root / "outputs/figures/model_2_1_outcome_distribution.png", "Model 2.1 goal difference", y21)
    draw_distribution(root / "outputs/figures/model_2_2_outcome_distribution.png", "Model 2.2 team goals", y22)
    result = {"model_2_1": s21, "model_2_2": s22}
    (root / "outputs/results/eda_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).resolve().parents[1]), indent=2))

