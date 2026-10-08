"""Chronological comparison of linear regression algorithms using NumPy only."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_datasets import MATCH_FEATURES, TEAM_FEATURES


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        scale[scale < 1e-12] = 1.0
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


def read_table(path: Path, features: list[str], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    y = np.array([float(row[target]) for row in rows], dtype=float)
    groups = np.array([int(row["match_id"]) for row in rows], dtype=int)
    return x, y, groups, rows


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(add_intercept(x)) @ y


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = add_intercept(x)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + alpha * penalty) @ design.T @ y


def soft_threshold(value: float, threshold: float) -> float:
    return math.copysign(max(abs(value) - threshold, 0.0), value)


def fit_elastic_net(x: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float, iterations: int = 5000) -> np.ndarray:
    intercept = float(y.mean())
    yc = y - intercept
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        previous = beta.copy()
        for j in range(x.shape[1]):
            residual = yc - x @ beta + x[:, j] * beta[j]
            rho = float(x[:, j] @ residual) / len(x)
            denom = float(x[:, j] @ x[:, j]) / len(x) + alpha * (1.0 - l1_ratio)
            beta[j] = soft_threshold(rho, alpha * l1_ratio) / max(denom, 1e-12)
        if np.max(np.abs(beta - previous)) < 1e-8:
            break
    return np.r_[intercept, beta]


def fit_huber(x: np.ndarray, y: np.ndarray, delta: float, iterations: int = 100) -> np.ndarray:
    design = add_intercept(x)
    beta = np.linalg.pinv(design) @ y
    for _ in range(iterations):
        residual = y - design @ beta
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(1.4826 * mad, 1e-6)
        u = np.abs(residual) / scale
        weights = np.ones_like(u)
        mask = u > delta
        weights[mask] = delta / u[mask]
        root_w = np.sqrt(weights)
        weighted_x = design * root_w[:, None]
        weighted_y = y * root_w
        updated = np.linalg.pinv(weighted_x) @ weighted_y
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    return beta


def predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ beta


def metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = y - predictions
    ss_res = float(errors @ errors)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "bias": float(np.mean(predictions - y)),
    }


def chronological_folds(groups: np.ndarray, minimum_train_fraction: float = 0.40, folds: int = 4):
    unique = np.unique(groups)
    minimum = int(math.ceil(len(unique) * minimum_train_fraction))
    remaining = len(unique) - minimum
    block = max(1, remaining // folds)
    splits = []
    for fold in range(folds):
        train_end = minimum + fold * block
        test_end = len(unique) if fold == folds - 1 else min(len(unique), train_end + block)
        train_groups = set(unique[:train_end])
        test_groups = set(unique[train_end:test_end])
        train_idx = np.array([i for i, value in enumerate(groups) if value in train_groups], dtype=int)
        test_idx = np.array([i for i, value in enumerate(groups) if value in test_groups], dtype=int)
        if len(test_idx):
            splits.append((train_idx, test_idx))
    return splits


def candidate_specs() -> list[dict]:
    specs = [{"name": "OLS", "family": "ols"}]
    specs += [{"name": f"Ridge a={a:g}", "family": "ridge", "alpha": a} for a in (0.1, 1.0, 10.0, 100.0)]
    specs += [{"name": f"Lasso a={a:g}", "family": "elastic", "alpha": a, "l1_ratio": 1.0} for a in (0.01, 0.05, 0.1, 0.25)]
    specs += [
        {"name": f"ElasticNet a={a:g} l1={ratio:g}", "family": "elastic", "alpha": a, "l1_ratio": ratio}
        for a in (0.01, 0.05, 0.1)
        for ratio in (0.25, 0.5, 0.75)
    ]
    specs += [{"name": f"Huber d={delta:g}", "family": "huber", "delta": delta} for delta in (1.0, 1.35, 2.0)]
    return specs


def fit_spec(spec: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if spec["family"] == "ols":
        return fit_ols(x, y)
    if spec["family"] == "ridge":
        return fit_ridge(x, y, spec["alpha"])
    if spec["family"] == "elastic":
        return fit_elastic_net(x, y, spec["alpha"], spec["l1_ratio"])
    if spec["family"] == "huber":
        return fit_huber(x, y, spec["delta"])
    raise ValueError(spec)


def select_model(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[dict, list[dict]]:
    folds = chronological_folds(groups)
    comparisons = []
    baseline_fold_metrics = []
    for train_idx, test_idx in folds:
        pred = np.full(len(test_idx), y[train_idx].mean())
        baseline_fold_metrics.append(metrics(y[test_idx], pred))
    comparisons.append(
        {
            "name": "Historical mean baseline",
            "family": "baseline",
            "cv_mae": float(np.mean([m["mae"] for m in baseline_fold_metrics])),
            "cv_rmse": float(np.mean([m["rmse"] for m in baseline_fold_metrics])),
            "cv_r2": float(np.mean([m["r2"] for m in baseline_fold_metrics])),
        }
    )
    for spec in candidate_specs():
        fold_metrics = []
        for train_idx, test_idx in folds:
            scaler = Standardizer.fit(x[train_idx])
            train_x = scaler.transform(x[train_idx])
            test_x = scaler.transform(x[test_idx])
            beta = fit_spec(spec, train_x, y[train_idx])
            fold_metrics.append(metrics(y[test_idx], predict(beta, test_x)))
        comparisons.append(
            {
                **spec,
                "cv_mae": float(np.mean([m["mae"] for m in fold_metrics])),
                "cv_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
                "cv_r2": float(np.mean([m["r2"] for m in fold_metrics])),
            }
        )
    eligible = [row for row in comparisons if row["family"] != "baseline"]
    best = min(eligible, key=lambda row: (row["cv_rmse"], row["cv_mae"]))
    return best, comparisons


def bootstrap_mae_interval(y: np.ndarray, pred: np.ndarray, seed: int = 140, repeats: int = 4000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for i in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        values[i] = np.mean(np.abs(y[indices] - pred[indices]))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def permutation_importance(beta: np.ndarray, scaler: Standardizer, x: np.ndarray, y: np.ndarray, names: list[str], seed: int = 140) -> list[dict]:
    rng = np.random.default_rng(seed)
    baseline = metrics(y, predict(beta, scaler.transform(x)))["rmse"]
    rows = []
    for j, name in enumerate(names):
        increases = []
        for _ in range(300):
            shuffled = x.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            rmse = metrics(y, predict(beta, scaler.transform(shuffled)))["rmse"]
            increases.append(rmse - baseline)
        rows.append({"feature": name, "rmse_increase": float(np.mean(increases))})
    return sorted(rows, key=lambda row: abs(row["rmse_increase"]), reverse=True)


def _font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_model_comparison(path: Path, title: str, comparisons: list[dict]) -> None:
    selected = sorted(comparisons, key=lambda row: row["cv_rmse"])[:6]
    width, height = 1500, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 45), title, fill="#17233c", font=_font(44, True))
    draw.text((80, 105), "Mean chronological cross-validation RMSE (lower is better)", fill="#4b5563", font=_font(25))
    left, top, right, bottom = 460, 180, 1400, 760
    maximum = max(row["cv_rmse"] for row in selected) * 1.12
    bar_height, gap = 62, 28
    for i, row in enumerate(selected):
        y = top + i * (bar_height + gap)
        draw.text((80, y + 13), row["name"], fill="#111827", font=_font(23))
        bar_width = int((right - left) * row["cv_rmse"] / maximum)
        color = "#f47b4f" if i == 0 else "#3b82a0"
        draw.rounded_rectangle((left, y, left + bar_width, y + bar_height), radius=12, fill=color)
        draw.text((left + bar_width + 14, y + 14), f"{row['cv_rmse']:.3f}", fill="#111827", font=_font(22, True))
    draw.line((left, bottom, right, bottom), fill="#9ca3af", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def draw_observed_predicted(path: Path, title: str, y: np.ndarray, pred: np.ndarray) -> None:
    width, height = 1100, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 40), title, fill="#17233c", font=_font(40, True))
    draw.text((65, 94), "Final chronological holdout", fill="#4b5563", font=_font(24))
    left, top, right, bottom = 120, 160, 1010, 790
    low = min(float(y.min()), float(pred.min())) - 0.5
    high = max(float(y.max()), float(pred.max())) + 0.5
    if high <= low:
        high = low + 1
    def sx(value): return left + (float(value) - low) / (high - low) * (right - left)
    def sy(value): return bottom - (float(value) - low) / (high - low) * (bottom - top)
    for tick in np.linspace(math.ceil(low), math.floor(high), max(2, int(math.floor(high)-math.ceil(low)+1))):
        xpix, ypix = sx(tick), sy(tick)
        draw.line((xpix, top, xpix, bottom), fill="#e5e7eb", width=1)
        draw.line((left, ypix, right, ypix), fill="#e5e7eb", width=1)
        draw.text((xpix - 10, bottom + 12), f"{tick:g}", fill="#4b5563", font=_font(20))
        draw.text((left - 45, ypix - 10), f"{tick:g}", fill="#4b5563", font=_font(20))
    draw.line((sx(low), sy(low), sx(high), sy(high)), fill="#f47b4f", width=4)
    for actual, fitted in zip(y, pred):
        xpix, ypix = sx(actual), sy(fitted)
        draw.ellipse((xpix-7, ypix-7, xpix+7, ypix+7), fill="#276c84", outline="white", width=2)
    draw.rectangle((left, top, right, bottom), outline="#6b7280", width=2)
    draw.text(((left+right)//2-75, 835), "Observed", fill="#111827", font=_font(24, True))
    draw.text((18, (top+bottom)//2), "Predicted", fill="#111827", font=_font(24, True))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def analyse(name: str, table: Path, features: list[str], target: str, output_dir: Path, clip_zero: bool = False) -> dict:
    x, y, groups, rows = read_table(table, features, target)
    unique_groups = np.unique(groups)
    split = int(math.floor(len(unique_groups) * 0.80))
    train_groups, test_groups = set(unique_groups[:split]), set(unique_groups[split:])
    train_idx = np.array([i for i, value in enumerate(groups) if value in train_groups], dtype=int)
    test_idx = np.array([i for i, value in enumerate(groups) if value in test_groups], dtype=int)

    best, comparisons = select_model(x[train_idx], y[train_idx], groups[train_idx])
    scaler = Standardizer.fit(x[train_idx])
    beta = fit_spec(best, scaler.transform(x[train_idx]), y[train_idx])
    raw_predictions = predict(beta, scaler.transform(x[test_idx]))
    predictions = np.maximum(raw_predictions, 0.0) if clip_zero else raw_predictions
    test_metrics = metrics(y[test_idx], predictions)
    baseline_predictions = np.full(len(test_idx), y[train_idx].mean())
    baseline_metrics = metrics(y[test_idx], baseline_predictions)
    residuals = y[test_idx] - predictions
    std = float(residuals.std(ddof=0))
    skew = float(np.mean(((residuals - residuals.mean()) / std) ** 3)) if std > 0 else 0.0
    kurtosis = float(np.mean(((residuals - residuals.mean()) / std) ** 4)) if std > 0 else 3.0
    jarque_bera = len(residuals) / 6.0 * (skew**2 + ((kurtosis - 3.0) ** 2) / 4.0)
    abs_residual_correlation = float(np.corrcoef(np.abs(residuals), predictions)[0, 1]) if len(residuals) > 2 else 0.0

    importance = permutation_importance(beta, scaler, x[test_idx], y[test_idx], features)
    coefficients = [{"feature": feature, "standardised_coefficient": float(value)} for feature, value in zip(features, beta[1:])]
    result = {
        "task": name,
        "row_count": len(rows),
        "predictor_count": len(features),
        "predictors": features,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_match_count": len(train_groups),
        "test_match_count": len(test_groups),
        "holdout_match_id_range": [int(min(test_groups)), int(max(test_groups))],
        "selected_model": best,
        "model_comparison": comparisons,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "mae_95_percent_bootstrap_interval": bootstrap_mae_interval(y[test_idx], predictions),
        "diagnostics": {
            "residual_mean": float(residuals.mean()),
            "residual_standard_deviation": std,
            "jarque_bera_statistic": float(jarque_bera),
            "absolute_residual_fitted_correlation": abs_residual_correlation,
            "negative_raw_predictions": int(np.sum(raw_predictions < 0)),
            "zero_clipping_applied": clip_zero,
        },
        "standardised_coefficients": coefficients,
        "permutation_importance": importance,
        "holdout_predictions": [
            {
                "match_id": int(groups[index]),
                "observed": float(y[index]),
                "predicted": float(predictions[position]),
                "residual": float(y[index] - predictions[position]),
            }
            for position, index in enumerate(test_idx)
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = name.lower().replace(" ", "_").replace(".", "_")
    (output_dir / "results" / f"{stem}.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "results" / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    draw_model_comparison(output_dir / "figures" / f"{stem}_comparison.png", f"{name} model comparison", comparisons)
    draw_observed_predicted(output_dir / "figures" / f"{stem}_observed_predicted.png", f"{name} observed and predicted", y[test_idx], predictions)
    return result


def run(root: Path) -> dict:
    output_dir = root / "outputs"
    model_21 = analyse(
        "Linear Regression 2.1",
        root / "data/processed/model_2_1_matches.csv",
        MATCH_FEATURES,
        "goal_difference",
        output_dir,
        clip_zero=False,
    )
    model_22 = analyse(
        "Linear Regression 2.2",
        root / "data/processed/model_2_2_team_matches.csv",
        TEAM_FEATURES,
        "team_goals",
        output_dir,
        clip_zero=True,
    )
    summary = {"model_2_1": model_21, "model_2_2": model_22}
    (output_dir / "results/combined_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run(root), indent=2))

