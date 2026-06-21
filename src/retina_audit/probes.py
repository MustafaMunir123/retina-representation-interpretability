"""Linear probe sanity checks for disease and bottleneck signals."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROBE_METRICS = (
    "auc",
    "balanced_accuracy",
    "accuracy",
    "train_positive",
    "train_negative",
    "test_positive",
    "test_negative",
)

DISEASE_PROBE_TARGETS = ("any_dr", "referable_dr", "severe_dr")
QUALITY_BIN_TARGETS = {
    "quality_bin_sharpness": "sharpness_high_vs_low",
    "quality_bin_brightness": "brightness_high_vs_low",
    "quality_bin_contrast": "contrast_high_vs_low",
    "quality_bin_border": "black_border_high_vs_low",
    "quality_bin_fov": "field_of_view_high_vs_low",
}


@dataclass(frozen=True)
class ProbeArtifacts:
    metrics: Path
    coefficients: Path
    predictions: Path
    metadata: Path
    figure: Path


def run_probe_suite(
    *,
    embeddings_path: str | Path,
    index_path: str | Path,
    quality_path: str | Path | None = "outputs/quality/quality_features.parquet",
    output_prefix: str | None = None,
    output_dir: str | Path = "outputs/probes",
    figure_dir: str | Path = "outputs/figures/probes",
    table_dir: str | Path = "outputs/tables",
    test_size: float = 0.2,
    random_state: int = 17,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """Run disease and bottleneck linear probes for one embedding artifact."""
    pd = _import_pandas()
    start = time.time()
    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)
    prefix = output_prefix or _infer_output_prefix(embeddings_path)
    artifacts = _probe_artifact_paths(
        output_dir=Path(output_dir), figure_dir=Path(figure_dir), prefix=prefix
    )

    embeddings = np.load(embeddings_path)
    index = pd.read_parquet(index_path).reset_index(drop=True)
    if len(index) != embeddings.shape[0]:
        raise ValueError(f"Embedding/index row mismatch: {embeddings.shape[0]} vs {len(index)}")
    index = index.copy()
    index["embedding_row"] = np.arange(len(index))

    table = _merge_quality(index, quality_path)

    metrics_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped_targets: list[dict[str, str]] = []

    for target in _target_specs(table):
        target_table = _target_table(table, target)
        if target_table is None:
            skipped_targets.append({"target": target["name"], "reason": "missing_or_empty"})
            continue
        target_rows = target_table["embedding_row"].to_numpy(dtype=int)
        y = target_table["probe_label"].to_numpy(dtype=int)
        groups = target_table.get("patient_id", target_table["image_id"]).astype(str).to_numpy()
        if len(np.unique(y)) < 2:
            skipped_targets.append({"target": target["name"], "reason": "single_class"})
            continue

        x_target = embeddings[target_rows].astype(np.float32, copy=False)
        result = _fit_probe(
            x_target,
            y,
            groups=groups,
            test_size=test_size,
            random_state=random_state,
            max_iter=max_iter,
        )
        if result is None:
            skipped_targets.append({"target": target["name"], "reason": "split_failed"})
            continue

        metric_row = {
            "run": prefix,
            "target": target["name"],
            "target_family": target["family"],
            "positive_class": target["positive_class"],
            "negative_class": target["negative_class"],
            **result["metrics"],
        }
        metrics_rows.append(metric_row)
        coefficient_rows.extend(
            {
                "run": prefix,
                "target": target["name"],
                "feature_index": feature_index,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
                "intercept": result["intercept"],
            }
            for feature_index, coefficient in enumerate(result["coefficients"])
        )
        prediction_rows.extend(
            {
                "run": prefix,
                "target": target["name"],
                "image_id": image_id,
                "split": split,
                "label": int(label),
                "score": float(score),
            }
            for image_id, split, label, score in zip(
                target_table["image_id"],
                result["split_labels"],
                y,
                result["scores"],
                strict=True,
            )
        )

    if not metrics_rows:
        raise RuntimeError("No probe targets produced metrics.")

    artifacts.metrics.parent.mkdir(parents=True, exist_ok=True)
    artifacts.figure.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(artifacts.metrics, index=False)
    pd.DataFrame(coefficient_rows).to_csv(artifacts.coefficients, index=False)
    pd.DataFrame(prediction_rows).to_parquet(artifacts.predictions, index=False)
    _write_auc_figure(pd.DataFrame(metrics_rows), artifacts.figure)
    summary_artifacts = refresh_probe_summary(output_dir=output_dir, table_dir=table_dir)

    metadata = {
        "run": prefix,
        "embeddings": str(embeddings_path),
        "index": str(index_path),
        "quality": str(quality_path) if quality_path else None,
        "num_embedding_rows": int(embeddings.shape[0]),
        "num_joined_rows": int(len(table)),
        "embedding_dim": int(embeddings.shape[1]),
        "num_targets": len(metrics_rows),
        "skipped_targets": skipped_targets,
        "test_size": test_size,
        "random_state": random_state,
        "runtime_seconds": round(time.time() - start, 3),
        "artifacts": {
            "metrics": str(artifacts.metrics),
            "coefficients": str(artifacts.coefficients),
            "predictions": str(artifacts.predictions),
            "metadata": str(artifacts.metadata),
            "figure": str(artifacts.figure),
            **summary_artifacts,
        },
    }
    artifacts.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def refresh_probe_summary(
    *,
    output_dir: str | Path = "outputs/probes",
    table_dir: str | Path = "outputs/tables",
) -> dict[str, str]:
    """Write combined probe metric tables from all per-run metrics in output_dir."""
    pd = _import_pandas()
    output_dir = Path(output_dir)
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = table_dir / "probe_metrics.csv"
    comparison_path = table_dir / "probe_comparison.csv"
    metric_files = sorted(output_dir.glob("*_probe_metrics.csv"))
    if not metric_files:
        return {
            "combined_metrics": str(metrics_path),
            "comparison": str(comparison_path),
        }
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    comparison = metrics.pivot_table(
        index=["target_family", "target"], columns="run", values="auc"
    ).reset_index()
    comparison.columns.name = None
    if {
        "eyepacs_resized_dinov2_resized_10k",
        "eyepacs_cropped_dinov2_cropped_10k",
    }.issubset(comparison.columns):
        comparison["cropped_minus_resized_auc"] = (
            comparison["eyepacs_cropped_dinov2_cropped_10k"]
            - comparison["eyepacs_resized_dinov2_resized_10k"]
        )
    comparison.to_csv(comparison_path, index=False)
    return {
        "combined_metrics": str(metrics_path),
        "comparison": str(comparison_path),
    }


def _target_specs(table: Any) -> list[dict[str, str]]:
    targets = [
        {
            "name": target,
            "family": "disease",
            "source_column": target,
            "positive_class": "true",
            "negative_class": "false",
        }
        for target in DISEASE_PROBE_TARGETS
        if target in table.columns
    ]
    targets.extend(
        {
            "name": target_name,
            "family": "quality",
            "source_column": column,
            "positive_class": "high",
            "negative_class": "low",
        }
        for column, target_name in QUALITY_BIN_TARGETS.items()
        if column in table.columns
    )
    if "laterality" in table.columns:
        targets.append(
            {
                "name": "laterality_right_vs_left",
                "family": "anatomy",
                "source_column": "laterality",
                "positive_class": "right",
                "negative_class": "left",
            }
        )
    return targets


def _target_table(table: Any, target: dict[str, str]) -> Any | None:
    source_column = target["source_column"]
    if source_column not in table.columns:
        return None
    selected = table[["image_id", "embedding_row", "patient_id", source_column]].copy()
    selected = selected[selected[source_column].notna()]
    if target["positive_class"] in {"true", "false"}:
        selected["probe_label"] = selected[source_column].astype(bool).astype(int)
    else:
        positive = target["positive_class"]
        negative = target["negative_class"]
        selected = selected[selected[source_column].isin([positive, negative])]
        selected["probe_label"] = (selected[source_column] == positive).astype(int)
    if selected.empty:
        return None
    return selected.reset_index(drop=True)


def _fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
    max_iter: int,
) -> dict[str, Any] | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    split = _train_test_split(y, groups=groups, test_size=test_size, random_state=random_state)
    if split is None:
        return None
    train_idx, test_idx = split
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=max_iter, solver="liblinear"),
    )
    model.fit(x[train_idx], y[train_idx])
    scores_test = model.predict_proba(x[test_idx])[:, 1]
    pred_test = (scores_test >= 0.5).astype(int)
    scores = np.full(len(y), np.nan, dtype=np.float32)
    scores[test_idx] = scores_test
    split_labels = np.array(["train"] * len(y), dtype=object)
    split_labels[test_idx] = "test"
    classifier = model.named_steps["logisticregression"]
    return {
        "metrics": {
            "auc": float(roc_auc_score(y[test_idx], scores_test)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred_test)),
            "accuracy": float(accuracy_score(y[test_idx], pred_test)),
            "train_positive": int(y[train_idx].sum()),
            "train_negative": int((y[train_idx] == 0).sum()),
            "test_positive": int(y[test_idx].sum()),
            "test_negative": int((y[test_idx] == 0).sum()),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "split_strategy": "group_shuffle",
        },
        "coefficients": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "scores": scores.tolist(),
        "split_labels": split_labels.tolist(),
    }


def _train_test_split(
    y: np.ndarray, *, groups: np.ndarray, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray] | None:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        for seed_offset in range(50):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state + seed_offset,
            )
            train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
            if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
                return train_idx, test_idx

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    except ValueError:
        return None
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        return None
    return train_idx, test_idx


def _merge_quality(index: Any, quality_path: str | Path | None) -> Any:
    if quality_path is None:
        return index
    quality_file = Path(quality_path)
    if not quality_file.exists():
        return index
    pd = _import_pandas()
    quality = pd.read_parquet(quality_file)
    quality_columns = [
        column
        for column in [
            "image_id",
            "quality_bin_sharpness",
            "quality_bin_brightness",
            "quality_bin_contrast",
            "quality_bin_border",
            "quality_bin_fov",
        ]
        if column in quality.columns
    ]
    merged = index.merge(quality[quality_columns], on="image_id", how="left")
    return merged.reset_index(drop=True)


def _write_auc_figure(metrics: Any, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_data = metrics.sort_values(["target_family", "target"]).reset_index(drop=True)
    colors = plot_data["target_family"].map(
        {"disease": "#4c78a8", "quality": "#f58518", "anatomy": "#54a24b"}
    )
    fig_width = max(8, 0.55 * len(plot_data))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(plot_data["target"], plot_data["auc"], color=colors)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Test AUC")
    ax.set_title("Linear Probe AUC")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _probe_artifact_paths(*, output_dir: Path, figure_dir: Path, prefix: str) -> ProbeArtifacts:
    return ProbeArtifacts(
        metrics=output_dir / f"{prefix}_probe_metrics.csv",
        coefficients=output_dir / f"{prefix}_probe_coefficients.csv",
        predictions=output_dir / f"{prefix}_probe_predictions.parquet",
        metadata=output_dir / f"{prefix}_probe_meta.json",
        figure=figure_dir / f"{prefix}_probe_auc.png",
    )


def _infer_output_prefix(embeddings_path: Path) -> str:
    name = embeddings_path.stem
    if name.endswith("_embeddings"):
        return name[: -len("_embeddings")]
    return name


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for probe evaluation.") from exc
    return pd
