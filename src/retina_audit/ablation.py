"""Bottleneck direction and subspace residualization."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retina_audit.directions import compute_direction_suite
from retina_audit.probes import run_probe_suite

ABLATION_TARGETS = ("single_direction", "bottleneck_subspace")
DEFAULT_REMOVE_DIRECTIONS = ("sharpness", "laterality")
DISEASE_TARGETS = ("any_dr", "referable_dr", "severe_dr")


@dataclass(frozen=True)
class AblationArtifacts:
    embeddings: Path
    index: Path
    metadata: Path


def run_residualization_experiment(
    *,
    embeddings_path: str | Path,
    index_path: str | Path,
    directions_path: str | Path,
    quality_path: str | Path | None = "outputs/quality/quality_features.parquet",
    remove_directions: list[str] | tuple[str, ...] = DEFAULT_REMOVE_DIRECTIONS,
    output_prefix: str | None = None,
    baseline_prefix: str | None = None,
    ablation_dir: str | Path = "outputs/ablation",
    probe_dir: str | Path = "outputs/probes",
    direction_dir: str | Path = "outputs/directions",
    figure_dir: str | Path = "outputs/figures/probes",
    table_dir: str | Path = "outputs/tables",
    evaluate: bool = True,
    n_bootstrap: int = 50,
    random_state: int = 17,
) -> dict[str, Any]:
    """Remove bottleneck directions, then optionally rerun probes/directions."""
    pd = _import_pandas()
    start = time.time()
    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)
    directions_path = Path(directions_path)
    baseline_prefix = baseline_prefix or _infer_output_prefix(embeddings_path)
    intervention = _intervention_name(remove_directions)
    output_prefix = output_prefix or f"{baseline_prefix}_resid_{intervention}"
    artifacts = _ablation_artifact_paths(Path(ablation_dir), output_prefix)

    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    index = pd.read_parquet(index_path)
    if len(index) != embeddings.shape[0]:
        raise ValueError(f"Embedding/index row mismatch: {embeddings.shape[0]} vs {len(index)}")

    direction_data = np.load(directions_path)
    direction_names = [str(name) for name in direction_data["names"]]
    direction_vectors = direction_data["directions"].astype(np.float32)
    direction_lookup = dict(zip(direction_names, direction_vectors, strict=True))
    missing = [name for name in remove_directions if name not in direction_lookup]
    if missing:
        raise ValueError(f"Directions not found in {directions_path}: {missing}")

    remove_matrix = np.stack([direction_lookup[name] for name in remove_directions])
    basis = _orthonormal_basis(remove_matrix)
    residualized = residualize_embeddings(embeddings, basis)

    artifacts.embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifacts.embeddings, residualized.astype(np.float32))
    index.to_parquet(artifacts.index, index=False)

    probe_metadata: dict[str, Any] | None = None
    direction_metadata: dict[str, Any] | None = None
    if evaluate:
        probe_metadata = run_probe_suite(
            embeddings_path=artifacts.embeddings,
            index_path=artifacts.index,
            quality_path=quality_path,
            output_prefix=output_prefix,
            output_dir=probe_dir,
            figure_dir=figure_dir,
            table_dir=table_dir,
            random_state=random_state,
        )
        direction_metadata = compute_direction_suite(
            embeddings_path=artifacts.embeddings,
            index_path=artifacts.index,
            quality_path=quality_path,
            output_prefix=output_prefix,
            direction_dir=direction_dir,
            table_dir=table_dir,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        write_improvement_summary(
            baseline_prefix=baseline_prefix,
            intervention_prefix=output_prefix,
            intervention=intervention,
            removed_directions=remove_directions,
            table_dir=table_dir,
        )

    metadata = {
        "baseline_run": baseline_prefix,
        "intervention_run": output_prefix,
        "intervention": intervention,
        "removed_directions": list(remove_directions),
        "embeddings": str(embeddings_path),
        "index": str(index_path),
        "directions": str(directions_path),
        "quality": str(quality_path) if quality_path else None,
        "num_images": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "subspace_rank": int(basis.shape[0]),
        "evaluate": evaluate,
        "runtime_seconds": round(time.time() - start, 3),
        "artifacts": {
            "embeddings": str(artifacts.embeddings),
            "index": str(artifacts.index),
            "metadata": str(artifacts.metadata),
            "probe_metadata": probe_metadata,
            "direction_metadata": direction_metadata,
            "improvement_summary": str(Path(table_dir) / "improvement_summary.csv"),
        },
    }
    artifacts.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def residualize_embeddings(embeddings: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the orthonormal row-space basis from each embedding row."""
    if basis.ndim != 2:
        raise ValueError("basis must be a 2D array")
    if basis.shape[1] != embeddings.shape[1]:
        raise ValueError(f"basis dim {basis.shape[1]} does not match {embeddings.shape[1]}")
    projection = embeddings @ basis.T
    return embeddings - projection @ basis


def write_improvement_summary(
    *,
    baseline_prefix: str,
    intervention_prefix: str,
    intervention: str,
    removed_directions: list[str] | tuple[str, ...],
    table_dir: str | Path = "outputs/tables",
) -> Path:
    """Compare baseline and intervention probe/projection outputs."""
    pd = _import_pandas()
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    baseline_probe = _read_run_probe_metrics(table_dir, baseline_prefix, pd)
    intervention_probe = _read_run_probe_metrics(table_dir, intervention_prefix, pd)
    baseline_projection = _read_run_projection_ratios(table_dir, baseline_prefix, pd)
    intervention_projection = _read_run_projection_ratios(table_dir, intervention_prefix, pd)
    stability = _read_run_stability(table_dir, intervention_prefix, pd)

    rows = []
    for target in DISEASE_TARGETS:
        baseline_auc = _metric_value(baseline_probe, target, "auc")
        intervention_auc = _metric_value(intervention_probe, target, "auc")
        baseline_projection_value = _max_projection(baseline_projection, target)
        intervention_projection_value = _max_projection(intervention_projection, target)
        stability_value = _stability_value(stability, target)
        rows.append(
            {
                "baseline_run": baseline_prefix,
                "intervention_run": intervention_prefix,
                "intervention": intervention,
                "removed_directions": ",".join(removed_directions),
                "target": target,
                "target_family": "disease",
                "baseline_auc": baseline_auc,
                "intervention_auc": intervention_auc,
                "auc_delta": _delta(intervention_auc, baseline_auc),
                "baseline_max_projection_ratio": baseline_projection_value,
                "intervention_max_projection_ratio": intervention_projection_value,
                "projection_ratio_delta": _delta(
                    intervention_projection_value, baseline_projection_value
                ),
                "intervention_direction_stability": stability_value,
                "notes": "negative projection delta means less disease-to-bottleneck overlap",
            }
        )

    baseline_bottleneck_auc = _mean_bottleneck_auc(baseline_probe, removed_directions)
    intervention_bottleneck_auc = _mean_bottleneck_auc(intervention_probe, removed_directions)
    rows.append(
        {
            "baseline_run": baseline_prefix,
            "intervention_run": intervention_prefix,
            "intervention": intervention,
            "removed_directions": ",".join(removed_directions),
            "target": "removed_bottleneck_mean",
            "target_family": "bottleneck",
            "baseline_auc": baseline_bottleneck_auc,
            "intervention_auc": intervention_bottleneck_auc,
            "auc_delta": _delta(intervention_bottleneck_auc, baseline_bottleneck_auc),
            "baseline_max_projection_ratio": np.nan,
            "intervention_max_projection_ratio": np.nan,
            "projection_ratio_delta": np.nan,
            "intervention_direction_stability": np.nan,
            "notes": "negative AUC delta means removed bottlenecks are less decodable",
        }
    )

    summary_path = table_dir / "improvement_summary.csv"
    new_rows = pd.DataFrame(rows)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        mask = ~(
            (existing["baseline_run"] == baseline_prefix)
            & (existing["intervention_run"] == intervention_prefix)
            & (existing["intervention"] == intervention)
        )
        new_rows = pd.concat([existing[mask], new_rows], ignore_index=True)
    new_rows.to_csv(summary_path, index=False)
    return summary_path


def _orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    _, singular_values, vt = np.linalg.svd(vectors, full_matrices=False)
    keep = singular_values > 1e-8
    if not np.any(keep):
        raise ValueError("Residualization subspace has zero rank")
    return vt[keep].astype(np.float32)


def _ablation_artifact_paths(ablation_dir: Path, output_prefix: str) -> AblationArtifacts:
    return AblationArtifacts(
        embeddings=ablation_dir / f"{output_prefix}_embeddings.npy",
        index=ablation_dir / f"{output_prefix}_index.parquet",
        metadata=ablation_dir / f"{output_prefix}_ablation_meta.json",
    )


def _intervention_name(remove_directions: list[str] | tuple[str, ...]) -> str:
    return "remove_" + "_".join(remove_directions)


def _infer_output_prefix(embeddings_path: Path) -> str:
    name = embeddings_path.stem
    if name.endswith("_embeddings"):
        return name[: -len("_embeddings")]
    return name


def _read_run_probe_metrics(table_dir: Path, run: str, pd: Any) -> Any:
    combined = table_dir / "probe_metrics.csv"
    if combined.exists():
        metrics = pd.read_csv(combined)
        selected = metrics[metrics["run"] == run]
        if not selected.empty:
            return selected
    per_run = Path("outputs/probes") / f"{run}_probe_metrics.csv"
    if per_run.exists():
        return pd.read_csv(per_run)
    raise FileNotFoundError(f"Probe metrics not found for run: {run}")


def _read_run_projection_ratios(table_dir: Path, run: str, pd: Any) -> Any:
    combined = table_dir / "projection_ratios.csv"
    if combined.exists():
        projection = pd.read_csv(combined)
        selected = projection[projection["run"] == run]
        if not selected.empty:
            return selected
    per_run = Path("outputs/directions") / f"{run}_projection_ratios.csv"
    if per_run.exists():
        return pd.read_csv(per_run)
    raise FileNotFoundError(f"Projection ratios not found for run: {run}")


def _read_run_stability(table_dir: Path, run: str, pd: Any) -> Any:
    combined = table_dir / "direction_stability.csv"
    if combined.exists():
        stability = pd.read_csv(combined)
        selected = stability[stability["run"] == run]
        if not selected.empty:
            return selected
    per_run = Path("outputs/directions") / f"{run}_direction_stability.csv"
    if per_run.exists():
        return pd.read_csv(per_run)
    raise FileNotFoundError(f"Direction stability not found for run: {run}")


def _metric_value(metrics: Any, target: str, column: str) -> float:
    row = metrics[metrics["target"] == target]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def _max_projection(projection: Any, disease_target: str) -> float:
    selected = projection[projection["disease_direction"] == disease_target]
    if selected.empty:
        return float("nan")
    return float(selected["projection_ratio"].max())


def _stability_value(stability: Any, target: str) -> float:
    row = stability[stability["direction"] == target]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["mean_cosine_to_full"])


def _mean_bottleneck_auc(metrics: Any, remove_directions: list[str] | tuple[str, ...]) -> float:
    target_names = {_bottleneck_probe_target(name) for name in remove_directions}
    selected = metrics[metrics["target"].isin(target_names)]
    if selected.empty:
        return float("nan")
    return float(selected["auc"].mean())


def _bottleneck_probe_target(name: str) -> str:
    mapping = {
        "sharpness": "sharpness_high_vs_low",
        "brightness": "brightness_high_vs_low",
        "contrast": "contrast_high_vs_low",
        "black_border": "black_border_high_vs_low",
        "field_of_view": "field_of_view_high_vs_low",
        "laterality": "laterality_right_vs_left",
    }
    return mapping.get(name, name)


def _delta(after: float, before: float) -> float:
    if np.isnan(after) or np.isnan(before):
        return float("nan")
    return float(after - before)


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for ablation experiments.") from exc
    return pd
