"""Disease and bottleneck direction analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DISEASE_DIRECTIONS = ("any_dr", "referable_dr", "severe_dr")
BOTTLENECK_DIRECTIONS = (
    "sharpness",
    "brightness",
    "contrast",
    "black_border",
    "field_of_view",
    "laterality",
)

DIRECTION_TARGETS = (
    {
        "name": "any_dr",
        "family": "disease",
        "source_column": "any_dr",
        "positive_class": "true",
        "negative_class": "false",
    },
    {
        "name": "referable_dr",
        "family": "disease",
        "source_column": "referable_dr",
        "positive_class": "true",
        "negative_class": "false",
    },
    {
        "name": "severe_dr",
        "family": "disease",
        "source_column": "severe_dr",
        "positive_class": "true",
        "negative_class": "false",
    },
    {
        "name": "sharpness",
        "family": "quality",
        "source_column": "quality_bin_sharpness",
        "positive_class": "high",
        "negative_class": "low",
    },
    {
        "name": "brightness",
        "family": "quality",
        "source_column": "quality_bin_brightness",
        "positive_class": "high",
        "negative_class": "low",
    },
    {
        "name": "contrast",
        "family": "quality",
        "source_column": "quality_bin_contrast",
        "positive_class": "high",
        "negative_class": "low",
    },
    {
        "name": "black_border",
        "family": "quality",
        "source_column": "quality_bin_border",
        "positive_class": "high",
        "negative_class": "low",
    },
    {
        "name": "field_of_view",
        "family": "quality",
        "source_column": "quality_bin_fov",
        "positive_class": "high",
        "negative_class": "low",
    },
    {
        "name": "laterality",
        "family": "anatomy",
        "source_column": "laterality",
        "positive_class": "right",
        "negative_class": "left",
    },
)


@dataclass(frozen=True)
class DirectionArtifacts:
    directions: Path
    direction_summary: Path
    cosine_matrix: Path
    projection_ratios: Path
    stability: Path
    metadata: Path


def compute_direction_suite(
    *,
    embeddings_path: str | Path,
    index_path: str | Path,
    quality_path: str | Path | None = "outputs/quality/quality_features.parquet",
    output_prefix: str | None = None,
    direction_dir: str | Path = "outputs/directions",
    table_dir: str | Path = "outputs/tables",
    n_bootstrap: int = 50,
    min_class_count: int = 20,
    random_state: int = 17,
) -> dict[str, Any]:
    """Compute disease/bottleneck directions and overlap tables."""
    pd = _import_pandas()
    start = time.time()
    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)
    prefix = output_prefix or _infer_output_prefix(embeddings_path)
    artifacts = _direction_artifact_paths(
        direction_dir=Path(direction_dir), table_dir=Path(table_dir), prefix=prefix
    )

    embeddings = np.load(embeddings_path)
    index = pd.read_parquet(index_path).reset_index(drop=True)
    if len(index) != embeddings.shape[0]:
        raise ValueError(f"Embedding/index row mismatch: {embeddings.shape[0]} vs {len(index)}")
    index = index.copy()
    index["embedding_row"] = np.arange(len(index))
    table = _merge_quality(index, quality_path)

    directions: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    skipped_targets: list[dict[str, str]] = []

    for target in DIRECTION_TARGETS:
        target_table = _target_table(table, target)
        if target_table is None:
            skipped_targets.append({"target": target["name"], "reason": "missing_or_empty"})
            continue
        positives = target_table[target_table["direction_label"] == 1]
        negatives = target_table[target_table["direction_label"] == 0]
        if len(positives) < min_class_count or len(negatives) < min_class_count:
            skipped_targets.append({"target": target["name"], "reason": "too_few_examples"})
            continue
        direction, raw_norm = _mean_difference_direction(
            embeddings,
            positives["embedding_row"].to_numpy(dtype=int),
            negatives["embedding_row"].to_numpy(dtype=int),
        )
        if direction is None:
            skipped_targets.append({"target": target["name"], "reason": "zero_norm"})
            continue
        directions[target["name"]] = direction
        summary_rows.append(
            {
                "run": prefix,
                "direction": target["name"],
                "direction_family": target["family"],
                "positive_class": target["positive_class"],
                "negative_class": target["negative_class"],
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "raw_norm": raw_norm,
            }
        )

    if not directions:
        raise RuntimeError("No directions were computed.")

    names = list(directions)
    direction_array = np.stack([directions[name] for name in names]).astype(np.float32)

    artifacts.directions.parent.mkdir(parents=True, exist_ok=True)
    artifacts.cosine_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts.directions,
        names=np.array(names),
        directions=direction_array,
        run=np.array([prefix]),
    )

    summary = pd.DataFrame(summary_rows)
    cosine = _cosine_matrix(prefix, directions)
    projection = _projection_ratios(prefix, directions)
    stability = _direction_stability(
        prefix=prefix,
        table=table,
        embeddings=embeddings,
        directions=directions,
        n_bootstrap=n_bootstrap,
        min_class_count=min_class_count,
        random_state=random_state,
    )

    summary.to_csv(artifacts.direction_summary, index=False)
    cosine.to_csv(artifacts.cosine_matrix, index=False)
    projection.to_csv(artifacts.projection_ratios, index=False)
    stability.to_csv(artifacts.stability, index=False)
    combined_artifacts = refresh_direction_summary(table_dir=table_dir, direction_dir=direction_dir)

    metadata = {
        "run": prefix,
        "embeddings": str(embeddings_path),
        "index": str(index_path),
        "quality": str(quality_path) if quality_path else None,
        "num_embedding_rows": int(embeddings.shape[0]),
        "num_joined_rows": int(len(table)),
        "embedding_dim": int(embeddings.shape[1]),
        "num_directions": len(names),
        "directions": names,
        "skipped_targets": skipped_targets,
        "n_bootstrap": n_bootstrap,
        "min_class_count": min_class_count,
        "random_state": random_state,
        "runtime_seconds": round(time.time() - start, 3),
        "artifacts": {
            "directions": str(artifacts.directions),
            "direction_summary": str(artifacts.direction_summary),
            "cosine_matrix": str(artifacts.cosine_matrix),
            "projection_ratios": str(artifacts.projection_ratios),
            "direction_stability": str(artifacts.stability),
            "metadata": str(artifacts.metadata),
            **combined_artifacts,
        },
    }
    artifacts.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def refresh_direction_summary(
    *,
    table_dir: str | Path = "outputs/tables",
    direction_dir: str | Path = "outputs/directions",
) -> dict[str, str]:
    """Write combined direction tables from all per-run direction artifacts."""
    pd = _import_pandas()
    table_dir = Path(table_dir)
    direction_dir = Path(direction_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    combined_cosine = table_dir / "cosine_matrix.csv"
    combined_projection = table_dir / "projection_ratios.csv"
    combined_stability = table_dir / "direction_stability.csv"
    combined_summary = table_dir / "direction_summary.csv"

    _concat_csvs(direction_dir.glob("*_direction_summary.csv"), combined_summary, pd)
    _concat_csvs(direction_dir.glob("*_cosine_matrix.csv"), combined_cosine, pd)
    _concat_csvs(direction_dir.glob("*_projection_ratios.csv"), combined_projection, pd)
    _concat_csvs(direction_dir.glob("*_direction_stability.csv"), combined_stability, pd)

    return {
        "combined_direction_summary": str(combined_summary),
        "combined_cosine_matrix": str(combined_cosine),
        "combined_projection_ratios": str(combined_projection),
        "combined_direction_stability": str(combined_stability),
    }


def _concat_csvs(paths: Any, destination: Path, pd: Any) -> None:
    files = sorted(paths)
    if not files:
        return
    pd.concat([pd.read_csv(path) for path in files], ignore_index=True).to_csv(
        destination, index=False
    )


def _cosine_matrix(prefix: str, directions: dict[str, np.ndarray]) -> Any:
    pd = _import_pandas()
    rows = []
    for source_name, source_vector in directions.items():
        for target_name, target_vector in directions.items():
            cosine = float(np.dot(source_vector, target_vector))
            rows.append(
                {
                    "run": prefix,
                    "source_direction": source_name,
                    "target_direction": target_name,
                    "cosine": cosine,
                    "abs_cosine": abs(cosine),
                }
            )
    return pd.DataFrame(rows)


def _projection_ratios(prefix: str, directions: dict[str, np.ndarray]) -> Any:
    pd = _import_pandas()
    rows = []
    bottleneck_names = [name for name in BOTTLENECK_DIRECTIONS if name in directions]
    for disease_name in DISEASE_DIRECTIONS:
        if disease_name not in directions:
            continue
        disease = directions[disease_name]
        for bottleneck_name in bottleneck_names:
            bottleneck = directions[bottleneck_name]
            cosine = float(np.dot(disease, bottleneck))
            rows.append(
                {
                    "run": prefix,
                    "disease_direction": disease_name,
                    "bottleneck_direction": bottleneck_name,
                    "cosine": cosine,
                    "abs_cosine": abs(cosine),
                    "projection_ratio": float(cosine**2),
                }
            )
    return pd.DataFrame(rows)


def _direction_stability(
    *,
    prefix: str,
    table: Any,
    embeddings: np.ndarray,
    directions: dict[str, np.ndarray],
    n_bootstrap: int,
    min_class_count: int,
    random_state: int,
) -> Any:
    pd = _import_pandas()
    rows = []
    rng = np.random.default_rng(random_state)
    for target in DIRECTION_TARGETS:
        name = target["name"]
        if name not in directions:
            continue
        target_table = _target_table(table, target)
        if target_table is None:
            continue
        positive_rows = target_table[target_table["direction_label"] == 1][
            "embedding_row"
        ].to_numpy(dtype=int)
        negative_rows = target_table[target_table["direction_label"] == 0][
            "embedding_row"
        ].to_numpy(dtype=int)
        similarities = []
        for _ in range(n_bootstrap):
            positive_sample = _sample_half(positive_rows, rng, min_class_count)
            negative_sample = _sample_half(negative_rows, rng, min_class_count)
            if positive_sample is None or negative_sample is None:
                continue
            sampled_direction, _ = _mean_difference_direction(
                embeddings, positive_sample, negative_sample
            )
            if sampled_direction is None:
                continue
            similarities.append(float(np.dot(directions[name], sampled_direction)))
        if similarities:
            rows.append(
                {
                    "run": prefix,
                    "direction": name,
                    "direction_family": target["family"],
                    "bootstrap_count": len(similarities),
                    "mean_cosine_to_full": float(np.mean(similarities)),
                    "std_cosine_to_full": float(np.std(similarities)),
                    "min_cosine_to_full": float(np.min(similarities)),
                    "max_cosine_to_full": float(np.max(similarities)),
                }
            )
    return pd.DataFrame(rows)


def _sample_half(
    rows: np.ndarray, rng: np.random.Generator, min_class_count: int
) -> np.ndarray | None:
    if len(rows) < min_class_count:
        return None
    sample_size = max(min_class_count, len(rows) // 2)
    if sample_size > len(rows):
        return None
    return rng.choice(rows, size=sample_size, replace=False)


def _mean_difference_direction(
    embeddings: np.ndarray, positive_rows: np.ndarray, negative_rows: np.ndarray
) -> tuple[np.ndarray | None, float]:
    positive_mean = embeddings[positive_rows].mean(axis=0)
    negative_mean = embeddings[negative_rows].mean(axis=0)
    direction = positive_mean - negative_mean
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return None, norm
    return (direction / norm).astype(np.float32), norm


def _target_table(table: Any, target: dict[str, str]) -> Any | None:
    source_column = target["source_column"]
    if source_column not in table.columns:
        return None
    selected = table[["image_id", "embedding_row", source_column]].copy()
    selected = selected[selected[source_column].notna()]
    if target["positive_class"] in {"true", "false"}:
        selected["direction_label"] = selected[source_column].astype(bool).astype(int)
    else:
        positive = target["positive_class"]
        negative = target["negative_class"]
        selected = selected[selected[source_column].isin([positive, negative])]
        selected["direction_label"] = (selected[source_column] == positive).astype(int)
    if selected.empty:
        return None
    return selected.reset_index(drop=True)


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
    return index.merge(quality[quality_columns], on="image_id", how="left").reset_index(drop=True)


def _direction_artifact_paths(
    *, direction_dir: Path, table_dir: Path, prefix: str
) -> DirectionArtifacts:
    return DirectionArtifacts(
        directions=direction_dir / f"{prefix}_directions.npz",
        direction_summary=direction_dir / f"{prefix}_direction_summary.csv",
        cosine_matrix=direction_dir / f"{prefix}_cosine_matrix.csv",
        projection_ratios=direction_dir / f"{prefix}_projection_ratios.csv",
        stability=direction_dir / f"{prefix}_direction_stability.csv",
        metadata=direction_dir / f"{prefix}_direction_meta.json",
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
        raise ModuleNotFoundError("pandas is required for direction analysis.") from exc
    return pd
