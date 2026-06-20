from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

QUALITY_FEATURES = (
    "sharpness_laplacian_var",
    "brightness_mean",
    "contrast_std",
    "black_border_fraction",
    "fundus_area_fraction",
    "fundus_radius_estimate",
)

BIN_COLUMNS = {
    "sharpness_laplacian_var": "quality_bin_sharpness",
    "brightness_mean": "quality_bin_brightness",
    "contrast_std": "quality_bin_contrast",
    "black_border_fraction": "quality_bin_border",
    "fundus_area_fraction": "quality_bin_fov",
}


@dataclass(frozen=True)
class QualityOutputs:
    features: Path
    summary: Path
    figure_dir: Path


def compute_quality_features(
    manifest_path: str | Path,
    output_path: str | Path = "outputs/quality/quality_features.parquet",
    figure_dir: str | Path = "outputs/figures/quality",
    summary_path: str | Path = "outputs/quality/quality_summary.json",
    *,
    limit: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    image_root: str | Path | None = None,
    make_figures: bool = True,
    samples_per_grid: int = 12,
) -> dict[str, Any]:
    """Compute Phase 2 pixel-derived bottleneck features from a Phase 1 manifest."""
    pd = _import_pandas()
    manifest = pd.read_parquet(manifest_path)
    manifest = _select_usable_rows(manifest)
    manifest = _resolve_manifest_image_paths(manifest, image_root=image_root)
    manifest = _select_shard(manifest, shard_id=shard_id, num_shards=num_shards)
    if limit is not None:
        manifest = manifest.head(limit)

    rows = [_feature_row(row) for row in manifest.to_dict(orient="records")]
    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No usable images found for quality feature extraction.")

    features = _add_quality_bins(features)
    output_path = Path(output_path)
    figure_dir = Path(figure_dir)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    features.to_parquet(output_path, index=False)
    summary = _quality_summary(features, output_path, figure_dir)
    summary["shard_id"] = shard_id
    summary["num_shards"] = num_shards
    summary["limit"] = limit
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if make_figures:
        _write_histograms(features, figure_dir)
        _write_sample_grids(features, figure_dir, samples_per_grid=samples_per_grid)

    return summary


def _feature_row(row: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(row["image_path"]))
    rgb = _load_rgb(image_path)
    height, width = rgb.shape[:2]
    gray = _to_gray(rgb)
    mask = _retinal_mask(rgb)
    masked = gray[mask]
    if masked.size == 0:
        masked = gray.reshape(-1)

    fundus_area_fraction = float(mask.mean())
    fundus_radius_estimate = math.sqrt(float(mask.sum()) / math.pi)

    return {
        "image_id": row["image_id"],
        "image_path": str(image_path),
        "sharpness_laplacian_var": _laplacian_variance(gray),
        "brightness_mean": float(masked.mean()),
        "contrast_std": float(masked.std()),
        "black_border_fraction": _black_border_fraction(rgb),
        "fundus_area_fraction": fundus_area_fraction,
        "fundus_radius_estimate": fundus_radius_estimate,
        "height": int(height),
        "width": int(width),
        "aspect_ratio": float(width / height) if height else 0.0,
        "laterality": row.get("laterality"),
    }


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required for quality feature extraction. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _retinal_mask(rgb: np.ndarray) -> np.ndarray:
    gray = _to_gray(rgb)
    # Fundus images often have black padding; this threshold keeps the retinal disk region.
    mask = gray > 10.0
    if mask.mean() < 0.01:
        mask = gray > np.percentile(gray, 5)
    return mask


def _laplacian_variance(gray: np.ndarray) -> float:
    try:
        import cv2
    except ModuleNotFoundError:
        center = -4 * gray[1:-1, 1:-1]
        laplacian = (
            center
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
    else:
        laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.var(laplacian))


def _black_border_fraction(rgb: np.ndarray) -> float:
    gray = _to_gray(rgb)
    return float((gray <= 10.0).mean())


def _add_quality_bins(features: Any) -> Any:
    for feature, bin_column in BIN_COLUMNS.items():
        low = features[feature].quantile(0.25)
        high = features[feature].quantile(0.75)
        features[bin_column] = np.select(
            [features[feature] <= low, features[feature] >= high],
            ["low", "high"],
            default="middle",
        )
    return features


def _quality_summary(features: Any, output_path: Path, figure_dir: Path) -> dict[str, Any]:
    feature_stats = {}
    kept_features = []
    dropped_features = []
    for feature in QUALITY_FEATURES:
        series = features[feature]
        spread = float(series.quantile(0.75) - series.quantile(0.25))
        stats = {
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "iqr": spread,
            "n_unique": int(series.nunique(dropna=True)),
        }
        feature_stats[feature] = stats
        if spread > 0 and stats["n_unique"] >= 3:
            kept_features.append(feature)
        else:
            dropped_features.append(feature)

    required_minimum = {"sharpness_laplacian_var", "brightness_mean", "contrast_std"}
    checks = {
        "at_least_three_features_with_spread": len(kept_features) >= 3,
        "minimum_bottleneck_set_has_spread": required_minimum.issubset(set(kept_features)),
    }
    return {
        "features_path": str(output_path),
        "figure_dir": str(figure_dir),
        "num_images": int(len(features)),
        "feature_stats": feature_stats,
        "kept_features": kept_features,
        "dropped_features": dropped_features,
        "go_no_go": {"decision": "GO" if all(checks.values()) else "NO_GO", "checks": checks},
    }


def _write_histograms(features: Any, figure_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    histogram_dir = figure_dir / "histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    for feature in QUALITY_FEATURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(features[feature].dropna(), bins=40, color="#3b82f6", alpha=0.85)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("image count")
        fig.tight_layout()
        fig.savefig(histogram_dir / f"{feature}.png", dpi=150)
        plt.close(fig)


def _write_sample_grids(features: Any, figure_dir: Path, samples_per_grid: int) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ModuleNotFoundError:
        return

    grid_dir = figure_dir / "sample_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for feature, bin_column in BIN_COLUMNS.items():
        for bin_value in ("low", "high"):
            sample = features[features[bin_column] == bin_value].head(samples_per_grid)
            if sample.empty:
                continue
            cols = min(4, len(sample))
            rows = math.ceil(len(sample) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
            axes_array = np.atleast_1d(axes).reshape(rows, cols)
            for ax in axes_array.ravel():
                ax.axis("off")
            for ax, record in zip(axes_array.ravel(), sample.to_dict(orient="records"), strict=False):
                with Image.open(record["image_path"]) as image:
                    ax.imshow(image.convert("RGB"))
                ax.set_title(str(record["image_id"]), fontsize=8)
                ax.axis("off")
            fig.suptitle(f"{feature}: {bin_value}", fontsize=10)
            fig.tight_layout()
            fig.savefig(grid_dir / f"{feature}_{bin_value}.png", dpi=150)
            plt.close(fig)


def _select_usable_rows(manifest: Any) -> Any:
    selected = manifest
    if "usable" in selected.columns:
        selected = selected[selected["usable"]]
    if "image_path" not in selected.columns and "relative_image_path" not in selected.columns:
        raise ValueError("Manifest must include image_path or relative_image_path.")
    if "relative_image_path" in selected.columns:
        rel_mask = selected["relative_image_path"].notna() & (selected["relative_image_path"] != "")
        if "image_path" in selected.columns:
            abs_mask = selected["image_path"].notna() & (selected["image_path"] != "")
            selected = selected[rel_mask | abs_mask]
        else:
            selected = selected[rel_mask]
    else:
        selected = selected[selected["image_path"].notna() & (selected["image_path"] != "")]
    return selected.reset_index(drop=True)


def _resolve_manifest_image_paths(manifest: Any, image_root: str | Path | None) -> Any:
    if "relative_image_path" not in manifest.columns or image_root is None:
        return manifest
    root = Path(image_root)
    manifest = manifest.copy()
    manifest["image_path"] = [
        str(root / str(record["relative_image_path"]))
        if record.get("relative_image_path")
        else str(record.get("image_path", ""))
        for record in manifest.to_dict(orient="records")
    ]
    return manifest


def _select_shard(manifest: Any, *, shard_id: int | None, num_shards: int | None) -> Any:
    if shard_id is None and num_shards is None:
        return manifest
    if shard_id is None or num_shards is None:
        raise ValueError("Provide both --shard-id and --num-shards, or neither.")
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards).")
    return manifest.iloc[shard_id::num_shards].reset_index(drop=True)


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for quality feature extraction. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return pd
