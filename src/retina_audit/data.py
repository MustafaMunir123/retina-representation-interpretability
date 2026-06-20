from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_ID_COLUMN = "image_id"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


@dataclass(frozen=True)
class AuditPaths:
    manifest: Path
    summary: Path
    label_counts: Path
    missing_files: Path
    corrupt_images: Path
    filename_parse_report: Path


def audit_dataset(config: dict[str, Any], project_root: str | Path = ".") -> dict[str, Any]:
    """Build and validate a dataset manifest from a project config."""
    project_root = Path(project_root)
    dataset = _section(config, "dataset")
    outputs = _section(config, "outputs")
    paths = _audit_paths(project_root / str(outputs.get("manifest_dir", "outputs/manifests")))

    if dataset.get("label_source") == "folder":
        return _audit_folder_dataset(dataset, paths, project_root)
    return _audit_csv_label_dataset(dataset, paths, project_root)


def _audit_csv_label_dataset(
    dataset: dict[str, Any], paths: AuditPaths, project_root: Path
) -> dict[str, Any]:
    label_csv = _resolve_project_path(project_root, _required(dataset, "label_csv"))
    image_root = _resolve_project_path(project_root, _required(dataset, "image_root"))
    image_id_column = str(dataset.get("image_id_column", "image"))
    label_column = str(dataset.get("label_column", "level"))

    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    pd = _import_pandas()
    labels = pd.read_csv(label_csv)
    _require_columns(labels, (image_id_column, label_column), label_csv)
    labels[image_id_column] = labels[image_id_column].astype(str)

    image_paths = _collect_image_paths(image_root)
    image_lookup = _build_image_lookup(image_paths)
    manifest = labels.copy()
    manifest["image_id"] = manifest[image_id_column].map(_normalize_image_id)
    manifest["image_path"] = manifest["image_id"].map(
        lambda image_id: str(image_lookup.by_stem.get(image_id, ""))
    )
    manifest["label"] = pd.to_numeric(manifest[label_column], errors="coerce")
    manifest["split"] = str(dataset.get("split", "train"))

    if bool(dataset.get("patient_id_from_filename", False)):
        manifest["patient_id"] = manifest["image_id"].map(_parse_patient_id)
    else:
        manifest["patient_id"] = None
    if bool(dataset.get("laterality_from_filename", False)):
        manifest["laterality"] = manifest["image_id"].map(_parse_laterality)
    else:
        manifest["laterality"] = None

    manifest["any_dr"] = manifest["label"] >= 1
    manifest["referable_dr"] = manifest["label"] >= 2
    manifest["severe_dr"] = manifest["label"] >= 3
    manifest["is_readable"] = manifest["image_path"].map(lambda path: _is_readable_image(path))

    missing_files = manifest[manifest["image_path"] == ""].copy()
    corrupt_images = manifest[(manifest["image_path"] != "") & (~manifest["is_readable"])].copy()
    duplicate_ids = manifest[manifest["image_id"].duplicated(keep=False)].copy()
    valid_label_mask = manifest["label"].isin([0, 1, 2, 3, 4])
    usable_mask = (manifest["image_path"] != "") & manifest["is_readable"] & valid_label_mask
    manifest["usable"] = usable_mask

    label_counts = (
        manifest["label"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    parse_report = _filename_parse_report(manifest)
    summary = {
        "dataset": dataset.get("name"),
        "source": dataset.get("source"),
        "task_type": "dr_severity",
        "label_csv": str(label_csv),
        "image_root": str(image_root),
        "total_csv_rows": int(len(labels)),
        "total_image_files": int(len(image_paths)),
        "matched_image_label_rows": int((manifest["image_path"] != "").sum()),
        "missing_files": int(len(missing_files)),
        "duplicate_image_ids": int(duplicate_ids["image_id"].nunique()),
        "corrupt_images": int(len(corrupt_images)),
        "valid_label_rows": int(valid_label_mask.sum()),
        "usable_rows": int(usable_mask.sum()),
        "readable_fraction_of_labeled_rows": _safe_ratio(int(usable_mask.sum()), len(labels)),
        "label_values_are_0_to_4": bool(valid_label_mask.all()),
        "patient_id_parse_success": parse_report["patient_id_parse_success"],
        "laterality_parse_success": parse_report["laterality_parse_success"],
        "go_no_go": _csv_go_no_go(labels, usable_mask, valid_label_mask, parse_report),
    }

    _write_common_outputs(paths, manifest, summary, label_counts, missing_files, corrupt_images)
    _write_json(paths.filename_parse_report, parse_report)
    return summary


def _audit_folder_dataset(
    dataset: dict[str, Any], paths: AuditPaths, project_root: Path
) -> dict[str, Any]:
    image_root = _resolve_project_path(project_root, _required(dataset, "image_root"))
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    pd = _import_pandas()
    rows: list[dict[str, Any]] = []
    for image_path in _collect_image_paths(image_root):
        rel = image_path.relative_to(image_root)
        parts = rel.parts
        split = parts[0] if len(parts) >= 3 else str(dataset.get("split", "unknown"))
        class_name = parts[-2] if len(parts) >= 2 else "unknown"
        normalized_class = _normalize_class_name(class_name)
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "label": class_name,
                "class_name": class_name,
                "split": split,
                "patient_id": None,
                "laterality": _parse_laterality(image_path.stem)
                if bool(dataset.get("laterality_from_filename", False))
                else None,
                "any_dr": normalized_class == "diabetic_retinopathy",
                "referable_dr": None,
                "severe_dr": None,
                "is_normal_fundus": normalized_class == "normal_fundus",
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            columns=[
                "image_id",
                "image_path",
                "label",
                "class_name",
                "split",
                "patient_id",
                "laterality",
                "any_dr",
                "referable_dr",
                "severe_dr",
                "is_normal_fundus",
            ]
        )
    manifest["is_readable"] = manifest["image_path"].map(lambda path: _is_readable_image(path))
    manifest["usable"] = manifest["is_readable"]

    corrupt_images = manifest[~manifest["is_readable"]].copy()
    missing_files = manifest.iloc[0:0].copy()
    label_counts = (
        manifest["class_name"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    parse_report = _filename_parse_report(manifest)
    class_counts = dict(zip(label_counts["label"].astype(str), label_counts["count"], strict=False))
    summary = {
        "dataset": dataset.get("name"),
        "source": dataset.get("source"),
        "task_type": "disease_category",
        "image_root": str(image_root),
        "total_image_files": int(len(manifest)),
        "classes": sorted(class_counts),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "missing_files": 0,
        "corrupt_images": int(len(corrupt_images)),
        "usable_rows": int(manifest["usable"].sum()),
        "readable_fraction_of_files": _safe_ratio(int(manifest["usable"].sum()), len(manifest)),
        "patient_id_parse_success": parse_report["patient_id_parse_success"],
        "laterality_parse_success": parse_report["laterality_parse_success"],
        "go_no_go": _folder_go_no_go(manifest, class_counts),
    }

    _write_common_outputs(paths, manifest, summary, label_counts, missing_files, corrupt_images)
    _write_json(paths.filename_parse_report, parse_report)
    return summary


@dataclass(frozen=True)
class ImageLookup:
    by_stem: dict[str, Path]
    duplicate_stems: set[str]


def _build_image_lookup(paths: list[Path]) -> ImageLookup:
    by_stem: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for path in paths:
        stem = _normalize_image_id(path.stem)
        if stem in by_stem:
            duplicate_stems.add(stem)
            continue
        by_stem[stem] = path
    return ImageLookup(by_stem=by_stem, duplicate_stems=duplicate_stems)


def _collect_image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _filename_parse_report(manifest: Any) -> dict[str, Any]:
    total = len(manifest)
    patient_success = int(manifest["patient_id"].notna().sum()) if "patient_id" in manifest else 0
    laterality_success = int(manifest["laterality"].notna().sum()) if "laterality" in manifest else 0
    return {
        "rows": int(total),
        "patient_id_parse_success_count": patient_success,
        "patient_id_parse_success": _safe_ratio(patient_success, total),
        "laterality_parse_success_count": laterality_success,
        "laterality_parse_success": _safe_ratio(laterality_success, total),
    }


def _csv_go_no_go(
    labels: Any, usable_mask: Any, valid_label_mask: Any, parse_report: dict[str, Any]
) -> dict[str, Any]:
    usable_rows = int(usable_mask.sum())
    readable_fraction = _safe_ratio(usable_rows, len(labels))
    checks = {
        "at_least_95_percent_readable": readable_fraction >= 0.95,
        "labels_are_valid_0_to_4": bool(valid_label_mask.all()),
        "at_least_2000_usable_images": usable_rows >= 2000,
        "patient_id_parse_available": parse_report["patient_id_parse_success"] >= 0.95
        or parse_report["patient_id_parse_success_count"] == 0,
        "laterality_parse_available": parse_report["laterality_parse_success"] >= 0.95
        or parse_report["laterality_parse_success_count"] == 0,
    }
    return {"decision": "GO" if all(checks.values()) else "NO_GO", "checks": checks}


def _folder_go_no_go(manifest: Any, class_counts: dict[str, int]) -> dict[str, Any]:
    usable_rows = int(manifest["usable"].sum()) if len(manifest) else 0
    readable_fraction = _safe_ratio(usable_rows, len(manifest))
    checks = {
        "at_least_95_percent_readable": readable_fraction >= 0.95,
        "at_least_one_class": len(class_counts) > 0,
        "all_classes_nonzero": all(count > 0 for count in class_counts.values()),
        "at_least_2000_usable_images": usable_rows >= 2000,
    }
    return {"decision": "GO" if all(checks.values()) else "NO_GO", "checks": checks}


def _write_common_outputs(
    paths: AuditPaths,
    manifest: Any,
    summary: dict[str, Any],
    label_counts: Any,
    missing_files: Any,
    corrupt_images: Any,
) -> None:
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(paths.manifest, index=False)
    _write_json(paths.summary, summary)
    label_counts.to_csv(paths.label_counts, index=False)
    missing_files.to_csv(paths.missing_files, index=False)
    corrupt_images.to_csv(paths.corrupt_images, index=False)


def _audit_paths(manifest_dir: Path) -> AuditPaths:
    return AuditPaths(
        manifest=manifest_dir / "image_manifest.parquet",
        summary=manifest_dir / "dataset_summary.json",
        label_counts=manifest_dir / "label_counts.csv",
        missing_files=manifest_dir / "missing_files.csv",
        corrupt_images=manifest_dir / "corrupt_images.csv",
        filename_parse_report=manifest_dir / "filename_parse_report.json",
    )


def _is_readable_image(path: str | Path) -> bool:
    if not path:
        return False
    image_path = Path(path)
    if not image_path.exists():
        return False
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return True
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception:
        return False
    return True


def _parse_patient_id(image_id: str) -> str | None:
    match = re.match(r"^(.+?)_(left|right)$", str(image_id), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _parse_laterality(image_id: str) -> str | None:
    match = re.search(r"(?:^|_)(left|right)(?:$|_)", str(image_id), flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def _normalize_image_id(value: str) -> str:
    path = Path(str(value))
    return path.stem if path.suffix.lower() in IMAGE_EXTENSIONS else str(value)


def _normalize_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return section


def _required(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required config value: {key}")
    return str(value)


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _require_columns(frame: Any, columns: tuple[str, ...], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing column(s) in {source}: {joined}")


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for dataset audit. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return pd
