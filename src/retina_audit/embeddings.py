from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retina_audit.models import FrozenEncoder, load_frozen_encoder
from retina_audit.preprocess import apply_preprocess

EMBEDDING_ARTIFACT_SUFFIXES = ("embeddings.npy", "index.parquet", "meta.json")
INDEX_COLUMNS = (
    "image_id",
    "image_path",
    "label",
    "any_dr",
    "referable_dr",
    "severe_dr",
    "patient_id",
    "laterality",
    "split",
)


@dataclass(frozen=True)
class EmbeddingArtifacts:
    embeddings: Path
    index: Path
    metadata: Path


def extract_embeddings_from_config(
    config: dict[str, Any],
    *,
    manifest_path: str | Path = "outputs/manifests/image_manifest.parquet",
    subset: str = "2000",
    device: str | None = None,
    output_prefix: str | None = None,
    allow_cpu: bool = False,
    image_root: str | Path | None = None,
) -> dict[str, Any]:
    """Extract frozen embeddings and write canonical Phase 3 artifacts."""
    pd = _import_pandas()
    torch = _import_torch()
    device = device or _default_device(torch)
    if device == "cpu" and not allow_cpu:
        raise RuntimeError(
            "Embedding extraction is GPU-oriented. Pass --allow-cpu only for tiny smoke tests."
        )

    dataset_config = _section(config, "dataset")
    preprocess_config = _section(config, "preprocess")
    model_config = _section(config, "model")
    outputs_config = _section(config, "outputs")

    manifest = pd.read_parquet(manifest_path)
    manifest = _select_usable_manifest_rows(manifest)
    manifest = _resolve_manifest_image_paths(
        manifest,
        image_root=image_root or dataset_config.get("image_root"),
    )
    manifest = _select_subset(manifest, subset)
    preprocess_variant = str(preprocess_config.get("variant", "resized"))
    batch_size = int(model_config.get("batch_size", 32))
    mixed_precision = bool(model_config.get("mixed_precision", True))

    encoder = load_frozen_encoder(model_config, device=device)
    artifacts = _artifact_paths(
        outputs_config,
        dataset_name=str(dataset_config.get("name", "dataset")),
        model_name=str(model_config.get("name", "model")),
        preprocess_variant=preprocess_variant,
        output_prefix=output_prefix,
    )

    start = time.time()
    embeddings: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []
    failed_images: list[dict[str, str]] = []

    for batch_records in _iter_batches(manifest.to_dict(orient="records"), batch_size):
        loaded_images = []
        loaded_records = []
        for record in batch_records:
            try:
                image = _load_image(record["image_path"])
                image = apply_preprocess(image, preprocess_variant)
            except Exception as exc:
                failed_images.append({"image_id": str(record["image_id"]), "error": str(exc)})
                continue
            loaded_images.append(image)
            loaded_records.append(record)
        if not loaded_images:
            continue
        batch_embeddings = _embed_batch(
            encoder,
            loaded_images,
            device=device,
            mixed_precision=mixed_precision,
            torch=torch,
        )
        embeddings.append(batch_embeddings)
        index_rows.extend(_index_row(record) for record in loaded_records)

    if not embeddings:
        raise RuntimeError("No embeddings were produced.")

    embedding_array = np.concatenate(embeddings, axis=0).astype(np.float32)
    index = pd.DataFrame(index_rows)
    artifacts.embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifacts.embeddings, embedding_array)
    index.to_parquet(artifacts.index, index=False)

    metadata = {
        "dataset": dataset_config.get("name"),
        "source": dataset_config.get("source"),
        "model": model_config.get("name"),
        "checkpoint": encoder.checkpoint,
        "backend": encoder.backend,
        "preprocess": preprocess_variant,
        "image_size": preprocess_config.get("image_size"),
        "embedding_dim": int(embedding_array.shape[1]),
        "num_images": int(embedding_array.shape[0]),
        "requested_subset": subset,
        "failed_images": failed_images,
        "runtime_seconds": round(time.time() - start, 3),
        "device": device,
        "mixed_precision": mixed_precision,
        "artifacts": {
            "embeddings": str(artifacts.embeddings),
            "index": str(artifacts.index),
            "metadata": str(artifacts.metadata),
        },
    }
    artifacts.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    validate_embedding_artifacts(artifacts)
    return metadata


def validate_embedding_artifacts(artifacts: EmbeddingArtifacts) -> None:
    """Validate row alignment and finite embedding values."""
    pd = _import_pandas()
    embedding_array = np.load(artifacts.embeddings)
    index = pd.read_parquet(artifacts.index)
    if len(index) != embedding_array.shape[0]:
        raise ValueError(
            f"Embedding/index row mismatch: {embedding_array.shape[0]} embeddings vs {len(index)} rows"
        )
    if not np.isfinite(embedding_array).all():
        raise ValueError("Embedding array contains non-finite values.")
    if "image_id" not in index.columns:
        raise ValueError("Embedding index must include image_id.")
    if index["image_id"].duplicated().any():
        raise ValueError("Embedding index contains duplicate image_id values.")


def _embed_batch(
    encoder: FrozenEncoder,
    images: list[Any],
    *,
    device: str,
    mixed_precision: bool,
    torch: Any,
) -> np.ndarray:
    with torch.no_grad():
        autocast_enabled = mixed_precision and device.startswith("cuda")
        with torch.autocast(device_type="cuda", enabled=autocast_enabled):
            if encoder.backend == "transformers":
                inputs = encoder.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = encoder.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0]
            elif encoder.backend == "timm":
                tensors = torch.stack([encoder.processor(image) for image in images]).to(device)
                embedding = encoder.model(tensors)
            else:
                raise ValueError(f"Unsupported encoder backend: {encoder.backend}")
    return embedding.detach().float().cpu().numpy()


def _artifact_paths(
    outputs_config: dict[str, Any],
    *,
    dataset_name: str,
    model_name: str,
    preprocess_variant: str,
    output_prefix: str | None,
) -> EmbeddingArtifacts:
    output_dir = Path(str(outputs_config.get("embedding_dir", "outputs/embeddings")))
    prefix = output_prefix or f"{dataset_name}_{model_name}_{preprocess_variant}"
    return EmbeddingArtifacts(
        embeddings=output_dir / f"{prefix}_embeddings.npy",
        index=output_dir / f"{prefix}_index.parquet",
        metadata=output_dir / f"{prefix}_meta.json",
    )


def _select_usable_manifest_rows(manifest: Any) -> Any:
    if "usable" in manifest.columns:
        manifest = manifest[manifest["usable"]]
    if "relative_image_path" in manifest.columns:
        has_path = manifest["relative_image_path"].notna() & (manifest["relative_image_path"] != "")
        manifest = manifest[has_path | (manifest["image_path"].notna() & (manifest["image_path"] != ""))]
    else:
        manifest = manifest[manifest["image_path"].notna() & (manifest["image_path"] != "")]
    return manifest.reset_index(drop=True)


def _resolve_manifest_image_paths(manifest: Any, image_root: str | Path | None) -> Any:
    if "relative_image_path" not in manifest.columns or image_root is None:
        return manifest
    root = Path(image_root)

    def resolve(record: Any) -> str:
        relative = record.get("relative_image_path")
        if relative:
            return str(root / str(relative))
        return str(record.get("image_path", ""))

    manifest = manifest.copy()
    manifest["image_path"] = [resolve(record) for record in manifest.to_dict(orient="records")]
    return manifest


def _select_subset(manifest: Any, subset: str) -> Any:
    if subset == "all":
        return manifest.reset_index(drop=True)
    try:
        n = int(subset)
    except ValueError as exc:
        raise ValueError("--subset must be an integer or 'all'.") from exc
    if n <= 0:
        raise ValueError("--subset must be positive.")
    return manifest.head(n).reset_index(drop=True)


def _iter_batches(records: list[dict[str, Any]], batch_size: int) -> Any:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _load_image(path: str | Path) -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required for embedding extraction.") from exc
    with Image.open(path) as image:
        return image.convert("RGB")


def _index_row(record: dict[str, Any]) -> dict[str, Any]:
    return {column: record.get(column) for column in INDEX_COLUMNS}


def _default_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return section


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for embedding extraction.") from exc
    return pd


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for embedding extraction.") from exc
    return torch
