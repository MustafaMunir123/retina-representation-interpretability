from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_FALLBACK_ORDER = ("retfound", "dinov2", "timm")


@dataclass
class FrozenEncoder:
    name: str
    checkpoint: str
    model: Any
    processor: Any
    backend: str
    embedding_dim: int | None = None


def load_frozen_encoder(model_config: dict[str, Any], device: str) -> FrozenEncoder:
    """Load a frozen image encoder from config."""
    name = str(model_config.get("name", "")).lower()
    if name == "dinov2":
        return _load_dinov2(model_config, device)
    if name == "timm":
        return _load_timm(model_config, device)
    if name == "retfound":
        raise NotImplementedError(
            "RETFound loading needs checkpoint-specific model glue. Use DINOv2/timm first, "
            "or add RETFound integration once the checkpoint is available."
        )
    raise ValueError(f"Unsupported model name: {name}")


def _load_dinov2(model_config: dict[str, Any], device: str) -> FrozenEncoder:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for DINOv2 embeddings. Install requirements first."
        ) from exc

    checkpoint = str(model_config.get("checkpoint", "facebook/dinov2-base"))
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    return FrozenEncoder(
        name="dinov2",
        checkpoint=checkpoint,
        model=model,
        processor=processor,
        backend="transformers",
        embedding_dim=hidden_size,
    )


def _load_timm(model_config: dict[str, Any], device: str) -> FrozenEncoder:
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("timm is required for timm embeddings.") from exc

    checkpoint = str(model_config.get("checkpoint", "resnet50"))
    model = timm.create_model(checkpoint, pretrained=True, num_classes=0)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data_config = resolve_model_data_config(model)
    processor = create_transform(**data_config, is_training=False)
    embedding_dim = getattr(model, "num_features", None)
    return FrozenEncoder(
        name="timm",
        checkpoint=checkpoint,
        model=model,
        processor=processor,
        backend="timm",
        embedding_dim=embedding_dim,
    )
