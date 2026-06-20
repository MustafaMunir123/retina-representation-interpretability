from __future__ import annotations

from typing import Any

import numpy as np

PREPROCESS_VARIANTS = ("resized", "cropped", "cropped_illumination_normalized")


def apply_preprocess(image: Any, variant: str) -> Any:
    """Apply lightweight fundus preprocessing before model-specific transforms."""
    variant = variant.lower()
    if variant == "resized":
        return image
    if variant == "cropped":
        return crop_fundus(image)
    if variant == "cropped_illumination_normalized":
        return normalize_illumination(crop_fundus(image))
    raise ValueError(f"Unsupported preprocessing variant: {variant}")


def crop_fundus(image: Any, threshold: int = 10, pad_fraction: float = 0.02) -> Any:
    """Crop around non-black fundus pixels while preserving the original image object type."""
    rgb = np.asarray(image.convert("RGB"))
    gray = _to_gray(rgb)
    mask = gray > threshold
    if not mask.any():
        return image

    ys, xs = np.where(mask)
    height, width = gray.shape
    pad = int(max(height, width) * pad_fraction)
    left = max(0, int(xs.min()) - pad)
    upper = max(0, int(ys.min()) - pad)
    right = min(width, int(xs.max()) + pad + 1)
    lower = min(height, int(ys.max()) + pad + 1)
    return image.crop((left, upper, right, lower))


def normalize_illumination(image: Any) -> Any:
    """Apply simple per-channel contrast stretching for optional smoke experiments."""
    from PIL import Image

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    low = np.percentile(rgb, 1, axis=(0, 1), keepdims=True)
    high = np.percentile(rgb, 99, axis=(0, 1), keepdims=True)
    scaled = (rgb - low) / np.maximum(high - low, 1.0)
    scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="RGB")


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
