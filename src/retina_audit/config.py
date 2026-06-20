from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        config = _load_simple_yaml(text)
    else:
        config = yaml.safe_load(text) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def require_sections(config: dict[str, Any], sections: tuple[str, ...]) -> None:
    """Validate that required top-level sections exist."""
    missing = [section for section in sections if section not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config section(s): {joined}")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple two-level YAML configs used by the Phase 0 scaffold."""
    root: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" "):
            key, value = _split_yaml_pair(line)
            if value is None:
                root[key] = {}
                current_section = key
            else:
                root[key] = _coerce_scalar(value)
                current_section = None
            continue
        if current_section is None:
            raise ValueError(f"Nested value without a section: {raw_line}")
        key, value = _split_yaml_pair(line.strip())
        if value is None:
            raise ValueError("Install PyYAML for configs nested deeper than two levels.")
        section = root[current_section]
        if not isinstance(section, dict):
            raise ValueError(f"Cannot add nested key under scalar section: {current_section}")
        section[key] = _coerce_scalar(value)
    return root


def _split_yaml_pair(line: str) -> tuple[str, str | None]:
    if ":" not in line:
        raise ValueError(f"Invalid YAML line: {line}")
    key, value = line.split(":", 1)
    stripped = value.strip()
    return key.strip(), stripped if stripped else None


def _coerce_scalar(value: str) -> str | int | float | bool:
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")
