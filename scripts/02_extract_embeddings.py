#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retina_audit.config import load_config, require_sections


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen image embeddings.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--subset", default="2000", help="Image subset size or 'all'.")
    args = parser.parse_args()

    config = load_config(args.config)
    require_sections(config, ("dataset", "preprocess", "model", "outputs"))
    print(f"Embedding extraction scaffold ready for subset={args.subset}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
