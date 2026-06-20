#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retina_audit.config import load_config, require_sections
from retina_audit.data import audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit dataset labels, files, and image IDs.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Project root used to resolve relative paths.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    require_sections(config, ("dataset", "outputs"))
    summary = audit_dataset(config, project_root=args.project_root)
    decision = summary.get("go_no_go", {}).get("decision", "UNKNOWN")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nDataset audit decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
