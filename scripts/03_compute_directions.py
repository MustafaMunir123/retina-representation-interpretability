#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retina_audit.directions import compute_direction_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute disease and bottleneck directions.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file.")
    parser.add_argument("--index", required=True, help="Path to embedding index parquet.")
    parser.add_argument(
        "--quality",
        default="outputs/quality/quality_features.parquet",
        help="Path to Phase 2 quality features parquet.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output prefix. Defaults to embedding artifact prefix.",
    )
    parser.add_argument(
        "--direction-dir",
        default="outputs/directions",
        help="Directory for per-run direction artifacts.",
    )
    parser.add_argument(
        "--table-dir",
        default="outputs/tables",
        help="Directory for combined direction tables.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Bootstrap samples for direction stability.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=20,
        help="Minimum positive/negative examples needed for a direction.",
    )
    parser.add_argument("--random-state", type=int, default=17, help="Deterministic seed.")
    args = parser.parse_args()
    metadata = compute_direction_suite(
        embeddings_path=args.embeddings,
        index_path=args.index,
        quality_path=args.quality,
        output_prefix=args.output_prefix,
        direction_dir=args.direction_dir,
        table_dir=args.table_dir,
        n_bootstrap=args.n_bootstrap,
        min_class_count=args.min_class_count,
        random_state=args.random_state,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"\nDirection analysis complete: {metadata['num_directions']} directions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
