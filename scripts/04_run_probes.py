#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retina_audit.probes import run_probe_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run linear probe sanity checks.")
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
        "--output-dir",
        default="outputs/probes",
        help="Directory for probe tables and metadata.",
    )
    parser.add_argument(
        "--figure-dir",
        default="outputs/figures/probes",
        help="Directory for probe figures.",
    )
    parser.add_argument(
        "--table-dir",
        default="outputs/tables",
        help="Directory for combined probe summary tables.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction.")
    parser.add_argument("--random-state", type=int, default=17, help="Deterministic split seed.")
    parser.add_argument("--max-iter", type=int, default=2000, help="Logistic regression max_iter.")
    args = parser.parse_args()
    metadata = run_probe_suite(
        embeddings_path=args.embeddings,
        index_path=args.index,
        quality_path=args.quality,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        table_dir=args.table_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"\nProbe evaluation complete: {metadata['num_targets']} targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
