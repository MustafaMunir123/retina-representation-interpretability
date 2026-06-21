#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retina_audit.ablation import DEFAULT_REMOVE_DIRECTIONS, run_residualization_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate direction or subspace removal.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file.")
    parser.add_argument("--directions", required=True, help="Path to directions.npz.")
    parser.add_argument("--index", required=True, help="Path to embedding index parquet.")
    parser.add_argument(
        "--quality",
        default="outputs/quality/quality_features.parquet",
        help="Path to Phase 2 quality features parquet.",
    )
    parser.add_argument(
        "--remove-directions",
        nargs="+",
        default=list(DEFAULT_REMOVE_DIRECTIONS),
        help="Direction names to remove as an orthonormal subspace.",
    )
    parser.add_argument(
        "--baseline-prefix",
        default=None,
        help="Baseline run prefix for comparison tables.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix for residualized artifacts.",
    )
    parser.add_argument(
        "--ablation-dir",
        default="outputs/ablation",
        help="Directory for residualized embeddings and metadata.",
    )
    parser.add_argument(
        "--probe-dir", default="outputs/probes", help="Directory for probe outputs."
    )
    parser.add_argument(
        "--direction-dir",
        default="outputs/directions",
        help="Directory for direction outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        default="outputs/figures/probes",
        help="Directory for probe figures.",
    )
    parser.add_argument(
        "--table-dir",
        default="outputs/tables",
        help="Directory for combined tables and improvement summary.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Bootstrap samples for downstream direction stability.",
    )
    parser.add_argument("--random-state", type=int, default=17, help="Deterministic seed.")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only write residualized embeddings; skip downstream probes/directions.",
    )
    args = parser.parse_args()
    metadata = run_residualization_experiment(
        embeddings_path=args.embeddings,
        index_path=args.index,
        directions_path=args.directions,
        quality_path=args.quality,
        remove_directions=args.remove_directions,
        output_prefix=args.output_prefix,
        baseline_prefix=args.baseline_prefix,
        ablation_dir=args.ablation_dir,
        probe_dir=args.probe_dir,
        direction_dir=args.direction_dir,
        figure_dir=args.figure_dir,
        table_dir=args.table_dir,
        evaluate=not args.skip_evaluation,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"\nResidualization complete: {metadata['intervention_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
