#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retina_audit.quality import compute_quality_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute pixel-derived bottleneck features.")
    parser.add_argument("--manifest", required=True, help="Path to image_manifest.parquet.")
    parser.add_argument(
        "--output",
        default="outputs/quality/quality_features.parquet",
        help="Output parquet path for quality features.",
    )
    parser.add_argument(
        "--summary",
        default="outputs/quality/quality_summary.json",
        help="Output JSON path for quality summary and go/no-go.",
    )
    parser.add_argument(
        "--figure-dir",
        default="outputs/figures/quality",
        help="Directory for histograms and sample grids.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument("--shard-id", type=int, default=None, help="Optional deterministic shard id.")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shard count.")
    parser.add_argument("--no-figures", action="store_true", help="Skip histograms and sample grids.")
    parser.add_argument(
        "--samples-per-grid",
        type=int,
        default=12,
        help="Images per high/low sample grid.",
    )
    args = parser.parse_args()
    summary = compute_quality_features(
        args.manifest,
        output_path=args.output,
        figure_dir=args.figure_dir,
        summary_path=args.summary,
        limit=args.limit,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        make_figures=not args.no_figures,
        samples_per_grid=args.samples_per_grid,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nQuality feature decision: {summary['go_no_go']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
