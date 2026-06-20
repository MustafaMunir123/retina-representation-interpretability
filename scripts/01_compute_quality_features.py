#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute pixel-derived bottleneck features.")
    parser.add_argument("--manifest", required=True, help="Path to image_manifest.parquet.")
    parser.parse_args()
    print("Quality feature scaffold ready. Phase 2 will implement feature extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
