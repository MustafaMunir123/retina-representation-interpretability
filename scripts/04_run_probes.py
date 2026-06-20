#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Run linear probe sanity checks.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file.")
    parser.add_argument("--index", required=True, help="Path to embedding index parquet.")
    parser.parse_args()
    print("Probe scaffold ready. Phase 4 will implement probe evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
