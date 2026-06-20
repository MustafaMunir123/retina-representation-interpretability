#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare preprocessing variants.")
    parser.add_argument("--baseline-table", required=True, help="Baseline metrics CSV.")
    parser.add_argument("--comparison-table", required=True, help="Comparison metrics CSV.")
    parser.parse_args()
    print("Preprocessing comparison scaffold ready. Phase 7 will implement comparisons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
