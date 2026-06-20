#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Build publication figures from result tables.")
    parser.add_argument("--table-dir", default="outputs/tables", help="Directory with result CSVs.")
    parser.add_argument("--figure-dir", default="outputs/figures", help="Output figure directory.")
    parser.parse_args()
    print("Figure scaffold ready. Phase 8 will implement publication plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
