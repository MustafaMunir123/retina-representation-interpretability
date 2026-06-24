#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


BASE_TO_SEED23 = {
    "eyepacs_resized_dinov2_resized_10k": "eyepacs_resized_dinov2_resized_10k_seed23",
    "eyepacs_cropped_dinov2_cropped_10k": "eyepacs_cropped_dinov2_cropped_10k_seed23",
    "eyepacs_resized_dinov2_resized_10k_resid_remove_sharpness_laterality": "eyepacs_resized_dinov2_resized_10k_resid_remove_sharpness_laterality_seed23",
    "eyepacs_cropped_dinov2_cropped_10k_resid_remove_sharpness_laterality": "eyepacs_cropped_dinov2_cropped_10k_resid_remove_sharpness_laterality_seed23",
}
KEY_TARGETS = [
    "any_dr",
    "referable_dr",
    "severe_dr",
    "sharpness_high_vs_low",
    "laterality_right_vs_left",
]


def _variant(run: str) -> str:
    if run.startswith("eyepacs_resized"):
        variant = "resized"
    elif run.startswith("eyepacs_cropped"):
        variant = "cropped"
    else:
        variant = "unknown"
    if "resid_remove_sharpness_laterality" in run:
        return f"{variant}_residualized"
    return variant


def _summarize_probe_robustness(main_tables: Path, robustness_tables: Path, output_dir: Path):
    import pandas as pd

    main = pd.read_csv(main_tables / "probe_metrics.csv")
    robust = pd.read_csv(robustness_tables / "probe_metrics.csv")
    rows = []
    for base_run, seed_run in BASE_TO_SEED23.items():
        base_rows = main[main["run"] == base_run].set_index("target")
        seed_rows = robust[robust["run"] == seed_run].set_index("target")
        for target in KEY_TARGETS:
            if target not in base_rows.index or target not in seed_rows.index:
                continue
            base_auc = float(base_rows.loc[target, "auc"])
            seed_auc = float(seed_rows.loc[target, "auc"])
            rows.append(
                {
                    "variant": _variant(base_run),
                    "baseline_run": base_run,
                    "robustness_run": seed_run,
                    "target": target,
                    "seed17_auc": base_auc,
                    "seed23_auc": seed_auc,
                    "seed23_minus_seed17_auc": seed_auc - base_auc,
                }
            )
    summary = pd.DataFrame(rows)
    output = output_dir / "robustness_probe_seed23_vs_seed17.csv"
    summary.to_csv(output, index=False)
    return output


def _summarize_direction_robustness(main_tables: Path, robustness_tables: Path, output_dir: Path):
    import pandas as pd

    main = pd.read_csv(main_tables / "direction_stability.csv")
    robust = pd.read_csv(robustness_tables / "direction_stability.csv")
    rows = []
    for base_run, seed_run in BASE_TO_SEED23.items():
        base_rows = main[main["run"] == base_run].set_index("direction")
        seed_rows = robust[robust["run"] == seed_run].set_index("direction")
        for direction in ["any_dr", "referable_dr", "severe_dr", "sharpness", "laterality"]:
            if direction not in base_rows.index or direction not in seed_rows.index:
                continue
            base_cosine = float(base_rows.loc[direction, "mean_cosine_to_full"])
            seed_cosine = float(seed_rows.loc[direction, "mean_cosine_to_full"])
            rows.append(
                {
                    "variant": _variant(base_run),
                    "baseline_run": base_run,
                    "robustness_run": seed_run,
                    "direction": direction,
                    "seed17_mean_cosine_to_full": base_cosine,
                    "seed23_mean_cosine_to_full": seed_cosine,
                    "seed23_minus_seed17_mean_cosine": seed_cosine - base_cosine,
                }
            )
    summary = pd.DataFrame(rows)
    output = output_dir / "robustness_direction_seed23_vs_seed17.csv"
    summary.to_csv(output, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize seed-23 robustness checks.")
    parser.add_argument("--main-tables", default="outputs/tables")
    parser.add_argument("--robustness-tables", default="outputs/robustness/tables")
    parser.add_argument("--output-dir", default="outputs/tables")
    args = parser.parse_args()

    main_tables = Path(args.main_tables)
    robustness_tables = Path(args.robustness_tables)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _summarize_probe_robustness(main_tables, robustness_tables, output_dir),
        _summarize_direction_robustness(main_tables, robustness_tables, output_dir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
