#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


DISEASE_TARGETS = ["any_dr", "referable_dr", "severe_dr"]
DISEASE_LABELS = {
    "any_dr": "Any DR",
    "referable_dr": "Referable DR",
    "severe_dr": "Severe DR",
}
RUN_LABELS = {
    "eyepacs_resized_dinov2_resized_10k": "Resized",
    "eyepacs_cropped_dinov2_cropped_10k": "Cropped",
    "eyepacs_resized_dinov2_resized_10k_resid_remove_sharpness_laterality": "Resized - S/L",
    "eyepacs_cropped_dinov2_cropped_10k_resid_remove_sharpness_laterality": "Cropped - S/L",
}
BASELINE_RUNS = [
    "eyepacs_resized_dinov2_resized_10k",
    "eyepacs_cropped_dinov2_cropped_10k",
]
INTERVENTION_RUNS = [
    "eyepacs_resized_dinov2_resized_10k",
    "eyepacs_resized_dinov2_resized_10k_resid_remove_sharpness_laterality",
    "eyepacs_cropped_dinov2_cropped_10k",
    "eyepacs_cropped_dinov2_cropped_10k_resid_remove_sharpness_laterality",
]
BOTTLENECK_LABELS = {
    "sharpness": "Sharpness",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "black_border": "Black border",
    "field_of_view": "Field of view",
    "laterality": "Laterality",
}
TARGET_LABELS = {
    **DISEASE_LABELS,
    "sharpness_high_vs_low": "Sharpness",
    "laterality_right_vs_left": "Laterality",
}


def _setup_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "font.size": 8.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _read_tables(table_dir: Path):
    import pandas as pd

    return {
        "probes": pd.read_csv(table_dir / "probe_metrics.csv"),
        "projection": pd.read_csv(table_dir / "projection_ratios.csv"),
    }


def _save_dr_auc_figure(probes, figure_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = probes[
        probes["run"].isin(INTERVENTION_RUNS) & probes["target"].isin(DISEASE_TARGETS)
    ].copy()
    pivot = rows.pivot_table(index="target", columns="run", values="auc").reindex(DISEASE_TARGETS)
    x = np.arange(len(DISEASE_TARGETS))
    width = 0.18
    colors = ["#2f6f9f", "#7cb7b8", "#c55a4c", "#d9a441"]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for offset, run in enumerate(INTERVENTION_RUNS):
        values = pivot[run].to_numpy()
        ax.bar(
            x + (offset - 1.5) * width,
            values,
            width=width,
            label=RUN_LABELS[run],
            color=colors[offset],
        )
    ax.axhline(0.5, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("Held-out AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([DISEASE_LABELS[target] for target in DISEASE_TARGETS])
    ax.set_title("DR probe AUC across preprocessing and intervention variants")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    output = figure_dir / "dr_probe_auc_comparison.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def _save_projection_figure(projection, figure_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = projection[
        projection["run"].isin(BASELINE_RUNS)
        & projection["disease_direction"].isin(DISEASE_TARGETS)
    ].copy()
    rows["pair"] = rows["disease_direction"] + " -> " + rows["bottleneck_direction"]
    top_pairs = (
        rows.groupby("pair")["projection_ratio"]
        .max()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    rows = rows[rows["pair"].isin(top_pairs)].copy()
    rows["pair_label"] = rows.apply(
        lambda row: f"{DISEASE_LABELS[row['disease_direction']]} -> "
        f"{BOTTLENECK_LABELS[row['bottleneck_direction']]}",
        axis=1,
    )
    order = (
        rows.groupby("pair_label")["projection_ratio"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    pivot = rows.pivot_table(index="pair_label", columns="run", values="projection_ratio").reindex(
        order
    )
    y = np.arange(len(order))
    height = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(
        y - height / 2,
        pivot["eyepacs_resized_dinov2_resized_10k"],
        height=height,
        color="#2f6f9f",
        label="Resized",
    )
    ax.barh(
        y + height / 2,
        pivot["eyepacs_cropped_dinov2_cropped_10k"],
        height=height,
        color="#7cb7b8",
        label="Cropped",
    )
    ax.set_xlabel("Projection ratio (squared cosine)")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_title("Top disease-to-bottleneck direction overlaps")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    output = figure_dir / "top_projection_ratios.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def _save_residualization_figure(probes, figure_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    targets = [
        "any_dr",
        "referable_dr",
        "severe_dr",
        "sharpness_high_vs_low",
        "laterality_right_vs_left",
    ]
    run_pairs = [
        (
            "eyepacs_resized_dinov2_resized_10k",
            "eyepacs_resized_dinov2_resized_10k_resid_remove_sharpness_laterality",
            "Resized",
        ),
        (
            "eyepacs_cropped_dinov2_cropped_10k",
            "eyepacs_cropped_dinov2_cropped_10k_resid_remove_sharpness_laterality",
            "Cropped",
        ),
    ]
    rows = probes[probes["target"].isin(targets)].copy()
    deltas = []
    for baseline, intervention, label in run_pairs:
        base = rows[rows["run"] == baseline].set_index("target")["auc"]
        resid = rows[rows["run"] == intervention].set_index("target")["auc"]
        for target in targets:
            deltas.append(
                {
                    "variant": label,
                    "target": target,
                    "auc_delta": float(resid[target] - base[target]),
                }
            )
    import pandas as pd

    data = pd.DataFrame(deltas)
    x = np.arange(len(targets))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for offset, (variant, color) in enumerate([("Resized", "#2f6f9f"), ("Cropped", "#7cb7b8")]):
        values = (
            data[data["variant"] == variant].set_index("target").reindex(targets)["auc_delta"]
        )
        ax.bar(
            x + (offset - 0.5) * width,
            values,
            width=width,
            label=variant,
            color=color,
        )
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel("AUC after removal minus baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[target] for target in targets], rotation=25, ha="right")
    ax.set_title("AUC change after sharpness/laterality subspace removal")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    output = figure_dir / "residualization_auc_delta.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build submission figures from result tables.")
    parser.add_argument("--table-dir", default="outputs/tables", help="Directory with result CSVs.")
    parser.add_argument(
        "--figure-dir",
        default="outputs/figures/submission",
        help="Output directory for submission figures.",
    )
    args = parser.parse_args()
    _setup_matplotlib()
    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = _read_tables(table_dir)
    outputs = [
        _save_dr_auc_figure(tables["probes"], figure_dir),
        _save_projection_figure(tables["projection"], figure_dir),
        _save_residualization_figure(tables["probes"], figure_dir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
