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
    "eyepacs_resized_dinov2_resized_full": "Resized",
    "eyepacs_cropped_dinov2_cropped_35k": "Cropped",
    "eyepacs_resized_dinov2_resized_full_resid_remove_sharpness_laterality": "Resized - S/L",
    "eyepacs_cropped_dinov2_cropped_35k_resid_remove_sharpness_laterality": "Cropped - S/L",
}
BASELINE_RUNS = [
    "eyepacs_resized_dinov2_resized_full",
    "eyepacs_cropped_dinov2_cropped_35k",
]
INTERVENTION_RUNS = [
    "eyepacs_resized_dinov2_resized_full",
    "eyepacs_resized_dinov2_resized_full_resid_remove_sharpness_laterality",
    "eyepacs_cropped_dinov2_cropped_35k",
    "eyepacs_cropped_dinov2_cropped_35k_resid_remove_sharpness_laterality",
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
    ax.set_ylim(0.45, 1.0)
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
        pivot["eyepacs_resized_dinov2_resized_full"],
        height=height,
        color="#2f6f9f",
        label="Resized",
    )
    ax.barh(
        y + height / 2,
        pivot["eyepacs_cropped_dinov2_cropped_35k"],
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
    baseline = "eyepacs_resized_dinov2_resized_full"
    intervention = "eyepacs_resized_dinov2_resized_full_resid_remove_sharpness_laterality"
    rows = probes[probes["target"].isin(targets)].copy()
    base = rows[rows["run"] == baseline].set_index("target")["auc"]
    resid = rows[rows["run"] == intervention].set_index("target")["auc"]
    values = np.array([float(resid[target] - base[target]) for target in targets])
    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = ["#2f6f9f" if target in DISEASE_TARGETS else "#c55a4c" for target in targets]
    ax.bar(x, values, width=0.62, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel("AUC after removal minus baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[target] for target in targets], rotation=25, ha="right")
    ax.set_title("Full resized AUC change after sharpness/laterality removal")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    output = figure_dir / "residualization_auc_delta.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def _save_pipeline_figure(figure_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def box(ax, xy, width, height, title, color):
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0,
            facecolor=color,
        )
        ax.add_patch(patch)
        x, y = xy
        ax.text(
            x + width / 2,
            y + height * 0.50,
            title,
            ha="center",
            va="center",
            color="white",
            fontsize=10.5,
            fontweight="bold",
        )

    def arrow(ax, start, end, color="#4a4a4a", style="-", rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2.0,
                color=color,
                linestyle=style,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    blue = "#3d73a8"
    green = "#4b934b"
    purple = "#8060a2"
    red = "#c04a3d"
    gray = "#4a4a4a"

    ax.text(
        3.4,
        3.32,
        "Embedding audit pipeline",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="normal",
    )

    box(ax, (0.25, 2.25), 1.15, 0.62, "Images", blue)
    box(ax, (1.75, 2.25), 1.15, 0.62, "DINOv2", blue)
    box(ax, (3.25, 2.25), 1.15, 0.62, "Embeds", blue)
    box(ax, (4.82, 2.25), 1.15, 0.62, "Mean-diff\naxes", purple)

    box(ax, (0.25, 1.15), 1.15, 0.62, "DR\nlabels", green)
    box(ax, (1.75, 1.15), 1.35, 0.62, "Quality\nfeatures", green)

    box(ax, (3.05, 0.20), 1.05, 0.62, "Probes", red)
    box(ax, (4.35, 0.20), 1.05, 0.62, "Overlap", red)
    box(ax, (5.65, 0.20), 1.05, 0.62, "Remove\naxes", red)

    arrow(ax, (1.42, 2.56), (1.72, 2.56), gray)
    arrow(ax, (2.92, 2.56), (3.22, 2.56), gray)
    arrow(ax, (4.42, 2.56), (4.78, 2.56), gray)

    arrow(ax, (0.83, 2.23), (0.83, 1.80), green, style="--")
    arrow(ax, (1.42, 1.46), (1.72, 1.46), green)
    arrow(ax, (3.12, 1.48), (4.78, 2.35), green, rad=-0.12)
    arrow(ax, (0.85, 1.80), (4.78, 2.45), green, rad=-0.10)

    arrow(ax, (3.82, 2.23), (3.58, 0.85), gray)
    arrow(ax, (5.40, 2.23), (4.88, 0.85), gray)
    arrow(ax, (5.40, 2.23), (6.18, 0.85), gray)

    output = figure_dir / "pipeline_schematic.png"
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
        _save_pipeline_figure(figure_dir),
        _save_dr_auc_figure(tables["probes"], figure_dir),
        _save_projection_figure(tables["projection"], figure_dir),
        _save_residualization_figure(tables["probes"], figure_dir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
