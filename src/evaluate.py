"""
Utilities to analyze aggregated results and visualize:
- Model performance vs think-token budget (sft/grpo vs baseline).
- Correlation between translation quality (COMET QE) and model performance.

Expects the CSV produced by `combine_results.py`, with per-language accuracy columns.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# COMET QE scores for instruction translation: en -> target_lang
COMET_SCORES: Dict[str, float] = {
    "chinese_simplified": 0.8170591963160543,
    "hindi": 0.8323525959859535,
    "french": 0.8526536049272896,
    "german": 0.8371817441719412,
    # english is the source; left out intentionally
}


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load combined_results CSV and ensure numeric types."""
    df = pd.read_csv(csv_path)
    df["think_tokens"] = df["think_tokens"].astype(int)
    # Identify languages by *_accuracy suffix
    lang_cols = [c for c in df.columns if c.endswith("_accuracy")]
    for col in lang_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def available_languages(df: pd.DataFrame) -> List[str]:
    return sorted(c[: -len("_accuracy")] for c in df.columns if c.endswith("_accuracy"))


def to_long(df: pd.DataFrame, languages: Sequence[str]) -> pd.DataFrame:
    """Convert wide per-language accuracy columns into a tidy DataFrame."""
    records = []
    for _, row in df.iterrows():
        for lang in languages:
            acc = row.get(f"{lang}_accuracy")
            records.append(
                {
                    "attribute": row["attribute"],
                    "model": row["model"],
                    "think_tokens": row["think_tokens"],
                    "language": lang,
                    "accuracy": acc,
                }
            )
    return pd.DataFrame.from_records(records)


def add_baseline_delta(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add accuracy deltas vs baseline for each attribute/language/think_tokens."""
    base = (
        long_df[long_df["model"] == "baseline"]
        .rename(columns={"accuracy": "baseline_accuracy"})
        .drop(columns=["model"])
    )
    merged = long_df.merge(
        base,
        on=["attribute", "language", "think_tokens"],
        how="left",
        validate="many_to_one",
    )
    merged["delta_vs_baseline"] = merged["accuracy"] - merged["baseline_accuracy"]
    return merged


def plot_accuracy_vs_budget(long_df: pd.DataFrame, outdir: Path) -> Path:
    """Line plots of accuracy vs think_tokens per language, comparing models."""
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    languages = sorted(long_df["language"].unique())
    g = sns.relplot(
        data=long_df,
        x="think_tokens",
        y="accuracy",
        hue="model",
        style="model",
        col="language",
        kind="line",
        markers=True,
        markersize=12,  # oversized markers for visibility
        linewidth=3.2,
        facet_kws={"sharey": True},
    )
    bbox_props = dict(facecolor="lightgrey", edgecolor="grey", alpha=0.7, boxstyle="round", pad=0.3)
    for ax, lang in zip(g.axes.flatten(), languages):
        ax.set_title(lang, fontsize=10, weight="bold", pad=8, bbox=bbox_props, ha="center", va="top")
        ax.set_xticks([512, 1024, 2048])
        ax.set_xticklabels(["512", "1024", "2048"])
    if g._legend:
        handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
        g._legend.remove()
        g.figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=3,
            frameon=True,
            fancybox=True,
            shadow=False,
        )
    g.set_axis_labels("Think tokens", "Accuracy")
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave room for bottom legend
    out_path = outdir / "accuracy_vs_tokens.png"
    g.figure.savefig(out_path, dpi=300)
    plt.close(g.figure)
    return out_path


def plot_improvement_vs_baseline(long_df: pd.DataFrame, outdir: Path) -> Path:
    """Plot improvement of sft/grpo over baseline across token budgets."""
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    df_delta = add_baseline_delta(long_df)
    df_delta = df_delta[df_delta["model"].isin(["sft", "grpo"])]
    languages = sorted(df_delta["language"].unique())
    g = sns.relplot(
        data=df_delta,
        x="think_tokens",
        y="delta_vs_baseline",
        hue="model",
        style="model",
        col="language",
        kind="line",
        markers=True,
        markersize=12,
        linewidth=3.2,
        facet_kws={"sharey": True},
    )
    bbox_props = dict(facecolor="lightgrey", edgecolor="grey", alpha=0.7, boxstyle="round", pad=0.3)
    for ax, lang in zip(g.axes.flatten(), languages):
        ax.set_title(lang, fontsize=10, weight="bold", pad=8, bbox=bbox_props, ha="center", va="top")
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks([512, 1024, 2048])
        ax.set_xticklabels(["512", "1024", "2048"])
    if g._legend:
        handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
        g._legend.remove()
        g.figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=2,
            frameon=True,
            fancybox=True,
            shadow=False,
        )
    g.set_axis_labels("Think tokens", "Accuracy gain vs baseline")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = outdir / "improvement_vs_baseline.png"
    g.figure.savefig(out_path, dpi=300)
    plt.close(g.figure)
    return out_path


def plot_comet_correlation(long_df: pd.DataFrame, outdir: Path) -> List[Path]:
    """
    Correlate COMET QE scores with model performance across languages.
    Generates one figure per model with subplots for each think-token budget.
    Regression shaded region is the 95% confidence interval from seaborn.regplot.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    sns.set_theme(style="whitegrid", context="talk")

    comet = pd.Series(COMET_SCORES, name="comet_score")

    # Precompute global y-range for consistent axes across plots
    sub_all = []
    for think in sorted(long_df["think_tokens"].unique()):
        for model_name in ["sft", "grpo"]:
            sub = (
                long_df[(long_df["model"] == model_name) & (long_df["think_tokens"] == think)]
                .groupby(["language"])["accuracy"]
                .mean()
                .reset_index()
            )
            sub["think_tokens"] = think
            sub_all.append(sub)
    if not sub_all:
        return paths
    combined = pd.concat(sub_all, ignore_index=True)
    y_min = combined["accuracy"].min()
    y_max = combined["accuracy"].max()
    y_margin = max(0.01, (y_max - y_min) * 0.05)
    y_bounds = (max(0, y_min - y_margin), min(1.0, y_max + y_margin))

    think_list = sorted(long_df["think_tokens"].unique())

    for model_name in ["sft", "grpo"]:
        fig, axes = plt.subplots(1, len(think_list), figsize=(5 * len(think_list), 5), sharey=True)
        if len(think_list) == 1:
            axes = [axes]

        made_plot = False
        for ax, think in zip(axes, think_list):
            sub = (
                long_df[(long_df["model"] == model_name) & (long_df["think_tokens"] == think)]
                .groupby(["language"])["accuracy"]
                .mean()
                .reset_index()
                .rename(columns={"accuracy": "mean_accuracy"})
            )
            sub["comet_score"] = sub["language"].map(comet)
            sub = sub.dropna(subset=["comet_score"])
            if sub.empty:
                ax.axis("off")
                continue
            made_plot = True
            r = sub["comet_score"].corr(sub["mean_accuracy"])
            sns.regplot(
                data=sub,
                x="comet_score",
                y="mean_accuracy",
                scatter_kws={"s": 80},
                line_kws={"color": "gray"},
                ax=ax,
            )
            for _, row in sub.iterrows():
                ax.text(row["comet_score"], row["mean_accuracy"], row["language"], fontsize=9)
            ax.set_title(f"{model_name.upper()} @ {think}", fontsize=11, weight="bold")
            ax.set_xlabel("COMET QE (en → target)")
            ax.set_ylabel("Mean accuracy")
            ax.set_ylim(y_bounds)
            ax.text(0.02, 0.95, f"r = {r:.3f}", transform=ax.transAxes, ha="left", va="top")

        if made_plot:
            plt.tight_layout()
            out_path = outdir / f"comet_corr_{model_name}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            paths.append(out_path)
        else:
            plt.close(fig)
    return paths


def plot_radar_attributes(long_df: pd.DataFrame, outdir: Path) -> Path:
    """
    Radar charts of attribute-wise accuracy for each model/think-token combo.
    One row per model (sft, grpo); columns are think-token budgets.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    think_list = sorted(long_df["think_tokens"].unique())
    attributes = sorted(long_df["attribute"].unique())
    num_attrs = len(attributes)
    angles = np.linspace(0, 2 * np.pi, num_attrs, endpoint=False).tolist()
    angles += angles[:1]

    colors = sns.color_palette("husl", 2)
    bbox_props = dict(facecolor="lightgrey", edgecolor="grey", alpha=0.7, boxstyle="round", pad=0.3)

    fig, axes = plt.subplots(
        2, len(think_list), figsize=(4 * len(think_list), 7), subplot_kw=dict(polar=True), sharey=True
    )
    if len(think_list) == 1:
        axes = np.array([[axes[0]], [axes[1]]]) if isinstance(axes, np.ndarray) else np.array([[axes], [axes]])

    for row, model_name in enumerate(["sft", "grpo"]):
        for col, think in enumerate(think_list):
            ax = axes[row, col]
            sub = (
                long_df[(long_df["model"] == model_name) & (long_df["think_tokens"] == think)]
                .groupby("attribute")["accuracy"]
                .mean()
            )
            values = sub.reindex(attributes).to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            values_closed = list(values) + [values[0]]

            color = colors[row]
            ax.plot(angles, values_closed, color=color, linewidth=2.5, linestyle="solid", marker="o")
            ax.fill(angles, values_closed, color=color, alpha=0.2)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(attributes, fontsize=9, rotation=90, ha="center")
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_ylim(0, 1.0)
            ax.grid(True, color="#d3d3d3", linestyle="--")
            ax.spines["polar"].set_color("#d3d3d3")
            ax.spines["polar"].set_linewidth(0.6)
            ax.set_title(
                f"{model_name.upper()} @ {think}",
                fontsize=10,
                weight="bold",
                pad=14,
                bbox=bbox_props,
                ha="center",
                va="top",
            )

    plt.tight_layout()
    out_path = outdir / "radar_attribute_accuracy.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze combined results.")
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "combined_results.csv",
        help="Path to combined_results.csv produced by combine_results.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "analysis_plots",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    df = load_results(args.results_csv)
    langs = available_languages(df)
    long_df = to_long(df, langs)

    out_paths = []
    out_paths.append(plot_accuracy_vs_budget(long_df, args.outdir))
    out_paths.append(plot_improvement_vs_baseline(long_df, args.outdir))
    out_paths.extend(plot_comet_correlation(long_df, args.outdir))
    out_paths.append(plot_radar_attributes(long_df, args.outdir))

    print("Saved plots:")
    for p in out_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
