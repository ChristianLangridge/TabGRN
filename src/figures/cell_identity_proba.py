"""
Cell identity probability plots for NearestCentroid and XGBClassifier.

Three figures per model:
  1. Mean probability per cell type stacked by timepoint (trajectory sanity check)
  2. Per-cell stacked bar (random sample, ordered by dominant cell type)
  3. Max-probability confidence histogram comparing both models
"""

from spatialmt.config.paths import Dirs

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CELL_TYPE_COLOURS = {
    "Neurectoderm":                    "#4C72B0",
    "Late Neurectoderm":               "#64B5CD",
    "Prosencephalic progenitors":      "#55A868",
    "Late Prosencephalic progenitors": "#8FBC8F",
    "Telencephalic progenitors":       "#C44E52",
    "Diencephalic progenitors":        "#DD8452",
    "Tel/Die neurons":                 "#9467BD",
    "Unknown proliferating cells":     "#8C8C8C",
}

DAY_ORDER = ["D5", "D7", "D11", "D16", "D21", "D30"]

NC_PROBA_PATH  = Dirs.results / "trained_models" / "cell_identity" / "nearest_centroid" / "nearest_centroid_proba.csv"
XGB_PROBA_PATH = Dirs.results / "trained_models" / "cell_identity" / "xgb_classifier"   / "xgb_classifier_proba.csv"

FIG_DIR = Dirs.results / "figures" / "cell_identity"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_day(cell_id: str) -> str:
    """Pull day label from barcode e.g. 'HB4 D7_TACATTC...' → 'D7'."""
    match = re.search(r"D(\d+)", cell_id)
    return f"D{match.group(1)}" if match else "Unknown"


def load_proba(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index.name = "cell_id"
    df["day"] = df.index.map(extract_day)
    return df


def cell_type_colours(columns):
    return [CELL_TYPE_COLOURS.get(c, "#AAAAAA") for c in columns]


# ---------------------------------------------------------------------------
# Plot 1 — mean probability per timepoint (stacked bar)
# ---------------------------------------------------------------------------

def plot_mean_by_day(df: pd.DataFrame, model_name: str):
    cell_types = [c for c in df.columns if c != "day"]
    colours    = cell_type_colours(cell_types)

    mean_df = (
        df.groupby("day")[cell_types]
          .mean()
          .reindex([d for d in DAY_ORDER if d in df["day"].values])
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(mean_df))
    for col, colour in zip(cell_types, colours):
        ax.bar(mean_df.index, mean_df[col], bottom=bottom,
               color=colour, label=col, width=0.6, edgecolor="white", linewidth=0.4)
        bottom += mean_df[col].values

    ax.set_ylabel("Mean predicted probability")
    ax.set_xlabel("Timepoint")
    ax.set_title(f"{model_name} — mean cell identity probability by timepoint")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / f"{model_name.lower().replace(' ', '_')}_mean_by_day.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — per-cell stacked bar (sample of 500, ordered by dominant type)
# ---------------------------------------------------------------------------

def plot_per_cell(df: pd.DataFrame, model_name: str, n_cells: int = 500):
    cell_types = [c for c in df.columns if c != "day"]
    colours    = cell_type_colours(cell_types)

    sample = df.sample(n=min(n_cells, len(df)), random_state=42)
    dominant = sample[cell_types].idxmax(axis=1)
    order = dominant.map({ct: i for i, ct in enumerate(cell_types)}).argsort()
    sample = sample.iloc[order]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(sample))
    bottom = np.zeros(len(sample))
    for col, colour in zip(cell_types, colours):
        ax.bar(x, sample[col].values, bottom=bottom,
               color=colour, label=col, width=1.0, linewidth=0)
        bottom += sample[col].values

    ax.set_ylabel("Predicted probability")
    ax.set_xlabel(f"Cells (n={len(sample)}, ordered by dominant identity)")
    ax.set_title(f"{model_name} — per-cell identity probability distribution")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xticks([])
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / f"{model_name.lower().replace(' ', '_')}_per_cell.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — confidence histogram (both models overlaid)
# ---------------------------------------------------------------------------

def plot_confidence_comparison(nc_df: pd.DataFrame, xgb_df: pd.DataFrame):
    cell_types_nc  = [c for c in nc_df.columns  if c != "day"]
    cell_types_xgb = [c for c in xgb_df.columns if c != "day"]

    nc_conf  = nc_df[cell_types_nc].max(axis=1)
    xgb_conf = xgb_df[cell_types_xgb].max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(nc_conf,  bins=bins, alpha=0.6, label="Nearest Centroid", color="#4C72B0", density=True)
    ax.hist(xgb_conf, bins=bins, alpha=0.6, label="XGBClassifier",    color="#C44E52", density=True)

    ax.axvline(nc_conf.median(),  color="#4C72B0", linestyle="--", linewidth=1.2,
               label=f"NC median  {nc_conf.median():.2f}")
    ax.axvline(xgb_conf.median(), color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"XGB median {xgb_conf.median():.2f}")

    ax.set_xlabel("Max predicted probability (model confidence)")
    ax.set_ylabel("Density")
    ax.set_title("Model confidence: Nearest Centroid vs XGBClassifier")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / "confidence_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading probability CSVs...")
    nc_df  = load_proba(NC_PROBA_PATH)
    xgb_df = load_proba(XGB_PROBA_PATH)

    print(f"NC  — {len(nc_df)} cells | XGB — {len(xgb_df)} cells")

    plot_mean_by_day(nc_df,  "Nearest Centroid")
    plot_mean_by_day(xgb_df, "XGBClassifier")

    plot_per_cell(nc_df,  "Nearest Centroid")
    plot_per_cell(xgb_df, "XGBClassifier")

    plot_confidence_comparison(nc_df, xgb_df)

    print(f"\nAll figures saved to {FIG_DIR}")
