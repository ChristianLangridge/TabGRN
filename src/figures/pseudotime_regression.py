"""
Pseudotime regression diagnostic plots for LinearRegressor and XGBRegressor.

Two figures per model:
  1. Observed vs. predicted pseudotime scatter
  2. Predicted vs. residual scatter (residual diagnostic)
"""

from spatialmt.config.paths import Dirs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_COLOURS = {
    "Linear Regression": "#4C72B0",
    "XGBRegressor":      "#C44E52",
}

LR_PRED_PATH  = Dirs.linear_regressor / "linear_regressor_predictions.csv"
XGB_PRED_PATH = Dirs.xgb_regressor    / "xgb_regressor_predictions.csv"

FIG_DIR = Dirs.results / "figures" / "pseudotime_regression"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predictions(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["residual"] = df["y_true"] - df["y_pred"]
    return df


# ---------------------------------------------------------------------------
# Plot 1 — observed vs. predicted scatter
# ---------------------------------------------------------------------------

def plot_obs_vs_pred(df: pd.DataFrame, model_name: str):
    colour = MODEL_COLOURS.get(model_name, "#555555")

    lo = min(df["y_true"].min(), df["y_pred"].min())
    hi = max(df["y_true"].max(), df["y_pred"].max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["y_true"], df["y_pred"], s=6, alpha=0.4, color=colour, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="y = x")

    ax.set_xlabel("Observed pseudotime")
    ax.set_ylabel("Predicted pseudotime")
    ax.set_title(f"{model_name} — observed vs. predicted pseudotime")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / f"{model_name.lower().replace(' ', '_')}_obs_vs_pred.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — predicted vs. residual scatter
# ---------------------------------------------------------------------------

def plot_pred_vs_residual(df: pd.DataFrame, model_name: str):
    colour = MODEL_COLOURS.get(model_name, "#555555")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["y_pred"], df["residual"], s=6, alpha=0.4, color=colour, linewidths=0)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

    ax.set_xlabel("Predicted pseudotime")
    ax.set_ylabel("Residual (observed − predicted)")
    ax.set_title(f"{model_name} — predicted vs. residual")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / f"{model_name.lower().replace(' ', '_')}_pred_vs_residual.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading prediction CSVs...")
    lr_df  = load_predictions(LR_PRED_PATH)
    xgb_df = load_predictions(XGB_PRED_PATH)

    print(f"Linear — {len(lr_df)} cells | XGB — {len(xgb_df)} cells")

    plot_obs_vs_pred(lr_df,  "Linear Regression")
    plot_obs_vs_pred(xgb_df, "XGBRegressor")

    plot_pred_vs_residual(lr_df,  "Linear Regression")
    plot_pred_vs_residual(xgb_df, "XGBRegressor")

    print(f"\nAll figures saved to {FIG_DIR}")
