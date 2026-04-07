"""
Batch effect QC diagnostics for anomalous D21 pseudotime.

Generates per-timepoint violin plots for:
  1. Mitochondrial gene fraction (% MT)
  2. Ribosomal gene fraction (% RB)
  3. S.Score distribution
  4. G2M.Score distribution

All figures saved to results/figures/batch_qc/.
"""

from spatialmt.config.paths import Dirs

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DAY_ORDER = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]

FIG_DIR = Dirs.results / "figures" / "batch_qc"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violin(data_by_day: list, ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    parts = ax.violinplot(data_by_day, positions=range(len(DAY_ORDER)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)

    # Overlay per-cell jitter for small-n visibility
    for i, vals in enumerate(data_by_day):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, size=min(len(vals), 500))
        sample = np.random.default_rng(i).choice(vals, size=min(len(vals), 500), replace=False)
        ax.scatter(i + jitter, sample, s=1, alpha=0.2, color="steelblue", linewidths=0)

    ax.set_xticks(range(len(DAY_ORDER)))
    ax.set_xticklabels(DAY_ORDER)
    ax.set_xlabel("Collection day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# QC metric computation
# ---------------------------------------------------------------------------

def compute_qc_metrics(adata: sc.AnnData) -> sc.AnnData:
    """Annotate MT and RB fractions in adata.obs if not already present."""
    if "pct_counts_mt" not in adata.obs.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )
        print(f"MT genes found: {adata.var['mt'].sum()}")

    if "pct_counts_rb" not in adata.obs.columns:
        adata.var["rb"] = (
            adata.var_names.str.upper().str.startswith("RPS") |
            adata.var_names.str.upper().str.startswith("RPL")
        )
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["rb"], percent_top=None, log1p=False, inplace=True
        )
        print(f"Ribosomal genes found: {adata.var['rb'].sum()}")

    return adata


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mt_fraction(adata: sc.AnnData) -> None:
    data = [
        adata.obs.loc[adata.obs["orig.ident"] == d, "pct_counts_mt"].dropna().values
        for d in DAY_ORDER
    ]
    _violin(
        data,
        ylabel="Mitochondrial gene fraction (%)",
        title="MT gene fraction by collection day",
        filename="mt_fraction_by_day.png",
    )


def plot_rb_fraction(adata: sc.AnnData) -> None:
    data = [
        adata.obs.loc[adata.obs["orig.ident"] == d, "pct_counts_rb"].dropna().values
        for d in DAY_ORDER
    ]
    _violin(
        data,
        ylabel="Ribosomal gene fraction (%)",
        title="Ribosomal gene fraction by collection day",
        filename="rb_fraction_by_day.png",
    )


def plot_cell_cycle_scores(adata: sc.AnnData) -> None:
    for score, label, fname in [
        ("S.Score",   "S phase score",   "s_score_by_day.png"),
        ("G2M.Score", "G2M phase score", "g2m_score_by_day.png"),
    ]:
        if score not in adata.obs.columns:
            print(f"Warning: {score} not in obs — skipping")
            continue
        data = [
            adata.obs.loc[adata.obs["orig.ident"] == d, score].dropna().values
            for d in DAY_ORDER
        ]
        _violin(
            data,
            ylabel=label,
            title=f"{label} by collection day",
            filename=fname,
        )


def print_summary_stats(adata: sc.AnnData) -> None:
    cols = [c for c in ["pct_counts_mt", "pct_counts_rb", "S.Score", "G2M.Score"]
            if c in adata.obs.columns]
    summary = adata.obs.groupby("orig.ident")[cols].agg(["mean", "median", "std"])
    summary = summary.reindex([d for d in DAY_ORDER if d in summary.index])
    print("\n── Per-day QC summary ──")
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Progenitor composition
# ---------------------------------------------------------------------------

PROGENITOR_TYPES = [
    "Prosencephalic progenitors",
    "Late Prosencephalic progenitors",
    "Telencephalic progenitors",
    "Diencephalic progenitors",
]

PROGENITOR_COLOURS = {
    "Prosencephalic progenitors":      "#55A868",
    "Late Prosencephalic progenitors": "#8FBC8F",
    "Telencephalic progenitors":       "#C44E52",
    "Diencephalic progenitors":        "#DD8452",
}


def print_progenitor_counts(adata: sc.AnnData) -> pd.DataFrame:
    """Print and return raw progenitor cell counts per collection day."""
    present = [p for p in PROGENITOR_TYPES if p in adata.obs["class3"].values]
    obs_prog = adata.obs[adata.obs["class3"].isin(present)]

    counts = (
        obs_prog.groupby(["orig.ident", "class3"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(DAY_ORDER, fill_value=0)
    )
    day_totals = adata.obs.groupby("orig.ident", observed=True).size().reindex(DAY_ORDER)
    pct = counts.div(day_totals, axis=0) * 100

    print("\n── Progenitor counts per day ──")
    print(counts.to_string())
    print("\n── Progenitor % of day total ──")
    print(pct.round(2).to_string())
    return counts, pct


def plot_progenitor_composition(adata: sc.AnnData) -> None:
    counts, pct = print_progenitor_counts(adata)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts stacked bar
    bottom = np.zeros(len(DAY_ORDER))
    for ct in pct.columns:
        colour = PROGENITOR_COLOURS.get(ct, "#AAAAAA")
        axes[0].bar(DAY_ORDER, counts[ct].values, bottom=bottom,
                    color=colour, label=ct, edgecolor="white", linewidth=0.4)
        bottom += counts[ct].values
    axes[0].set_ylabel("Cell count")
    axes[0].set_title("Progenitor counts by collection day")
    axes[0].set_xticks(range(len(DAY_ORDER)))
    axes[0].set_xticklabels(DAY_ORDER, rotation=20, ha="right")
    axes[0].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    axes[0].spines[["top", "right"]].set_visible(False)

    # % of day total stacked bar
    bottom = np.zeros(len(DAY_ORDER))
    for ct in pct.columns:
        colour = PROGENITOR_COLOURS.get(ct, "#AAAAAA")
        axes[1].bar(DAY_ORDER, pct[ct].values, bottom=bottom,
                    color=colour, label=ct, edgecolor="white", linewidth=0.4)
        bottom += pct[ct].values
    axes[1].set_ylabel("% of day total")
    axes[1].set_title("Progenitor fraction by collection day")
    axes[1].set_xticks(range(len(DAY_ORDER)))
    axes[1].set_xticklabels(DAY_ORDER, rotation=20, ha="right")
    axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "progenitor_composition_by_day.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# kBET and LISI
# ---------------------------------------------------------------------------

def _compute_lisi(X_pca: np.ndarray, labels: np.ndarray, n_neighbors: int = 90) -> np.ndarray:
    """
    Pure-Python LISI implementation using sklearn NearestNeighbors.
    Returns per-cell LISI scores. Higher = more diverse neighbourhood (better mixed).
    Formula from Korsunsky et al. 2019 (Harmony paper).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(X_pca)
    _, indices = nn.kneighbors(X_pca)

    unique_labels = np.unique(labels)
    lisi_scores = np.zeros(len(labels))

    for i, neighbours in enumerate(indices):
        neighbour_labels = labels[neighbours]
        perplexity = 0.0
        for lab in unique_labels:
            p = np.mean(neighbour_labels == lab)
            if p > 0:
                perplexity += p * np.log(p)
        lisi_scores[i] = np.exp(-perplexity)

    return lisi_scores


def run_lisi(adata: sc.AnnData) -> None:
    print("\nRunning LISI (pure Python)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=10, svd_solver="arpack")

    print("Computing LISI scores...")
    labels = adata.obs["orig.ident"].values.astype(str)
    lisi_scores = _compute_lisi(adata.obsm["X_pca"], labels)
    adata.obs["iLISI"] = lisi_scores

    lisi_by_day = [
        adata.obs.loc[adata.obs["orig.ident"] == d, "iLISI"].dropna().values
        for d in DAY_ORDER
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(lisi_by_day, positions=range(len(DAY_ORDER)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(DAY_ORDER)))
    ax.set_xticklabels(DAY_ORDER)
    ax.set_xlabel("Collection day")
    ax.set_ylabel("LISI score")
    ax.set_title(f"LISI score by collection day (max={len(DAY_ORDER)}, higher = better mixed)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "lisi_by_day.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    lisi_summary = adata.obs.groupby("orig.ident", observed=True)["iLISI"].agg(
        ["mean", "median", "std"]
    ).reindex(DAY_ORDER)
    print("\n── LISI summary per day ──")
    print(lisi_summary.round(4).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    h5ad_path = Dirs.model_data_anndata / "neurectoderm_complete.h5ad"
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # QC metrics must be computed on raw counts before normalisation
    print("\nComputing QC metrics on raw counts...")
    adata = compute_qc_metrics(adata)

    plot_mt_fraction(adata)
    plot_rb_fraction(adata)
    plot_cell_cycle_scores(adata)
    print_summary_stats(adata)

    print("\nAnalysing progenitor composition...")
    plot_progenitor_composition(adata)

    print("\nRunning batch mixing metrics...")
    run_lisi(adata)

    print(f"\nAll figures saved to {FIG_DIR}")
