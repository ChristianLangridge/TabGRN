"""
Diffusion pseudotime computation for brain organoid scRNA-seq.

Structured as a function module (mirrors prep.py) — called from data_prep.py.

Entry point: compute_dpt_from_css_embedding
--------------------------------------------
Uses a CSS-PCA-CC embedding produced by css_pseudotime.R (simspec).
Skips HVG selection, cell-cycle regression, scaling, and PCA — handled in R.

Pipeline
--------
1.  Exclude "Unknown proliferating cells" (off-trajectory)
2.  Inject CSS embedding as X_pca → neighbour graph (k=20) → diffusion map (15 components)
3.  Root cell: highest POU5F1 in HB4_D5 (fallback: random D5 cell)
4.  DPT → rank-transform → pseudotime ∈ (0, 1]
5.  Post-hoc pseudotime for excluded proliferating cells (NN in CSS space)
6.  Save full h5ad to model_data_anndata
"""

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from spatialmt.config.paths import Dirs


# ---------------------------------------------------------------------------
# Step 1 — exclude off-trajectory cells
# ---------------------------------------------------------------------------

def exclude_proliferating(adata: sc.AnnData, cell_type_key: str = "class3") -> tuple[sc.AnnData, sc.AnnData]:
    """
    Split adata into trajectory cells and excluded proliferating cells.

    Returns
    -------
    adata_traj : AnnData  — cells used for diffusion pseudotime
    adata_prolif : AnnData — excluded "Unknown proliferating cells"
    """
    mask = adata.obs[cell_type_key] != "Unknown proliferating cells"
    adata_traj   = adata[mask].copy()
    adata_prolif = adata[~mask].copy()
    print(f"Trajectory cells : {adata_traj.n_obs}")
    print(f"Excluded (prolif): {adata_prolif.n_obs}")
    return adata_traj, adata_prolif


# ---------------------------------------------------------------------------
# Steps 2-3 — neighbours, diffusion map, root selection
# ---------------------------------------------------------------------------

def select_root(adata: sc.AnnData, day_key: str = "orig.ident", root_day: str = "HB4_D5") -> sc.AnnData:
    """
    Set iroot to the D5 cell with highest POU5F1 expression.
    Falls back to a random D5 cell if POU5F1 is absent.
    """
    d5_mask = adata.obs[day_key] == root_day
    d5_idx  = np.where(d5_mask)[0]

    if "POU5F1" in adata.var_names:
        gene_idx = adata.var_names.get_loc("POU5F1")
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        expr = X[d5_idx, gene_idx]
        root = d5_idx[np.argmax(expr)]
        print(f"Root cell: D5 cell with highest POU5F1 (index {root})")
    else:
        warnings.warn("POU5F1 not found in var_names — selecting random HB4_D5 cell as root")
        root = int(np.random.choice(d5_idx))
        print(f"Root cell: random D5 cell (index {root})")

    adata.uns["iroot"] = root
    return adata


# ---------------------------------------------------------------------------
# Step 9-10 — DPT + rank transform
# ---------------------------------------------------------------------------

def compute_dpt(adata: sc.AnnData, n_dcs: int = 10) -> sc.AnnData:
    sc.tl.dpt(adata, n_dcs=n_dcs)
    adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"].rank() / len(adata)
    print("DPT computed. Rank-transformed pseudotime ∈ (0, 1]")
    return adata


# ---------------------------------------------------------------------------
# Step 11 — post-hoc pseudotime for excluded proliferating cells
# ---------------------------------------------------------------------------

def assign_prolif_pseudotime(
    adata_traj: sc.AnnData,
    adata_prolif: sc.AnnData,
    prolif_embedding: np.ndarray | None = None,
) -> sc.AnnData:
    """
    Assign pseudotime to excluded proliferating cells via nearest neighbour
    in PCA space (fitted on trajectory cells).

    Parameters
    ----------
    prolif_embedding : optional pre-computed embedding for prolif cells (n_prolif × n_dims).
        When provided (e.g. CSS embedding), skip the gene-space projection step.
        Must be aligned to adata_prolif row order.
    """
    if adata_prolif.n_obs == 0:
        adata_prolif.obs["pseudotime"] = pd.Series(dtype=float)
        adata_prolif.obs["dpt_pseudotime"] = pd.Series(dtype=float)
        return adata_prolif

    print("Assigning pseudotime to proliferating cells via NN in PCA space...")

    traj_pca = adata_traj.obsm["X_pca"]

    if prolif_embedding is not None:
        # CSS path: embedding already in the same space as traj X_pca
        prolif_pca = prolif_embedding.astype(np.float32)
    else:
        # Standard PCA path: project scaled gene expression through stored loadings
        shared_genes = adata_traj.var_names.intersection(adata_prolif.var_names)

        prolif_X = adata_prolif[:, shared_genes].X
        if hasattr(prolif_X, "toarray"):
            prolif_X = prolif_X.toarray()

        traj_X = adata_traj[:, shared_genes].X
        if hasattr(traj_X, "toarray"):
            traj_X = traj_X.toarray()

        mean = traj_X.mean(axis=0)
        std  = traj_X.std(axis=0)
        std[std == 0] = 1.0
        prolif_scaled = (prolif_X - mean) / std
        prolif_scaled = np.clip(prolif_scaled, -10, 10)

        pca_components = adata_traj.varm["PCs"]  # (n_hvgs, n_comps)
        shared_idx = [list(adata_traj.var_names).index(g) for g in shared_genes]
        prolif_pca = prolif_scaled @ pca_components[shared_idx, :]

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(traj_pca)
    _, indices = nn.kneighbors(prolif_pca)

    neighbour_pseudotime = adata_traj.obs["pseudotime"].values[indices[:, 0]]
    adata_prolif.obs["pseudotime"] = neighbour_pseudotime.values if hasattr(neighbour_pseudotime, "values") else neighbour_pseudotime
    adata_prolif.obs["dpt_pseudotime"] = np.nan

    print(f"Assigned pseudotime to {adata_prolif.n_obs} proliferating cells")
    return adata_prolif


# ---------------------------------------------------------------------------
# Step 12 — merge and save
# ---------------------------------------------------------------------------

def merge_and_save(adata_traj: sc.AnnData, adata_prolif: sc.AnnData, original: sc.AnnData) -> sc.AnnData:
    """
    Merge pseudotime back into the full original AnnData and save.
    """
    pseudotime = pd.concat([
        adata_traj.obs[["pseudotime", "dpt_pseudotime"]],
        adata_prolif.obs[["pseudotime", "dpt_pseudotime"]],
    ])
    original.obs["rank-transformed-pseudotime"] = pseudotime["pseudotime"]
    original.obs["raw_pseudotime"]              = pseudotime["dpt_pseudotime"]

    out_path = Dirs.model_data_anndata / "neurectoderm_with_pseudotime.h5ad"
    original.write_h5ad(out_path)
    print(f"Saved full AnnData with pseudotime to {out_path}")
    return original


# ---------------------------------------------------------------------------
# Validation plots
# ---------------------------------------------------------------------------

def plot_pseudotime_vs_day(adata: sc.AnnData, fig_dir) -> None:
    import matplotlib.pyplot as plt

    day_order = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
    groups = [adata.obs.loc[adata.obs["orig.ident"] == d, "pseudotime"].dropna() for d in day_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.violinplot(groups, positions=range(len(day_order)), showmedians=True)
    ax.set_xticks(range(len(day_order)))
    ax.set_xticklabels(day_order)
    ax.set_xlabel("Collection day")
    ax.set_ylabel("Rank-transformed pseudotime")
    ax.set_title("Pseudotime distribution by collection day")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = fig_dir / "pseudotime_vs_collection_day.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_markers_over_pseudotime(adata_traj: sc.AnnData, fig_dir) -> None:
    import matplotlib.pyplot as plt

    markers = {
        "Early":        ["POU5F1", "THY1"],
        "Intermediate": ["SOX2",   "PAX6"],
        "Late":         ["DLX5",   "TCF7L2"],
    }
    all_markers = [g for genes in markers.values() for g in genes]
    present = [g for g in all_markers if g in adata_traj.var_names]
    missing = set(all_markers) - set(present)
    if missing:
        warnings.warn(f"Markers not in var_names, skipping: {missing}")

    n = len(present)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    pt = adata_traj.obs["pseudotime"].values

    for i, gene in enumerate(present[:6]):
        idx = adata_traj.var_names.get_loc(gene)
        X = adata_traj.X if not hasattr(adata_traj.X, "toarray") else adata_traj.X.toarray()
        expr = X[:, idx]
        axes[i].scatter(pt, expr, s=2, alpha=0.3, rasterized=True)
        axes[i].set_xlabel("Pseudotime")
        axes[i].set_ylabel("Expression")
        axes[i].set_title(gene)
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, 6):
        axes[j].set_visible(False)

    fig.suptitle("Marker expression over pseudotime", y=1.01)
    fig.tight_layout()
    out = fig_dir / "marker_expression_over_pseudotime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_raw_vs_ranked(adata_traj: sc.AnnData, fig_dir) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    day_order = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
    colours   = cm.tab10(np.linspace(0, 0.6, len(day_order)))
    colour_map = dict(zip(day_order, colours))

    fig, ax = plt.subplots(figsize=(7, 6))
    for day in day_order:
        mask = adata_traj.obs["orig.ident"] == day
        ax.scatter(
            adata_traj.obs.loc[mask, "dpt_pseudotime"],
            adata_traj.obs.loc[mask, "pseudotime"],
            s=3, alpha=0.4, color=colour_map[day], label=day, rasterized=True,
        )
    ax.set_xlabel("Raw DPT pseudotime")
    ax.set_ylabel("Rank-transformed pseudotime")
    ax.set_title("Raw DPT vs rank-transformed pseudotime")
    ax.legend(fontsize=8, frameon=False, markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = fig_dir / "raw_vs_ranked_dpt.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CSS entry point for CSS-integrated DPT-pseudotime label generation
# — called from data_prep.py after generating  CSS embedding from R
# ---------------------------------------------------------------------------

def compute_dpt_from_css_embedding(
    adata: sc.AnnData,
    css_embedding_path,
    cell_type_key: str = "class3",
    n_neighbors: int = 20,
) -> pd.Series:
    """
    DPT pipeline using a CSS-PCA-CC embedding produced by css_pseudotime.R.

    Skips HVG selection, cell-cycle regression, scaling, and PCA — those are
    handled in R via simspec. Injects the CSS embedding directly as X_pca,
    then runs the existing build_graph → select_root → compute_dpt pipeline.

    Parameters
    ----------
    adata : AnnData
        Log-normalised AnnData (same cells as the CSS embedding).
    css_embedding_path : path-like
        Path to css_embedding.csv produced by css_pseudotime.R.
        Must have a 'cell_id' column and one column per CSS dimension.
    cell_type_key : str
        obs column for cell type labels (used to exclude proliferating cells).
    n_neighbors : int
        k for neighbour graph.

    Returns
    -------
    pd.Series
        Rank-transformed pseudotime ∈ (0, 1], indexed by cell barcode,
        for all cells (including post-hoc assigned proliferating cells).
        Named "pseudotime".
    """
    print(f"Loading CSS embedding from {css_embedding_path}...")
    css_df = pd.read_csv(css_embedding_path, index_col=0)
    n_dims = css_df.shape[1]
    print(f"CSS embedding: {css_df.shape[0]} cells × {n_dims} dimensions")

    # Align adata cell order to CSS embedding
    shared = adata.obs_names.intersection(css_df.index)
    if len(shared) < len(adata.obs_names):
        warnings.warn(
            f"{len(adata.obs_names) - len(shared)} cells in adata not found "
            f"in CSS embedding — subsetting to shared cells"
        )
    adata = adata[shared].copy()
    css_df = css_df.loc[shared]

    print(f"Input: {adata.n_obs} cells × {adata.n_vars} genes")
    adata = adata.copy()

    print("\n── Excluding proliferating cells ──")
    adata_traj, adata_prolif = exclude_proliferating(adata, cell_type_key)

    # Inject CSS embedding as X_pca so build_graph uses it for neighbours
    traj_css = css_df.loc[adata_traj.obs_names].values.astype(np.float32)
    adata_traj.obsm["X_pca"] = traj_css

    print(f"\n── Neighbour graph + diffusion map (CSS, k={n_neighbors}) ──")
    sc.pp.neighbors(adata_traj, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.diffmap(adata_traj, n_comps=15)
    print(f"Neighbours: k={n_neighbors}, use_rep=X_pca (CSS) | Diffusion map: 15 components")

    print("\n── Root cell selection ──")
    adata_traj = select_root(adata_traj)

    print("\n── DPT + rank transform ──")
    adata_traj = compute_dpt(adata_traj)

    print("\n── Post-hoc pseudotime for proliferating cells ──")
    prolif_css = css_df.loc[adata_prolif.obs_names].values.astype(np.float32) if adata_prolif.n_obs > 0 else None
    adata_prolif = assign_prolif_pseudotime(adata_traj, adata_prolif, prolif_embedding=prolif_css)

    print("\n── Saving full AnnData ──")
    merge_and_save(adata_traj, adata_prolif, adata)

    print("\n── Validation plots ──")
    fig_dir = Dirs.results / "figures" / "pseudotime_css"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full_pseudotime = pd.concat([
        adata_traj.obs[["pseudotime"]],
        adata_prolif.obs[["pseudotime"]],
    ])
    adata.obs["pseudotime"]     = full_pseudotime["pseudotime"]
    adata.obs["dpt_pseudotime"] = pd.concat([
        adata_traj.obs[["dpt_pseudotime"]],
        adata_prolif.obs[["dpt_pseudotime"]],
    ])["dpt_pseudotime"]

    plot_pseudotime_vs_day(adata, fig_dir)
    plot_markers_over_pseudotime(adata_traj, fig_dir)
    plot_raw_vs_ranked(adata_traj, fig_dir)

    print("\n── Per-day pseudotime summary ──")
    summary = adata.obs.groupby("orig.ident")["pseudotime"].agg(["mean", "std", "min", "max"])
    print(summary.to_string())

    pseudotime = adata.obs["pseudotime"].rename("rank-transformed-pseudotime")
    return pseudotime
