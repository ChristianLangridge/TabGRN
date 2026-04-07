"""
Diffusion pseudotime computation for brain organoid scRNA-seq.

Structured as a function module (mirrors prep.py) — called from data_prep.py.

Pipeline
--------
1.  Exclude "Unknown proliferating cells" (off-trajectory)
2.  HVG selection (seurat, 2000 genes)
3.  Regress out cell-cycle scores
4.  Scale
5.  PCA (10 components)
6.  Neighbours (k=20, 10 PCs)
7.  Diffusion map (15 components)
8.  Root cell: highest POU5F1 in HB4_D5 (fallback: random D5 cell)
9.  DPT
10. Rank-transform → pseudotime ∈ (0, 1]
11. Post-hoc pseudotime assignment for excluded proliferating cells (NN in PCA space)
12. Save full h5ad to model_data_anndata
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
# Step 2-3 — HVG selection + cell-cycle regression
# ---------------------------------------------------------------------------

def select_hvgs(adata: sc.AnnData, n_top_genes: int = 2000) -> sc.AnnData:
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"HVG selection: retained {adata.n_vars} genes")
    return adata


def regress_cell_cycle(adata: sc.AnnData) -> sc.AnnData:
    if "S.Score" in adata.obs.columns and "G2M.Score" in adata.obs.columns:
        print("Regressing out cell-cycle scores...")
        sc.pp.regress_out(adata, ["S.Score", "G2M.Score"])
    else:
        warnings.warn("S.Score / G2M.Score not found in obs — skipping cell-cycle regression")
    return adata


# ---------------------------------------------------------------------------
# Steps 4-8 — scale, PCA, neighbours, diffusion map, root selection
# ---------------------------------------------------------------------------

def run_pca(adata: sc.AnnData, n_comps: int = 10) -> sc.AnnData:
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    print(f"PCA: {n_comps} components")
    return adata


def build_graph(
    adata: sc.AnnData,
    n_neighbors: int = 20,
    n_pcs: int = 10,
    use_harmony: bool = False,
    batch_key: str = "orig.ident",
) -> sc.AnnData:
    """
    Build neighbour graph and diffusion map.

    Parameters
    ----------
    use_harmony : bool
        If True, run Harmony batch correction on the PCA embedding before
        building the neighbour graph. Requires harmonypy to be installed.
        Use this to attempt recovery of anomalous timepoint pseudotime
        (e.g. D21 batch effect). Compare outputs with and without to assess
        whether correction resolves the discontinuity.
    batch_key : str
        obs column used as the Harmony batch variable. Default "orig.ident".
    """
    if use_harmony:
        try:
            sc.external.pp.harmony_integrate(adata, key=batch_key)
            use_rep = "X_pca_harmony"
            print(f"Harmony integration on '{batch_key}' complete")
        except Exception as e:
            warnings.warn(f"Harmony failed ({e}) — falling back to uncorrected PCA")
            use_rep = "X_pca"
    else:
        use_rep = "X_pca"

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    sc.tl.diffmap(adata, n_comps=15)
    print(f"Neighbours: k={n_neighbors}, n_pcs={n_pcs}, use_rep={use_rep} | Diffusion map: 15 components")
    return adata


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
) -> sc.AnnData:
    """
    Assign pseudotime to excluded proliferating cells via nearest neighbour
    in PCA space (fitted on trajectory cells).
    """
    if adata_prolif.n_obs == 0:
        return adata_prolif

    print("Assigning pseudotime to proliferating cells via NN in PCA space...")

    # Re-use the same HVGs as the trajectory adata
    shared_genes = adata_traj.var_names.intersection(adata_prolif.var_names)
    traj_pca  = adata_traj.obsm["X_pca"]

    # Project prolif cells onto the same gene space, then scale using traj stats
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

    # Fit NN on trajectory PCA, approximate prolif PCA by projecting scaled data
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
    original.obs["pseudotime"]     = pseudotime["pseudotime"]
    original.obs["dpt_pseudotime"] = pseudotime["dpt_pseudotime"]

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


def plot_pseudotime_umap(adata_traj: sc.AnnData, fig_dir) -> None:
    import matplotlib.pyplot as plt

    sc.tl.umap(adata_traj)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(adata_traj, color="pseudotime", ax=ax, show=False,
               title="UMAP — rank-transformed pseudotime", color_map="viridis")
    fig.tight_layout()
    out = fig_dir / "pseudotime_umap.png"
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
# Top-level function — called from data_prep.py
# ---------------------------------------------------------------------------

def compute_diffusion_pseudotime(
    adata: sc.AnnData,
    cell_type_key: str = "class3",
    n_top_genes: int = 2000,
    n_pcs: int = 10,
    n_neighbors: int = 20,
    use_harmony: bool = False,
) -> pd.Series:
    """
    Full diffusion pseudotime pipeline on a log-normalised AnnData.

    The caller (data_prep.py) is responsible for loading the raw h5ad and
    applying normalize_total + log1p before passing adata in here.

    Parameters
    ----------
    adata : AnnData
        Log-normalised AnnData (normalize_total + log1p already applied).
    cell_type_key : str
        obs column for cell type labels.
    n_top_genes : int
        HVGs to retain.
    n_pcs : int
        PCA components for neighbour graph and DPT.
    n_neighbors : int
        k for neighbour graph.

    Returns
    -------
    pd.Series
        Rank-transformed pseudotime ∈ (0, 1], indexed by cell barcode,
        for all cells (including post-hoc assigned proliferating cells).
        Named "pseudotime".
    """
    print(f"Input: {adata.n_obs} cells × {adata.n_vars} genes")
    adata = adata.copy()

    print("\n── Excluding proliferating cells ──")
    adata_traj, adata_prolif = exclude_proliferating(adata, cell_type_key)

    print("\n── HVG selection ──")
    adata_traj = select_hvgs(adata_traj, n_top_genes)

    print("\n── Cell-cycle regression ──")
    adata_traj = regress_cell_cycle(adata_traj)

    print("\n── PCA ──")
    adata_traj = run_pca(adata_traj, n_pcs)

    print("\n── Neighbour graph + diffusion map ──")
    adata_traj = build_graph(adata_traj, n_neighbors, n_pcs, use_harmony=use_harmony)

    print("\n── Root cell selection ──")
    adata_traj = select_root(adata_traj)

    print("\n── DPT + rank transform ──")
    adata_traj = compute_dpt(adata_traj)

    print("\n── Post-hoc pseudotime for proliferating cells ──")
    adata_prolif = assign_prolif_pseudotime(adata_traj, adata_prolif)

    print("\n── Saving full AnnData ──")
    merge_and_save(adata_traj, adata_prolif, adata)

    print("\n── Validation plots ──")
    fig_dir = Dirs.results / "figures" / "pseudotime"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full_pseudotime = pd.concat([
        adata_traj.obs[["pseudotime"]],
        adata_prolif.obs[["pseudotime"]],
    ])
    adata.obs["pseudotime"] = full_pseudotime["pseudotime"]
    adata.obs["dpt_pseudotime"] = pd.concat([
        adata_traj.obs[["dpt_pseudotime"]],
        adata_prolif.obs[["dpt_pseudotime"]],
    ])["dpt_pseudotime"]

    plot_pseudotime_vs_day(adata, fig_dir)
    plot_pseudotime_umap(adata_traj, fig_dir)
    plot_markers_over_pseudotime(adata_traj, fig_dir)
    plot_raw_vs_ranked(adata_traj, fig_dir)

    print("\n── Per-day pseudotime summary ──")
    summary = adata.obs.groupby("orig.ident")["pseudotime"].agg(["mean", "std", "min", "max"])
    print(summary.to_string())

    pseudotime = adata.obs["pseudotime"].rename("pseudotime")
    return pseudotime
