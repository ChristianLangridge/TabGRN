import datadata_preparation.prep  
from spatialmt.config.paths import Dirs, setup_output_dirs, validate_raw_inputs

if __name__ == "__main__":
    setup_output_dirs()
    validate_raw_inputs()

    h5ad_path = Dirs.model_data / "dataset.h5ad"

    dataset = prepare_dataset(
        h5ad_path=h5ad_path,
        cell_type_key="cell_type",
        n_top_genes=2000,
    )

    X = dataset["X"]
    cell_labels = dataset["cell_labels"]
    gene_labels = dataset["gene_labels"]
    y = dataset["y"]

    print(f"Expression matrix : {X.shape}  (cells x HVGs)")
    print(f"Cell labels        : {len(cell_labels)}")
    print(f"Gene labels        : {len(gene_labels)}")
    print(f"Cell-type classes  : {y.nunique()}  →  {y.unique().tolist()}")

    out_dir = Dirs.model_data
    np.save(out_dir / "X_hvg.npy", X)
    pd.Series(cell_labels, name="cell_id").to_csv(out_dir / "cell_labels.csv", index=False)
    pd.Series(gene_labels, name="gene_id").to_csv(out_dir / "gene_labels.csv", index=False)
    y.reset_index(drop=True).to_csv(out_dir / "cell_type_labels.csv", index=False)

    print(f"\nSaved preprocessed data to {out_dir}")