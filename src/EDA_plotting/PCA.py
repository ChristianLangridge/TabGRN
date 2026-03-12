from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from spatialmt.config import Paths, Dirs, setup_output_dirs, validate_raw_inputs

if __name__ == '__main__':
    setup_output_dirs()
    validate_raw_inputs()

processed_tpm = pd.read_csv(Paths.processed_tpm, header=0, index_col=0)
print(f"Processed TPM data: {processed_tpm.shape}")

# keeping all 21 raw sample columns (drop mean/sd and hgnc_symbol)
sample_cols = [c for c in processed_tpm.columns 
               if c not in ('hgnc_symbol',) 
               and not c.startswith('mean') 
               and not c.startswith('sd')]
expr = processed_tpm[sample_cols]

# transpose for PCA so samples as rows, genes as columns
expr_T = expr.T  # shape: (21, 16763)
print(f"Transposed TPM data: {expr_T.shape}")

# Log2(TPM + 1) transformation to avoid log(0) on zero-expressed genes
expr_log = np.log2(expr_T + 1)

# calculate variance for each gene across all 21 samples
# axis=0 because of transposition
gene_variance = expr_log.var(axis=0)

# get  indices of the top 3000 genes with the highest variance
top_3k_indices = gene_variance.sort_values(ascending=False).head(3000).index

# filter the log-transformed data to only include these genes
expr_filtered = expr_log[top_3k_indices]
print(f"Filtered to top 3000 most variable genes: {expr_filtered.shape}")

# scale across genes as per typical PCA (mean 0, std 1)
scaler = StandardScaler()
expr_scaled = scaler.fit_transform(expr_filtered)

# PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(expr_scaled)

# parse metadata from column names e.g. "H9_d5_0103"
days      = [int(re.search(r'd(\d+)', c).group(1)) for c in sample_cols]
replicates = [re.search(r'_(\d{4})$', c).group(1) for c in sample_cols]

# plot
colours  = {'0103': 'steelblue', '1704': 'tomato', '2204': 'seagreen'}
timepoints_ordered = sorted(set(days))

fig, ax = plt.subplots(figsize=(8, 6))

for rep, colour in colours.items():
    mask = [r == rep for r in replicates]
    rep_pcs  = pcs[mask]
    rep_days = [d for d, r in zip(days, replicates) if r == rep]
    
    # Sort by day so the line connects in order
    order = np.argsort(rep_days)
    rep_pcs  = rep_pcs[order]
    rep_days_sorted = [rep_days[i] for i in order]
    
    ax.plot(rep_pcs[:, 0], rep_pcs[:, 1], '-o', color=colour, label=f'Replicate {rep}', alpha=0.8)
    
    # Label each point with its day
    for (x, y), day in zip(rep_pcs, rep_days_sorted):
        ax.annotate(f'd{day}', (x, y), textcoords='offset points', xytext=(5, 5), fontsize=12)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=15)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()

print(f"Variance explained — PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")