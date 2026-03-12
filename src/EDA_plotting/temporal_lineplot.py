import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spatialmt.config import Paths, Dirs, setup_output_dirs, validate_raw_inputs

if __name__ == '__main__':
    setup_output_dirs()
    validate_raw_inputs()

# load data
processed_tpm = pd.read_csv(Paths.processed_tpm, header=0, index_col=0)

# subset data
mean_sd_cols = [c for c in processed_tpm.columns 
               if not c.startswith('H9')]
expr = processed_tpm[mean_sd_cols]

# all the same as original Lancaster paper (missing TBR2 in data)
genes_of_interest = ['GATA3','CDH2','CDH1','VIM','TBR1'] 
# fibronectin, junction protein TJP1, 2x non-muscle myo2 gene, neurodev spectrin, actin gene
structural_genes_of_interest = ['FN1','TJP1','MYH9','MYH10','SPTAN1','ACTG1']
all_genes = genes_of_interest + structural_genes_of_interest

# subset rows 
subset_expr = expr[expr['hgnc_symbol'].isin(all_genes)]

# time points in order
timepoints = ['d5', 'd7', 'd8', 'd10', 'd15', 'd20', 'd30']
mean_cols = [f'meanH9_{t}' for t in timepoints]
sd_cols   = [f'sdH9_{t}'   for t in timepoints]
x = np.arange(len(timepoints))

# line plotting - UPDATED to 2 rows x 6 columns
fig, axes = plt.subplots(2, 6, figsize=(24, 10))

# Group the lists by their target row index
gene_rows = [
    (0, genes_of_interest),              # Top row (index 0) gets 5 genes
    (1, structural_genes_of_interest)    # Bottom row (index 1) gets 6 genes
]

for row_idx, gene_list in gene_rows:
    for col_idx, gene in enumerate(gene_list):
        ax = axes[row_idx, col_idx]
        
        # Extract data for the specific gene
        gene_data = subset_expr[subset_expr['hgnc_symbol'] == gene]
        
        # Plot if the gene exists in the subset
        if not gene_data.empty:
            row = gene_data.iloc[0]
            means = row[mean_cols].values.astype(float)
            sds   = row[sd_cols].values.astype(float)
            
            ax.errorbar(x, means, yerr=sds, fmt='-o', capsize=4, linewidth=2, markersize=5)
            ax.set_title(gene, fontsize=15, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(timepoints)
            ax.tick_params(axis='both', labelsize=15)
            ax.set_xlabel('Day', fontsize=17, fontweight='bold')
            ax.set_ylabel('Mean TPM', fontsize=17, fontweight='bold')
            ax.grid(True, alpha=0.3)

# Hide the empty 6th subplot in the first row (row 0, column 5)
axes[0, 5].set_visible(False)

plt.tight_layout()
plt.show()