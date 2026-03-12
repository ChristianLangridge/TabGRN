from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from spatialmt.config import Paths, Dirs, setup_output_dirs, validate_raw_inputs

timepoints = ['d5', 'd7', 'd8', 'd10', 'd15', 'd20', 'd30']
replicates = ['0103', '1704', '2204']

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
expr.index = processed_tpm['hgnc_symbol'].values

# Log2(TPM + 1) transformation to avoid log(0) on zero-expressed genes
expr_log = np.log2(expr + 1)

# calculate variance for each gene across all 21 samples
gene_variance = expr_log.var(axis=1)

# get  indices of the top 3000 genes with the highest variance, ensuring no constant gene ever reaches z-score
top_3k_indices = gene_variance[gene_variance > 0].sort_values(ascending=False).head(3000).index

# filter the log-transformed data to only include these genes
expr_filtered = expr_log.loc[top_3k_indices]
print(f"Filtered to top 3000 most variable genes: {expr_filtered.shape}")

# redoing timepoint mean expression across replicates 
means = {}
for tp in timepoints:
    cols = [f'H9_{tp}_{rep}' for rep in replicates]
    means[tp] = expr_filtered[cols].mean(axis=1)

mean_expr = pd.DataFrame(means, index=expr_filtered.index)
print(f"Averaged time points with top 3K most variable genes: {mean_expr.shape}")

# row-wise across genes as per typical for heatmapping
z_matrix = mean_expr.apply(zscore, axis=1, result_type='broadcast')
print(f"Row-wise Z-scored mean expression time points with top 3K most variable genes: {mean_expr.shape}")

# heatmap plotting
g = sns.clustermap(
        z_matrix,
        method='ward',
        metric='euclidean',
        cmap='mako',
        center=0,
        vmin=-2.5, vmax=2.5,
        row_cluster=True,
        col_cluster=False, # preserve chronological timepoint order
        figsize=(12, max(12, len(z_matrix) * 0.003)),
        yticklabels=False, # change if you want specific genes labels to show up 
        xticklabels=True,
        linewidths=0.0,
        cbar_pos=(0.035, 0.40, 0.05, 0.25),
        cbar_kws={'label': 'Z-score (log₂ TPM+1)', 'shrink': 0.5, 'format': '%.1f'}
    )
 
 # Rotate x-axis labels for readability
g.ax_heatmap.set_xticklabels(
g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=15)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)

# increase size of label 
g.ax_cbar.set_ylabel('Z-score (log₂ TPM+1)', fontsize=15, fontweight = 'bold')
g.ax_cbar.tick_params(labelsize=10)
 
# including same labels to the genes of interest we care about retrospectively
# all the same as original Lancaster paper (missing TBR2 in data)
genes_of_interest = ['GATA3','CDH2','CDH1','VIM','TBR1'] 
# fibronectin, junction protein TJP1, 2x non-muscle myo2 gene, neurodev spectrin, actin gene
structural_genes_of_interest = ['FN1','TJP1','MYH9','MYH10','SPTAN1','ACTG1']
all_genes = genes_of_interest + structural_genes_of_interest

#for label in g.ax_heatmap.get_yticklabels():
    #if label.get_text() in all_genes:
        #label.set_visible(True)
        #label.set_fontsize(13) 
    #else:
        #label.set_visible(False)
        
plt.show()