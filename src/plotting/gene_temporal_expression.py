import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spatialmt.config import Paths, Dirs, setup_output_dirs, validate_raw_inputs

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

# 11 genes from subset_expr
genes = subset_expr['hgnc_symbol'].tolist()  

# line plotting 

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()  # easier to index

for i, (ax, gene) in enumerate(zip(axes, genes)):
    row = subset_expr[subset_expr['hgnc_symbol'] == gene].iloc[0]
    
    means = row[mean_cols].values.astype(float)
    sds   = row[sd_cols].values.astype(float)
    
    ax.errorbar(x, means, yerr=sds, fmt='-o', capsize=4, linewidth=2, markersize=5)
    ax.set_title(gene, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints)
    ax.set_xlabel('Day')
    ax.set_ylabel('Mean TPM')
    ax.grid(True, alpha=0.3)

# Hide the empty 12th subplot (4x3 grid = 12 slots, you have 11 genes)
axes[-1].set_visible(False)

plt.suptitle('Structural Gene Panel Expression Over NE Organoid Developmental Time (H9 line)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()