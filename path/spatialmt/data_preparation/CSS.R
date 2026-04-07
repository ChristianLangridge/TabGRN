library(dplyr)
library(Timecourse)
library(simspec)
library(destiny)
library(ggplot2)
library(MetBrewer)
library(simspec)
library(tidyverse)
library(Seurat)
library(MetBrewer)

Timecourse <- readRDS("Timecourse.rds")









Timecourse <- cluster_sim_spectrum(Timecourse, label_tag = "dataset", cluster_resolution = 1) %>%
  run_PCA(reduction = "css", npcs = 10, reduction.name = "csspca", reduction.key = "CSSPCA_") %>%
  regress_out_from_embeddings(reduction = "csspca", vars_to_regress = c("G2M.Score","S.Score"), reduction.name = "csspcacc", reduction.key = "CSSPCACC_")

cor_css2cc <- cor(Embeddings(Timecourse,"csspcacc"), Timecourse@meta.data[,c("G2M.Score","S.Score")], method="spearman")
Timecourse <- RunUMAP(Timecourse, reduction = "csspcacc", dims = which(apply(abs(cor_css2cc), 1, max) < 0.2))
p1 <- UMAPPlot(Timecourse, group.by="class3.2") & NoAxes()
p2 <- UMAPPlot(Timecourse, group.by="dataset") & NoAxes()
p3 <- FeaturePlot(Timecourse, c("POU5F1","SOX2","NES","MKI67"), cols = beach_colscheme(30), order=T) & NoAxes() & NoLegend()
((p1 / p2) | p3) + patchwork::plot_layout(widths = c(1,2))