#!/usr/bin/env Rscript
# Convert a Seurat .rds file to .h5ad using anndataR.
# Raw counts are extracted from the RNA assay's 'counts' layer.
#
# Usage:
#   Rscript scripts/convert_rds_to_h5ad.R <input.rds> <output.h5ad>
#
# Required R packages:
#   install.packages("Seurat")
#   BiocManager::install("anndataR")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: convert_rds_to_h5ad.R <input.rds> <output.h5ad>")
}

rds_path  <- args[1]
h5ad_path <- args[2]

message("Loading Seurat object from: ", rds_path)
suppressPackageStartupMessages({
  library(Seurat)
  library(anndataR)
})

obj <- readRDS(rds_path)

raw_counts <- GetAssayData(obj, assay = "RNA", layer = "counts")
message("Raw counts matrix: ", nrow(raw_counts), " genes × ", ncol(raw_counts), " cells")

# anndataR expects cells × genes (obs × vars); Seurat stores genes × cells
adata <- AnnData(
  X   = Matrix::t(raw_counts),
  obs = obj@meta.data,
  var = data.frame(row.names = rownames(raw_counts))
)

message("Writing .h5ad to: ", h5ad_path)
write_h5ad(adata, h5ad_path)
message("Done.")
