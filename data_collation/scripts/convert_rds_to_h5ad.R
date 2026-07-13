#!/usr/bin/env Rscript
# Convert a Seurat .rds file to .h5ad using zellkonverter.
# Raw counts are extracted from the RNA assay's 'counts' slot.
#
# Usage:
#   Rscript scripts/convert_rds_to_h5ad.R <input.rds> <output.h5ad>
#
# Required R packages:
#   install.packages("Seurat")
#   BiocManager::install("zellkonverter")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: convert_rds_to_h5ad.R <input.rds> <output.h5ad>")
}

rds_path  <- args[1]
h5ad_path <- args[2]

message("Loading Seurat object from: ", rds_path)
suppressPackageStartupMessages({
  library(Seurat)
  library(zellkonverter)
  library(SingleCellExperiment)
})

obj <- readRDS(rds_path)

# Extract raw counts from the RNA assay
raw_counts <- GetAssayData(obj, assay = "RNA", slot = "counts")
message("Raw counts matrix: ", nrow(raw_counts), " genes × ", ncol(raw_counts), " cells")

# Build a minimal SingleCellExperiment with raw counts only
sce <- SingleCellExperiment(
  assays = list(counts = raw_counts),
  colData = obj@meta.data
)

message("Writing .h5ad to: ", h5ad_path)
writeH5AD(sce, file = h5ad_path, X_name = "counts")
message("Done.")
