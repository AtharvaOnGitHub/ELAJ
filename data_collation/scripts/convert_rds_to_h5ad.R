#!/usr/bin/env Rscript
# Export raw counts from a Seurat .rds file as binary CSC components.
# Python reassembles these into an AnnData and writes the final .h5ad,
# avoiding any HDF5 compilation dependency on the R side.
#
# Writes to <output_dir>/:
#   values.bin        float32 non-zero values  (CSC order: genes × cells)
#   row_indices.bin   int32 row indices
#   col_pointers.bin  int32 column pointers
#   shape.txt         "<n_genes> <n_cells>"
#   features.tsv      gene names, one per line
#   barcodes.tsv      cell barcodes, one per line
#   obs.csv           cell metadata (row names = barcodes)
#
# Usage:
#   Rscript scripts/convert_rds_to_h5ad.R <input.rds> <output_dir>
#
# Required R packages:
#   install.packages("Seurat")   # Matrix is a Seurat dependency

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: convert_rds_to_h5ad.R <input.rds> <output_dir>")
}

rds_path   <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

message("Loading Seurat object from: ", rds_path)
suppressPackageStartupMessages(library(Seurat))

obj <- readRDS(rds_path)

raw_counts <- GetAssayData(obj, assay = "RNA", layer = "counts")
message("Raw counts matrix: ", nrow(raw_counts), " genes × ", ncol(raw_counts), " cells")

counts_csc <- as(raw_counts, "CsparseMatrix")

writeBin(as.double(counts_csc@x), file.path(output_dir, "values.bin"),      size = 4L)
writeBin(as.integer(counts_csc@i), file.path(output_dir, "row_indices.bin"), size = 4L)
writeBin(as.integer(counts_csc@p), file.path(output_dir, "col_pointers.bin"), size = 4L)
writeLines(paste(nrow(counts_csc), ncol(counts_csc)), file.path(output_dir, "shape.txt"))
writeLines(rownames(counts_csc), file.path(output_dir, "features.tsv"))
writeLines(colnames(counts_csc), file.path(output_dir, "barcodes.tsv"))
write.csv(obj@meta.data, file.path(output_dir, "obs.csv"))

message("Written to: ", output_dir)
