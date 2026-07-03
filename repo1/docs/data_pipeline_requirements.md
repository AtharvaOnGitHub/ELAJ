# Data Pipeline Requirements for ELAJ

This document specifies every technical constraint that the data preprocessing pipeline must satisfy for the ELAJ dataloader to accept the data without modification. It is written as a standalone specification — the pipeline agent does not need to read any other part of the ELAJ codebase to fulfil these requirements.

---

## Outputs the pipeline must produce

| Output file | Location | Format |
|---|---|---|
| Per-study preprocessed data | `data/processed/<study_name>.h5ad` | AnnData HDF5 |
| Study manifest | `data/processed/studies.csv` | CSV |
| Gene vocabulary | `data/processed/vocabulary.json` | JSON (built by ELAJ, see §5) |

---

## 1. Counts matrix (`adata.X`)

- **Must be raw integer UMI counts.** Do not log-normalise, do not apply CPM/TPM, do not apply scran normalisation. The model performs library-size normalisation internally.
- Sparse matrices (`scipy.sparse.csr_matrix`, `csc_matrix`) are accepted and preferred for memory efficiency.
- Values must be ≥ 0. There is no upper bound.
- Do **not** round float arrays to integer — use the count matrix as output by your alignment tool (STARsolo, Cell Ranger, Kallisto). If a float representation is unavoidable (e.g., after ambient RNA correction returning fractional counts), rounding to integer is acceptable.
- **The ELAJ model internally computes `library_size = counts.clamp(min=0).sum()`** per cell after loading. Ensure your counts are the total per-cell UMI sum you want used for normalisation.

---

## 2. Gene names (`adata.var.index`)

- **Must use HGNC approved gene symbols** (e.g. `VEGFA`, `CD3E`, `EPCAM`). Do not use Ensembl IDs (`ENSG00000...`), RefSeq IDs, or gene aliases.
- Gene names must be unique within a study's var index.
- Gene names are matched case-sensitively against the vocabulary. `vegfa ≠ VEGFA`.
- Genes present in the study but absent from the vocabulary are **silently dropped** — this is safe and expected (the vocabulary covers MSigDB Hallmark genes, not all 30K+ human genes).
- Genes in the vocabulary but absent from this study are automatically assigned `DNS_SENTINEL = -1` (did-not-sequence). This is the correct representation for a gene not on the study's panel.

---

## 3. Cell-level metadata (`adata.obs`)

### 3a. Required columns

| Column | dtype | Description |
|---|---|---|
| `patient_id` | `str` | Unique identifier for the **biological donor**. Must be consistent across all cells from the same patient. Train/val/test splits are performed at the patient level — if this is wrong, you will have data leakage. |

`patient_id` values that appear in multiple studies are treated as the **same patient** — use the same identifier string if a patient appears in more than one study.

### 3b. Strongly recommended columns

| Column | dtype | Description |
|---|---|---|
| `disease_status` | `str` | Clinical status of the **tissue sample**. Valid values: `eutopic`, `ectopic`, `control`, `other` (case-insensitive). Defaults to `other` (code 3) if absent or unrecognised. |

Disease status valid values and their integer codes used by the model:

| String value | Integer code |
|---|---|
| `eutopic` | 0 |
| `ectopic` | 1 |
| `control` | 2 |
| `other` | 3 |

### 3c. Optional columns (default to 0 / NaN if absent)

| Column | dtype | Range | Description |
|---|---|---|---|
| `tissue_compartment` | `int` | 0–7 | Highest tissue hierarchy level (e.g. reproductive, immune). 0 = unknown. |
| `tissue_organ` | `int` | 0–15 | Specific organ (e.g. uterus, ovary, peritoneum). 0 = unknown. |
| `tissue_type` | `int` | 0–31 | Tissue sub-type (e.g. endometrium, endometrioma, DIE). 0 = unknown. |
| `tissue_microsite` | `int` | 0–15 | Microscopic location (e.g. surface epithelium, stroma, gland). 0 = unknown. |
| `age` | `float` | any | Patient age in years. Use `NaN` or leave column absent if unknown. |

**Encoding rules for tissue integers:**
- Use 0 for *unknown / not applicable* at any level. This is the safe default and does not introduce erroneous signal.
- Pick a consistent encoding **across all studies**. If study A encodes "uterus" as organ=1 and study B encodes it as organ=3, the model will learn contradictory representations. Decide once and apply everywhere.
- Never exceed the range upper bound listed above — values ≥ the bound will cause an embedding index-out-of-range error at training time.
- You do not need to use every integer in the range. Sparse categorical encodings (e.g., using only 0, 1, 2) are fine.

**Suggested tissue compartment encoding (adjust as needed):**

| Code | Compartment |
|---|---|
| 0 | unknown |
| 1 | reproductive |
| 2 | immune / haematopoietic |
| 3 | peritoneal / mesothelial |
| 4 | other |

---

## 4. Study manifest (`studies.csv`)

See [`docs/corpus_csv_spec.md`](corpus_csv_spec.md) for the full specification. Summary:

| Column | Type | Notes |
|---|---|---|
| `id` | int | 0-indexed, contiguous, unique per study. Determines `n_studies` in the model. |
| `name` | str | Short alphanumeric identifier (no spaces). |
| `path` | str | Absolute or root-relative path to the study's `.h5ad` file. |
| `tissue_types` | str | Pipe-separated list: e.g. `eutopic\|ectopic`. |
| `has_cycle_phase` | bool | `true`/`false`. |

**Critical:** study `id` values must be **contiguous starting from 0** with no gaps. The model uses them to index a learned per-study affine transform (DSBN). A gap (e.g., ids 0, 1, 3) causes an embedding out-of-range error.

---

## 5. Gene vocabulary (`vocabulary.json`)

The vocabulary is **not** produced by the preprocessing pipeline. It is built by the ELAJ script:

```bash
python scripts/build_vocabulary.py \
    --gene-list data/raw/gene_list.txt \   # optional: limit to your panel genes
    --out data/processed/vocabulary.json
```

The script downloads the 50 MSigDB Hallmark gene sets and assigns each gene to its highest-priority Hallmark group. If you want to restrict the vocabulary to genes measured in your studies, provide `--gene-list` with one HGNC symbol per line.

The pipeline should produce the `gene_list.txt` input (if desired). This is simply the union of all HGNC gene symbols that appear in `adata.var.index` across all studies.

```bash
# Example: collect union of gene names from all h5ad files
python - <<'EOF'
import anndata, pathlib, json

genes = set()
for p in pathlib.Path("data/processed").glob("*.h5ad"):
    adata = anndata.read_h5ad(p, backed="r")
    genes.update(adata.var_names.tolist())

pathlib.Path("data/raw/gene_list.txt").write_text("\n".join(sorted(genes)))
print(f"Wrote {len(genes)} genes")
EOF
```

---

## 6. Upstream QC the pipeline must perform

The ELAJ model makes no effort to clean noisy cells — all QC must be done before writing the `.h5ad` files.

### Mandatory
- **Doublet removal.** Run Scrublet, DoubletFinder, or equivalent. Remove predicted doublets (`predicted_doublet == True`) before saving.
- **Empty droplet filtering.** Remove barcodes that pass the knee of the UMI distribution but have no biological origin (EmptyDrops, knee-point filtering, or Cell Ranger's built-in filter is acceptable).
- **Minimum counts/genes filter.** Remove cells with total UMI < 200 or genes detected < 100 (adjust thresholds for your assay — 10x, Smart-seq2, etc. differ significantly).

### Strongly recommended
- **Ambient RNA correction** (SoupX, CellBender, or equivalent) if using 10x Chromium or similar droplet-based methods.
- **Mitochondrial fraction filter.** Remove cells with > 20–25% mitochondrial UMI (or > 35% for highly metabolically active cell types). Add `pct_counts_mt` to `adata.obs` for auditability.
- **Gene-level filter.** Remove genes detected in fewer than 3 cells (these add noise without signal).

### Not needed
- **Normalisation.** Do not normalise. The model uses raw counts.
- **Log-transformation.** Do not apply log1p or any other transformation.
- **Highly variable gene selection.** The vocabulary defines which genes are used; the model does not require pre-selection of HVGs.
- **Batch correction.** The model learns batch correction (DSBN + DAB). Do not apply Harmony, scVI, or combat before saving.
- **Dimensionality reduction.** UMAPs, PCA, etc. are not needed in the `.h5ad` files.

---

## 7. AnnData object structure summary

```python
import anndata as ad
import scipy.sparse

# Minimum valid AnnData for one study
adata = ad.AnnData(
    X=scipy.sparse.csr_matrix(counts_matrix),   # (n_cells, n_genes), raw int UMI
    obs=obs_dataframe,                           # must have: patient_id
                                                 # should have: disease_status
                                                 # optional: tissue_*, age
    var=pd.DataFrame(index=hgnc_gene_symbols),   # must be HGNC symbols
)
adata.write_h5ad("data/processed/study_name.h5ad")
```

---

## 8. Validation checklist before handing off

- [ ] `adata.X` contains raw integer UMI counts (not log1p, not normalised)
- [ ] `adata.var.index` contains HGNC symbols, no duplicates
- [ ] `adata.obs["patient_id"]` present and consistent — same donor = same string
- [ ] `adata.obs["disease_status"]` uses exactly `eutopic`, `ectopic`, `control`, or `other`
- [ ] Tissue integer codes are agreed upon and consistent across all studies
- [ ] All tissue integers are within the allowed ranges (compartment 0–7, organ 0–15, tissue_type 0–31, microsite 0–15)
- [ ] Doublet removal applied
- [ ] Ambient RNA correction applied (if droplet-based)
- [ ] `studies.csv` `id` column is 0-indexed and contiguous with no gaps
- [ ] `studies.csv` `path` values point to the correct `.h5ad` files
- [ ] `vocabulary.json` has been built (via `scripts/build_vocabulary.py`) and genes have been checked against the study panels
- [ ] At least 3 unique `patient_id` values per study (train/val/test each need ≥ 1 patient)
