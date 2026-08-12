# 03 — Flow: data collation

**Entry:** [`data_collation/scripts/run_pipeline.py`](../../data_collation/scripts/run_pipeline.py)
**In:** public archives (GEO, CELLxGENE, Figshare, Reproductive Atlas)
**Out:** `data/processed/<study>.h5ad`, `studies.csv`, `patient_id_map.csv`, `gene_mapping_issues.csv`

```bash
python scripts/run_pipeline.py --only duempelmann_2026 --skip-download --force
```

---

## 0. Setup — `run` [`:65`](../../data_collation/scripts/run_pipeline.py:65)

Before the loop, seven collaborators are constructed **once** and reused for
every study:

```python
registry     = PatientRegistry(cfg.paths.patient_id_map)
standardiser = GeneStandardiser(aliases, issues_csv, hgnc_table)
attacher     = MetadataAttacher(registry)
qc_filter    = QCFilter(cfg.qc)
extractor    = RawCountExtractor()
writer       = H5adWriter()
manifest     = ManifestWriter()
```

> **Decision:** `registry` and `standardiser` are hoisted out of the loop because
> both hold expensive state — the patient UUID map (which must be shared so the
> same donor gets the same id across studies in one run) and the HGNC table
> (45,021 symbols, 99,231 alias mappings, parsed once).

`sys.path.insert` at [`:24`](../../data_collation/scripts/run_pipeline.py:24) makes
`import src.*` work when the script is run directly — the same bootstrap idiom
used on the model side.

`--only` filters `datasets` and **exits 1 if nothing matched**, rather than
silently processing zero studies.

## The per-study loop — [`:90`](../../data_collation/scripts/run_pipeline.py:90)

Seven stages. Each takes an `AnnData` and returns one.

```
download → load → extract raw → standardise genes → attach metadata → QC → write
```

Two guards frame it:

- **Skip if output exists** unless `--force` ([`:94`](../../data_collation/scripts/run_pipeline.py:94)).
  Note the skipped study is still appended to `completed`, so the manifest keeps
  its row — skipping does not delete it from `studies.csv`.
- **Download and load failures `continue`, not raise** ([`:114`](../../data_collation/scripts/run_pipeline.py:114),
  [`:122`](../../data_collation/scripts/run_pipeline.py:122)). One broken study
  does not abort a 13-study run. ⚠ It also means a partial corpus looks like a
  successful run apart from two `ERROR` lines.

---

## 1. Download

Dispatched through `_DOWNLOADERS` on `dataset.source`. `DataSource.MANUAL` maps
to `None` — a no-op that logs "expecting pre-placed files".

All downloaders implement `AbstractDownloader.download(dataset, dest_dir)`
([`base.py:18`](../../data_collation/src/download/base.py:18)).

### `GEODownloader` — [`geo.py:32`](../../data_collation/src/download/geo.py:32)

Three fetches: the raw supplementary tar, any named supplementary file
(`dataset.supplementary_metadata`), and the **series matrix**
(`_download_series_matrix`, [`:77`](../../data_collation/src/download/geo.py:77)).
The series matrix is the sample-level metadata table that `MetadataAttacher`
later reads — its absence changes which resolution branch runs in §5.

`_http_get` [`:94`](../../data_collation/src/download/geo.py:94) and `_extract_tar`
[`:168`](../../data_collation/src/download/geo.py:168) are module-level helpers.

### The other three

- `CellxGeneDownloader` resolves a dataset UUID to an `.h5ad` URL via the
  CELLxGENE API, then downloads.
- `ReproductiveAtlasDownloader` — direct URL for E-MTAB accessions.
- `FigshareDownloader` — token-based share download.

---

## 2. Load

Dispatched through `_LOADERS` on `dataset.loader`. All implement
`AbstractLoader.load(dataset, raw_dir) -> AnnData`
([`base.py:22`](../../data_collation/src/load/base.py:22)).

**Contract, stated in `base.py`:** every loader must set `obs["source_sample_id"]`.
Everything in §5 keys off it — it is the join column between per-cell data and
sample-level metadata. Both `H5adLoader` and `RDSLoader` have an explicit
`_ensure_source_sample_id` for this reason.

### `MTXPerSampleLoader` — [`mtx_loader.py:56`](../../data_collation/src/load/mtx_loader.py:56)

By far the most involved, because GEO's 10x layouts are inconsistent. Its
helpers handle, in order:

| Helper | Problem it solves |
|---|---|
| `_resolve_sample_specs` [`:100`](../../data_collation/src/load/mtx_loader.py:100) | decide whether samples are in subdirs or flat |
| `_find_gsm_subdirs` [`:114`](../../data_collation/src/load/mtx_loader.py:114) | one dir per GSM |
| `_group_flat_files` [`:139`](../../data_collation/src/load/mtx_loader.py:139) | all files in one dir, prefixed by GSM |
| `_has_mtx_trio` [`:181`](../../data_collation/src/load/mtx_loader.py:181) | is this a valid 10x triple? |
| `_find_mtx_dir` [`:195`](../../data_collation/src/load/mtx_loader.py:195) | recurse for a nested trio |
| `_canonicalize_mtx_dir` [`:213`](../../data_collation/src/load/mtx_loader.py:213) | rename to `matrix.mtx` / `barcodes.tsv` / `features.tsv` |
| `_extract_filtered_mtx` [`:232`](../../data_collation/src/load/mtx_loader.py:232) | pull the filtered matrix out of a per-sample tarball |
| `_strip_gsm_prefix` [`:260`](../../data_collation/src/load/mtx_loader.py:260) | `GSM123_barcodes.tsv` → `barcodes.tsv` |

Each sample is read with `scanpy.read_10x_mtx`, tagged with its GSM as
`source_sample_id`, and the per-sample AnnDatas are concatenated.

### `RDSLoader` — [`rds_loader.py:34`](../../data_collation/src/load/rds_loader.py:34)

Seurat `.rds` cannot be read from Python, so this shells out to R.

1. If `<name>.h5ad` already exists beside the `.rds`, **skip the R step** and
   load it ([`:34`](../../data_collation/src/load/rds_loader.py:34)). Conversion
   of a 38.9 GB object is expensive; this makes re-runs cheap.
2. `_run_r` [`:72`](../../data_collation/src/load/rds_loader.py:72) invokes
   [`convert_rds_to_h5ad.R`](../../data_collation/scripts/convert_rds_to_h5ad.R),
   which writes **binary CSC components** to a temp dir rather than an HDF5 file:
   `values.bin`, `row_indices.bin`, `col_pointers.bin`, `shape.txt`,
   `features.tsv`, `barcodes.tsv`, `obs.csv`.
3. `_assemble_anndata` [`:88`](../../data_collation/src/load/rds_loader.py:88)
   reads those back into a `scipy.sparse` matrix and builds the AnnData.

> **Decision:** the R script writes raw binary rather than calling an R HDF5
> writer, so the R side needs only Seurat and Matrix — no `hdf5r`, no compilation
> toolchain. The cost is a bespoke on-disk format, confined to these two files.

The R script asks specifically for `GetAssayData(obj, assay="RNA", layer="counts")`
([`convert_rds_to_h5ad.R:35`](../../data_collation/scripts/convert_rds_to_h5ad.R:35))
and writes the **full `obj@meta.data`** ([`:46`](../../data_collation/scripts/convert_rds_to_h5ad.R:46)).
Writing all of meta.data is what made the duempelmann fix possible — the
`EndometriosisStatus` column survived into obs even though nothing was
configured to use it.

---

## 3. Extract raw counts — `RawCountExtractor.extract` [`raw_extractor.py:21`](../../data_collation/src/process/raw_extractor.py:21)

Many published `.h5ad` files put normalised values in `.X` and keep counts in
`.raw.X` or a layer. This finds them and swaps.

- `_x_is_raw` [`:49`](../../data_collation/src/process/raw_extractor.py:49) —
  samples up to 10,000 non-zero values and tests `>= 0 and == floor(x)`. If `.X`
  already looks like counts, return immediately.
- `_find_raw` [`:60`](../../data_collation/src/process/raw_extractor.py:60) —
  tries `.raw.X`, then layers named `counts`, `raw_counts`, `spliced`,
  `unspliced`, in that order, each gated by `_looks_like_raw`.
- On success, assigns `adata.X = matrix` **in place** and clears the source.
  The comment at [`:38`](../../data_collation/src/process/raw_extractor.py:38)
  explains why: `adata.copy()` would double peak memory on a 466k-cell object.
- On failure, logs a **warning and proceeds**.

⚠ That warning is load-bearing and easy to miss. duempelmann_2026 trips it: its
`.X` is 7.84 % non-integer, so `_x_is_raw` returns `False`, no layer looks
better, and the pipeline continues with data the model will treat as counts.
The values are the authors' own ambient/multi-mapping-corrected counts. See
[11](11-findings.md).

---

## 4. Standardise genes — `GeneStandardiser.standardise` [`gene_standardiser.py:42`](../../data_collation/src/process/gene_standardiser.py:42)

Maps `adata.var_names` to HGNC approved symbols. Four-tier resolution, per gene,
in `_build_mapping` [`:64`](../../data_collation/src/process/gene_standardiser.py:64):

| Tier | Source | Notes |
|---|---|---|
| 1 | `configs/gene_aliases.yaml` | manual override, highest priority; mapping to `None` **drops** the gene deliberately |
| 2 | `HGNCTable.lookup` | approved symbols pass through; aliases, previous symbols and unambiguous Ensembl IDs resolve |
| 3 | `mygene.info` API | batched — one `querymany` for all misses, not one call per gene |
| 4 | drop | appended to `gene_mapping_issues.csv` |

> **Decision:** tier 3 is batched into a single `querymany` call because a
> per-gene HTTP round-trip over ~9,500 misses would dominate the run. The 10
> `httpx` POSTs visible in the duempelmann log are `mygene`'s internal chunking
> of that one call.

The whole API step is wrapped in `try/except`
([`:122`](../../data_collation/src/process/gene_standardiser.py:122)) that logs
and returns `{}` — so on a firewalled compute node every tier-3 gene silently
becomes a tier-4 drop.

`_write_issues` [`:126`](../../data_collation/src/process/gene_standardiser.py:126)
**appends** and writes a header only if the file did not exist, so the CSV
accumulates across studies with a `study_name` column to separate them.

Finally the AnnData is subset to kept genes, renamed, and
`var_names_make_unique()` is called — two aliases can resolve to the same
approved symbol.

### `HGNCTable` — [`hgnc_table.py:17`](../../data_collation/src/process/hgnc_table.py:17)

Lazy-loading with a disk cache: `_ensure_loaded` → `_download` if absent →
`_parse`. `_parse` [`:70`](../../data_collation/src/process/hgnc_table.py:70)
builds the alias map and **excludes ambiguous conflicts** — one alias pointing at
two approved symbols is dropped rather than guessed (1,452 of them).

---

## 5. Attach metadata — `MetadataAttacher.attach` [`metadata_attacher.py:79`](../../data_collation/src/process/metadata_attacher.py:79)

Six steps in fixed order:

```python
sample_meta = self._load_sample_metadata(dataset, raw_dir)
self._set_patient_ids(adata, dataset, sample_meta)
self._set_disease_status(adata, dataset, sample_meta)
self._set_age(adata, sample_meta)
self._set_tissue_ontology(adata, dataset, sample_meta)
self._set_disease_ontology(adata, dataset, sample_meta)
adata = self._apply_cell_exclusions(adata, dataset)
```

`_load_sample_metadata` tries, in priority order: a supplementary metadata file,
the GEO series matrix (via `SeriesMatrixParser`), then filename-pattern inference
from `raw_dir`. It returns a DataFrame indexed by GSM, or `None`.

### `_set_patient_ids` — [`:65`](../../data_collation/src/process/metadata_attacher.py:65)

Maps each `source_sample_id` through `registry.get_or_create(study, sample)`.
One UUID per (study, sample) pair, persisted across runs.

> **Decision:** patient identity is a random UUID rather than the source sample
> id because sample ids collide across studies — three different studies all
> using "P1" would otherwise become one person, and the patient-held-out split
> would leak. (The preflight check verifies this: 302 ids for 302 patients.)

### `_set_disease_status` — [`:90`](../../data_collation/src/process/metadata_attacher.py:90)

Five branches, in order. **The order is the whole story here.**

```
1. obs already has disease_status  → normalise case, coerce invalid → "other", return
2. cfg set AND series matrix carries one of cfg's columns  → map per sample
3. cfg.column present in adata.obs                          → map per cell
4. cfg.default                                              → constant
5. len(tissue_types) == 1 → that value;  else "other" + WARNING
```

Branch 2's guard is `_resolvable_in(cfg, sample_meta.columns)`
([`:23`](../../data_collation/src/process/metadata_attacher.py:23)), which checks
whether the series matrix actually has `column`, `fallback_column`, or
`override_column`.

> That guard is the fix from `c45a6fc`, and the reason it is needed is worth
> understanding because the failure mode is invisible.
> `_resolve_mapped_values` iterates samples and falls back to `cfg.default` for
> each one that misses. It therefore returns a **fully populated map** even when
> the column does not exist — so `.map(per_sample)` succeeds, `.fillna()` finds
> nothing to fill, and the study gets a uniform wrong label with no warning at
> all. duempelmann_2026 lost all 466,371 labels to that, twice: once to branch 5
> (which at least warned), then once to branch 2 (which did not).

`_warn_if_column_missing` [`:38`](../../data_collation/src/process/metadata_attacher.py:38)
now fires when `cfg.column` is in neither source.

### `_set_tissue_ontology` / `_set_disease_ontology` — [`:154`](../../data_collation/src/process/metadata_attacher.py:154), [`:195`](../../data_collation/src/process/metadata_attacher.py:195)

Same shape, and both **preserve an existing column and return** if the loader
already provided one — that is how CELLxGENE `.h5ad` files keep their curated
ontology terms untouched.

Their fallbacks are chosen for this corpus specifically: `UBERON:0001295`
(endometrium) and `MONDO:0005133` (endometriosis). ⚠ The disease fallback is a
**silent wrong answer for control patients** — it is why duempelmann's 25
controls were labelled endometriosis with no warning. A `None` default would
have surfaced it.

### `_apply_cell_exclusions`

Drops cells by `exclude_tissue_ontology_terms` (e.g. organoids,
`UBERON:0000479`) and by `exclude_obs_column_values`.

---

## 6. QC — `QCFilter.filter` [`qc.py:30`](../../data_collation/src/process/qc.py:30)

Four steps, in order — and the order matters:

1. `_filter_genes` — `sc.pp.filter_genes(min_cells=...)`.
2. `_annotate_mt` [`:56`](../../data_collation/src/process/qc.py:56) — flags
   `MT-` prefixed genes (upper-cased first, so mouse `mt-` also matches) and runs
   `sc.pp.calculate_qc_metrics`, which **creates** `total_counts`,
   `n_genes_by_counts`, `pct_counts_mt`.
3. `_filter_cells` [`:67`](../../data_collation/src/process/qc.py:67) — a single
   boolean `keep` combining min UMI, min genes, max MT fraction.
4. `_remove_doublets` — Scrublet, only if `cfg.run_scrublet`.

Step 2 must precede step 3 because step 3 reads columns step 2 writes. Step 1
precedes both so gene-level metrics are computed on the retained gene set.

`_remove_doublets` [`:85`](../../data_collation/src/process/qc.py:85) has two
layers of tolerance: `ImportError` → warn and return unchanged; any other
exception → warn and return unchanged. Scrublet is currently not installed, so
every run logs `scrublet not installed — skipping doublet removal`. This is the
deferred item, not an accident.

`gc.collect()` after the cell subset ([`:76`](../../data_collation/src/process/qc.py:76))
is deliberate — `adata[keep].copy()` leaves the original alive until collected,
and at 466k cells that is a real amount of memory.

---

## 7. Write — `H5adWriter.write` [`h5ad_writer.py:27`](../../data_collation/src/write/h5ad_writer.py:27)

`_validate` [`:37`](../../data_collation/src/write/h5ad_writer.py:37) checks the
obs schema before writing, then `adata.write_h5ad`. The
`... storing 'x' as categorical` lines in the log are AnnData's own message.

---

## 8. Manifest — `ManifestWriter.write` [`manifest.py:23`](../../data_collation/src/write/manifest.py:23)

Run **once after the loop**, over `completed`.

Additive by design: it reads any existing `studies.csv` into a dict keyed by
name, overwrites entries for studies in this run, appends new ones, then
**reassigns contiguous ids in insertion order**.

> **Decision:** additive rather than destructive so `--only one_study` does not
> wipe the other twelve rows.

⚠ Two consequences worth holding in mind:

- Ids are positional, so **inserting a study renumbers everything after it**. A
  checkpoint trained before the insert has `DSBN.gamma` rows bound to the old
  numbering. Nothing detects this.
- A study removed from `datasets.yaml` keeps its `studies.csv` row forever —
  removal is only possible by editing the CSV or via `exclude_studies` on the
  model side.

---

## 9. Initialisation differences

The pipeline has essentially one flow, varied by flags:

| Invocation | What changes |
|---|---|
| default | all datasets; skip any with existing output |
| `--only A B` | dataset list filtered; exits 1 if empty |
| `--skip-download` | stage 1 skipped entirely; raw files assumed present |
| `--force` | the "output exists" guard is bypassed |
| `source: manual` | downloader is `None`; logs and proceeds to load |
| `.h5ad` already beside `.rds` | `RDSLoader` skips the R subprocess |
| no series matrix | `sample_meta is None` → §5 takes branch 3, 4 or 5 |
| CELLxGENE source | ontology columns already present → both ontology setters return early |
