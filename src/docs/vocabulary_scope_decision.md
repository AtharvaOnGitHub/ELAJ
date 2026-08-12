# Vocabulary Scope: Handling Genes Without Pathway Annotation

**Status:** Open — decision required before first training run
**Date:** 2026-07-24
**Context:** ELAJ foundation model, 13-study endometriosis scRNA-seq corpus

---

## 1. The observation

Building `vocabulary.json` from the union of genes across all 13 processed
studies yields **31,243 genes**. Using MSigDB Hallmark plus Reactome only,
20,134 genes (64%) received no pathway assignment and were placed into 40
arbitrary 500-gene chunks.

Two things were done in response, described below: **(a)** all major pathway
databases were added, and **(b)** two flaws in the grouping logic were fixed.
A scope decision on the genuinely unannotatable remainder is still open.

## 2. Why the gap exists — it is a database limit, not a data problem

Coverage ceiling of every candidate pathway library, measured against the full
human gene space. Marginal gain is greedy (best next library first):

| Library | Pathways | Genes | Marginal gain | Cumulative |
|---|---|---|---|---|
| MSigDB Hallmark | 48 | 4,259 | — | 4,259 |
| GO Biological Process 2023 | 5,407 | 14,698 | +10,823 | 15,082 |
| MSigDB Oncogenic Signatures | 189 | 11,250 | +2,636 | 17,718 |
| KEGG 2021 Human | 320 | 8,078 | +846 | 18,564 |
| GO Cellular Component 2023 | 474 | 10,972 | +419 | 18,983 |
| BioPlanet 2019 | 1,510 | 9,813 | +356 | 19,339 |
| GO Molecular Function 2023 | 1,147 | 12,126 | +226 | 19,565 |
| Reactome 2022 | 1,818 | 10,489 | +194 | 19,759 |
| Elsevier Pathway Collection | 1,721 | 5,718 | +160 | 19,919 |
| WikiPathways 2023 Human | 801 | 8,044 | +125 | **20,044** |
| ~~Panther 2016~~ | 112 | 2,041 | +30 | excluded — noise |
| ~~HumanCyc 2016~~ | 152 | 934 | +11 | excluded — noise |

**Maximum annotatable across all databases: ~20,085 genes** — effectively the
entire characterised protein-coding genome. Corpus size is 31,243, so
**~11,000 genes cannot be annotated by any database, ever.**

### Composition of the corpus

scRNA-seq references (GENCODE via CellRanger) count every transcribed feature,
not only protein-coding genes. The 31,243-gene union is approximately:

- ~20,000 protein-coding genes
- ~11,000 lncRNAs, antisense transcripts (`*-AS1`), pseudogenes, miRNAs,
  snoRNAs, and unnamed clone-based loci (e.g. `AC092155.1`, `LINC01234`)

The unassignable remainder matches the non-coding fraction almost exactly.
This is the key simplification: **after adding all databases, the open
question is no longer "why are genes unassigned" but simply "do we keep
non-coding features in the vocabulary?"**

## 3. Grouping-logic defects found and fixed

**(a) Generic-pathway bias.** Assignment was largest-pathway-first, so genes
landed in generic roots ("Immune System", "Metabolism") rather than specific
terms. Those roots then exceeded the size cap and were arbitrarily chunked —
producing groups like `reactome_immune_system_..._000/_001`. Now controlled by
`--pathway-preference` (default `specific`, i.e. smallest qualifying pathway
wins).

**(b) Fragmentation.** `--min-pathway-size` filtered on a pathway's *total*
corpus membership, but a group only receives genes no earlier pathway claimed.
Overlapping pathways left each other almost nothing, so groups qualified at 15
genes and then contributed 2 — median group size was **2 genes**, across 3,041
groups. Assignment is now greedy set-cover: a pathway is emitted only if it
contributes ≥ `min_pathway_size` *unclaimed* genes.

## 4. Group count is a first-class model-size cost

`HierarchicalEncoder` builds **one full 2-layer transformer per group**
(`nn.ModuleList` of `GroupEncoder`). At `d_token=64` that is **100,032
parameters per group**, before any other model component:

| Groups | Group-encoder parameters |
|---|---|
| 124 | 12.4M |
| 250 | 25.0M |
| 500 | 50.0M |
| 3,041 (pre-fix) | 304.2M |

`--min-pathway-size` is therefore the primary dial on model size. Measured
sweep against a corpus matched to ours (all databases, `specific` preference):

| min_pathway_size | Groups | Unassigned | Median group | Group-enc params |
|---|---|---|---|---|
| 15 | 644 | 12,364 | 20 | 64.4M |
| 30 | 371 | 13,250 | 39 | 37.1M |
| **50** | **249** | **14,195** | **67** | **24.9M** |
| 75 | 188 | 15,006 | 96 | 18.8M |
| 100 | 154 | 15,937 | 124 | 15.4M |
| 200 | 113 | 17,933 | 237 | 11.3M |

Note the tension: raising the threshold shrinks the model but pushes more
genes into arbitrary chunks, because pathways that cannot contribute enough
unclaimed genes are skipped entirely.

**Selected default: 50** — ~250 groups, ~25M group-encoder parameters, median
group of 67 genes (a biologically sensible pathway scale). Of the 14,195
unassigned at this setting, ~11,000 are the unannotatable non-coding fraction
and only ~3,000 are protein-coding genes that lost out to the threshold.

## 5. The remaining open question

With all databases in place and the logic fixed, the expected result is
roughly **~250 pathway groups covering ~17,000 genes**, plus **~14,000 genes
in ~28 arbitrary chunks** — of which ~11,000 are non-coding.

| # | Option | Vocab size | Arbitrary groups | Trade-off |
|---|---|---|---|---|
| A | **Annotated only** — drop all unassigned | ~17,000 | 0 | Every group biologically real; matches the architecture's premise. Discards the non-coding fraction, some of which (lncRNAs) may carry endometriosis signal. |
| B | **Annotated + protein-coding** — keep unannotated protein-coding, drop non-coding | ~20,000 | ~6 | Retains all protein-coding signal. Needs a gene-biotype lookup (Ensembl/HGNC) — one extra login-node fetch. |
| C | **Expression-filtered** — keep unannotated genes only if well-expressed/variable in our corpus | ~18–21,000 | ~8 | Most defensible scientifically: retains unannotated genes that demonstrably carry signal. Requires a per-gene statistics pass over all 13 h5ad files (~20–30 min SLURM job) before rebuilding. |
| D | **Keep all** (current) | 31,243 | ~28 | Nothing discarded. ~2.8M parameters and a meaningful share of attention compute spent on groups with no internal structure. |

## 6. Points for discussion

1. **Is lncRNA signal important for the endometriosis question?** Several
   lncRNAs have reported roles in endometriosis. If they are central, Option A
   is too aggressive and C is preferable.

2. **Does this affect the DNS sentinel?** No. DNS handles "gene is in the
   vocabulary but was not measured in this study's panel." Vocabulary scope is
   an independent decision and does not change DNS semantics.

3. **Precedent.** Comparable single-cell foundation models generally restrict
   to protein-coding genes (Geneformer, ~20k) or to highly variable genes
   (scGPT). Retaining a full unfiltered reference is uncommon.

4. **Reversibility.** Changing vocabulary scope later requires a full retrain —
   gene embedding tables and group structure are baked into the model. This is
   why it should be settled before the first run.

## 7. Recommendation

**Option C** if a ~30 minute preprocessing job is acceptable — it is the only
option that decides inclusion on evidence from our own corpus rather than on
annotation status alone, and it directly addresses point 1.

**Option A** if a decision is needed immediately — clean, defensible, and
consistent with the architecture, at the cost of discarding unannotated genes
without checking whether they carry signal.

Option D is not recommended: it preserves information but spends model capacity
on groups that cannot express meaningful structure.

---

## Appendix — reproducing these figures

Download all gene sets (any machine with internet access):

```bash
python src/scripts/build_vocabulary.py --save-gene-sets src/data/gene_sets.json
```

Corpus composition by gene-symbol class:

```bash
awk '{if($0~/^(AC|AL|AP|AF|BX|CR|CT|CU|FO|FP|Z|U)[0-9]{5,}\./)c["clone/unnamed"]++;
else if($0~/^LINC[0-9]/)c["lincRNA"]++;
else if($0~/-AS[0-9]*$/)c["antisense"]++;
else if($0~/^MIR[0-9]/)c["miRNA"]++;
else if($0~/^(SNOR|SNHG|SCARNA|RNU|RN7|RNA5)/)c["small RNA"]++;
else if($0~/P[0-9]+$/)c["pseudogene"]++;
else c["named gene"]++}
END{for(k in c)printf "%-20s %6d\n",k,c[k]}' \
  /rds/projects/a/acharjea-multi-omics/Atharva/data/processed/gene_list.txt \
  | sort -k2 -rn
```
