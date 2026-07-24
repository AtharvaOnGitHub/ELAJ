# Vocabulary Scope: Handling Genes Without Pathway Annotation

**Status:** Open — decision required before first training run
**Date:** 2026-07-24
**Context:** ELAJ foundation model, 13-study endometriosis scRNA-seq corpus

---

## 1. The observation

Building `vocabulary.json` from the union of genes across all 13 processed
studies yields **31,243 genes**. After assigning genes to biological groups
using MSigDB Hallmark (48 sets) and Reactome (1,818 pathways):

| | Genes |
|---|---|
| Assigned to Hallmark groups | 4,233 |
| Assigned to Reactome groups | 6,876 |
| **Unassigned** | **20,134 (64%)** |

The 20,134 unassigned genes are currently placed into 40 arbitrary
500-gene chunks (`unassigned_000` … `unassigned_039`).

## 2. Why this happens — it is a database limit, not a data problem

Coverage ceiling of every major pathway database, measured against the
full human gene space:

| Database | Unique genes | Cumulative |
|---|---|---|
| MSigDB Hallmark | 4,259 | 4,259 |
| + Reactome 2022 | 10,489 | 11,455 |
| + GO Biological Process 2023 | 14,698 | 16,287 |
| + KEGG 2021 | 8,078 | 16,796 |
| + WikiPathways 2023 | 8,044 | **17,083** |

**Maximum annotatable by any combination of databases: 17,083 genes.**
Corpus size: 31,243. Therefore **≥14,160 genes cannot be annotated by
construction.**

The current run captured 11,109 genes against a Hallmark+Reactome ceiling
of 11,455 — **97% of everything those two databases could reach.** The
assignment logic is working correctly.

### Composition of the corpus

scRNA-seq references (GENCODE via CellRanger) count every transcribed
feature, not only protein-coding genes. The 31,243-gene union is
approximately:

- ~20,000 protein-coding genes
- ~11,000 lncRNAs, antisense transcripts (`*-AS1`), pseudogenes, miRNAs,
  snoRNAs, and unnamed clone-based loci (e.g. `AC092155.1`, `LINC01234`)

Pathway databases annotate only well-characterised protein-coding genes.
Arithmetic check: ~20k protein-coding − ~17k annotated = ~3k unannotated
protein-coding, plus ~11k non-coding ≈ 14k unannotatable. This matches
the observed figure.

## 3. Why it matters for ELAJ specifically

ELAJ's `HierarchicalEncoder` performs **self-attention within each gene
group**. The architectural premise is that groups are biologically
coherent, so attention learns pathway-level co-expression structure.

Under the current vocabulary, 40 of 124 groups are alphabetically-adjacent
gene chunks sharing no biology. These consume a large share of the
gene-embedding parameters and group-attention compute to model structure
that does not exist. They also inflate the reconstruction target with a
large number of sparsely-expressed non-coding features.

Note also that groups must be size-capped regardless: self-attention is
O(n²), so the original single 27,010-gene `unassigned` group was
computationally infeasible (~30 GB attention matrix per layer at fp32).

## 4. Options

All options assume adding **GO Biological Process 2023** as a second
supplementary library, which is the single largest available coverage gain
(+4,832 genes over Hallmark+Reactome) at negligible cost.

| # | Option | Vocab size | Arbitrary groups | Trade-off |
|---|---|---|---|---|
| A | **Annotated only** — drop everything unassigned after GO BP | ~15,800 | 0 | Every group biologically real; matches the architecture's premise. Discards ~15k mostly non-coding features, some of which (lncRNAs) may carry endometriosis signal. |
| B | **Annotated + protein-coding** — keep unannotated protein-coding in chunks, drop non-coding | ~19,000 | ~6 | Retains all protein-coding signal. Requires a gene biotype lookup (Ensembl/HGNC) — one additional login-node fetch. |
| C | **Expression-filtered** — keep unannotated genes only if well-expressed/variable across the corpus | ~17–20,000 | ~4–8 | Most defensible scientifically: retains unannotated genes that demonstrably carry signal, drops dead weight. Requires a per-gene statistics pass over all 13 h5ad files (~20–30 min SLURM job) before rebuilding. |
| D | **Keep all** (current behaviour + GO BP) | 31,243 | ~31 | Nothing discarded. Model spends significant capacity on biologically meaningless attention groups. |

## 5. Points for discussion

1. **Is lncRNA signal important for the endometriosis question?** Several
   lncRNAs have reported roles in endometriosis. If they are central to the
   research question, Option A is too aggressive and C is preferable.

2. **Is the DNS sentinel affected?** No. DNS handles "gene is in the
   vocabulary but was not measured in this study's panel." Vocabulary scope
   is an independent decision — restricting scope does not change DNS
   semantics.

3. **Precedent.** Comparable single-cell foundation models generally
   restrict to protein-coding genes (Geneformer, ~20k) or to highly
   variable genes (scGPT). Retaining the full unfiltered reference is
   uncommon.

4. **Reversibility.** Changing vocabulary scope later requires a full
   retrain — the gene embedding tables and group structure are baked into
   the model. This is the reason to settle it now rather than after a
   first run.

## 6. Recommendation

**Option C** if a ~30 minute preprocessing job is acceptable — it is the
only option that decides inclusion on evidence from our own corpus rather
than on annotation status alone, and it directly addresses the concern in
point 1.

**Option A** if a decision is needed immediately — it is clean, defensible,
and consistent with the architecture, at the cost of discarding
unannotated genes without checking whether they carry signal.

Option D is not recommended: it preserves information but spends the
majority of the model's gene-level capacity on groups that cannot express
meaningful structure.

---

## Appendix — reproducing these figures

Coverage ceilings (any machine with internet access):

```bash
python src/scripts/build_vocabulary.py --save-gene-sets /tmp/gs.json \
    --supplementary-libraries Reactome_2022 GO_Biological_Process_2023
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
