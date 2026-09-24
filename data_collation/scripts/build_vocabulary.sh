#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=build_vocabulary
#SBATCH --output=logs/build_vocabulary_%j.log

# Builds data/processed/vocabulary.json from the union of all genes across
# the processed h5ad studies.  Using the union (not intersection) is correct
# because the model uses a DNS sentinel for genes not measured in a given
# study's panel.
#
# The excluded studies are READ FROM THE CORPUS CONFIG rather than typed here.
# They used to be typed, and the two drifted: the vocabulary in use was built
# with wang_2022 excluded while corpus_c1 trains on it, so the 1,791 genes
# only wang measures have no vocabulary row and are never read. Meanwhile
# garcia_alonso_2021_c was excluded from the corpus and not from the build.
# One source of truth removes the whole class of mistake.
#
# WHY EXCLUDE AT ALL. A gene contributed only by an excluded study sits in
# the vocabulary as DNS in every study that actually trains: a wasted
# embedding row, a wasted decoder output, and a dispersion parameter that
# never receives a gradient.
#
# NOTE the larger filter, which is NOT this one. --drop-unassigned below
# discards every gene that matched no curated pathway, which is half the
# transcriptome: 15,495 genes kept of 30,962 measured. That is deliberate,
# because the group encoder needs every gene to sit in a coherent group,
# but it means the vocabulary is the pathway-annotated transcriptome and
# the Methods section has to say so.

module purge
module load bear-apps/2024a
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

export PROCESSED=/rds/projects/a/acharjea-multi-omics/Atharva/data/processed
export GENE_LIST=${PROCESSED}/gene_list.txt
export VOCAB_OUT=${PROCESSED}/vocabulary.json
# Which corpus this vocabulary is for. Its exclude_studies is the one that
# counts, so point this at the config the runs actually load.
export CORPUS=${CORPUS:-/rds/homes/a/axw1008/axw1008/src/configs/data/corpus_c1.yaml}
export EXCLUDE=$(python - <<'PYEOF'
import os, yaml
cfg = yaml.safe_load(open(os.environ["CORPUS"])) or {}
print(" ".join(sorted(cfg.get("exclude_studies") or [])))
PYEOF
)
echo "Corpus ${CORPUS}"
echo "Excluding: ${EXCLUDE:-(nothing)}"

# ── Step 1: extract gene union from all processed h5ad files ──────────────
echo "Extracting gene union from processed studies ..."
python - <<'PYEOF'
import os
from pathlib import Path
import anndata as ad

processed = Path(os.environ["PROCESSED"])
gene_list_path = Path(os.environ["GENE_LIST"])

excluded = set(os.environ.get("EXCLUDE", "").split())

all_genes: set[str] = set()
for path in sorted(processed.glob("*.h5ad")):
    if path.stem in excluded:
        print(f"  {path.stem}: SKIPPED (in EXCLUDE)")
        continue
    adata = ad.read_h5ad(path, backed="r")
    genes = set(adata.var_names)
    print(f"  {path.stem}: {len(genes):,} genes")
    all_genes.update(genes)
    adata.file.close()

sorted_genes = sorted(all_genes)
gene_list_path.write_text("\n".join(sorted_genes) + "\n")
print(f"\nUnion: {len(sorted_genes):,} unique genes → {gene_list_path}")
PYEOF

# ── Step 2: assign genes to pathway groups → vocabulary.json ──────────────
# gene_sets.json is pre-downloaded on a login node (which has internet access)
# using:
#   python src/scripts/build_vocabulary.py --save-gene-sets src/data/gene_sets.json
# Compute nodes are firewalled, so we read from disk instead.
#
# Group COUNT is the dial on model size: the encoder builds one transformer per
# group, ~25k params at d_token=32.  --min-group-size is the control — it floors
# group size, so the group count falls out as D / (~2 x floor).  A ceiling
# (--cut tree) cannot do this: descending a node promotes all its children at
# once, so most groups land far below the bound.
#
# --max-group-size is a separate concern from model size.  The group encoder
# attends over a group's genes, so activation memory is quadratic in group size:
# GO:0008150 rolled up 3869 genes on the first build, which is 61 GB of
# attention at batch 256 for that one group.  The ceiling trims that tail.
# Keep it at >= 2x --min-group-size: the floor pass accumulates a merged group
# until it clears the floor, so merged groups can reach roughly twice it.
echo ""
echo "Building vocabulary by Reactome -> GO hierarchical roll-up ..."
cd /rds/homes/a/axw1008/axw1008/src
python scripts/build_vocabulary.py \
    --gene-list "${GENE_LIST}" \
    --gene-sets-file data/gene_sets.json \
    --hierarchy-dir data/hierarchy \
    --grouping hierarchical \
    --cut union \
    --min-group-size 250 \
    --max-group-size 500 \
    --go-groups 4 \
    --drop-unassigned \
    --out "${VOCAB_OUT}"

echo ""
echo "Done. Vocabulary written to ${VOCAB_OUT}"
