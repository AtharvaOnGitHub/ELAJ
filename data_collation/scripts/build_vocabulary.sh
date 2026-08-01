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
# EXCLUDE must match data.exclude_studies in configs/data/corpus.yaml.  A gene
# contributed only by an excluded study would sit in the vocabulary as DNS in
# every study that actually trains — a wasted embedding row and a wasted
# decoder output.

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
export EXCLUDE="wang_2022 zhang_2022"

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
    --go-groups 4 \
    --drop-unassigned \
    --out "${VOCAB_OUT}"

echo ""
echo "Done. Vocabulary written to ${VOCAB_OUT}"
