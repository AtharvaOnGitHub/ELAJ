#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=build_vocab_hier
#SBATCH --output=logs/build_vocab_hier_%j.log

# Builds vocabulary.json by HIERARCHICAL roll-up: Reactome pathways rolled up to
# a target group size, then GO Biological Process on whatever Reactome did not
# cover.  Groups are named biological categories, never arbitrary chunks.
#
# One-time login-node prerequisites (both need internet, compute nodes lack it):
#   1. Gene sets (if not already present):
#        python src/scripts/build_vocabulary.py --save-gene-sets src/data/gene_sets.json
#   2. Hierarchy structure files (Reactome relation + names, GO obo):
#        python src/scripts/build_vocabulary.py --save-hierarchy src/data/hierarchy

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

# ── Step 1: extract gene union from all processed h5ad files ──────────────
echo "Extracting gene union from processed studies ..."
python - <<'PYEOF'
import os
from pathlib import Path
import anndata as ad

processed = Path(os.environ["PROCESSED"])
gene_list_path = Path(os.environ["GENE_LIST"])

all_genes: set[str] = set()
for path in sorted(processed.glob("*.h5ad")):
    adata = ad.read_h5ad(path, backed="r")
    genes = set(adata.var_names)
    print(f"  {path.stem}: {len(genes):,} genes")
    all_genes.update(genes)
    adata.file.close()

sorted_genes = sorted(all_genes)
gene_list_path.write_text("\n".join(sorted_genes) + "\n")
print(f"\nUnion: {len(sorted_genes):,} unique genes → {gene_list_path}")
PYEOF

# ── Step 2: hierarchical roll-up → vocabulary.json ────────────────────────
# --target-group-size is the direct G knob (median group size ≈ this).
echo ""
echo "Building vocabulary by Reactome + GO hierarchical roll-up ..."
cd /rds/homes/a/axw1008/axw1008/src
python scripts/build_vocabulary.py \
    --grouping hierarchical \
    --gene-list "${GENE_LIST}" \
    --gene-sets-file data/gene_sets.json \
    --hierarchy-dir data/hierarchy \
    --target-group-size 500 \
    --out "${VOCAB_OUT}"

echo ""
echo "Done. Vocabulary written to ${VOCAB_OUT}"
