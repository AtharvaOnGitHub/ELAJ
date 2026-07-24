#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=build_vocabulary
#SBATCH --output=logs/build_vocabulary_%j.log

# Builds data/processed/vocabulary.json from the union of all genes across
# the 13 processed h5ad studies.  Using the union (not intersection) is
# correct because the model uses a DNS sentinel for genes not measured in
# a given study's panel.

module purge
module load bear-apps/2024a
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

PROCESSED=/rds/projects/a/acharjea-multi-omics/Atharva/data/processed
GENE_LIST=${PROCESSED}/gene_list.txt
VOCAB_OUT=${PROCESSED}/vocabulary.json

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

# ── Step 2: assign genes to MSigDB Hallmark groups → vocabulary.json ──────
echo ""
echo "Building vocabulary from Hallmark gene sets ..."
cd /rds/homes/a/axw1008/axw1008/src
python scripts/build_vocabulary.py \
    --gene-list "${GENE_LIST}" \
    --out "${VOCAB_OUT}"

echo ""
echo "Done. Vocabulary written to ${VOCAB_OUT}"
