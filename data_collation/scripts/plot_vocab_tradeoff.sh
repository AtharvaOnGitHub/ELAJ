#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=plot_vocab_tradeoff
#SBATCH --output=logs/plot_vocab_tradeoff_%j.log

# Renders the vocabulary trade-off figures (parameters vs coverage vs group
# size) from the real corpus gene list and the pre-downloaded gene sets.
# Uses the project visualisation theme (endo_model.viz).

module purge
module load bear-apps/2024a
module load matplotlib/3.9.2-gfbf-2024a
module load anndata/0.10.9-foss-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

PROCESSED=/rds/projects/a/acharjea-multi-omics/Atharva/data/processed
FIG_DIR=/rds/projects/a/acharjea-multi-omics/Atharva/figures/vocab_tradeoff

cd /rds/homes/a/axw1008/axw1008/src
python scripts/plot_vocab_tradeoff.py \
    --gene-list "${PROCESSED}/gene_list.txt" \
    --gene-sets-file data/gene_sets.json \
    --out-dir "${FIG_DIR}" \
    --max-group-size 500 \
    --d-token 64

echo ""
echo "Done. Figures written to ${FIG_DIR}"
