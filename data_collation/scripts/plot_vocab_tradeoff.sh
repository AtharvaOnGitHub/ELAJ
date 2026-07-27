#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --job-name=plot_vocab_tradeoff
#SBATCH --output=logs/plot_vocab_tradeoff_%j.log

# Renders the analytic parameter-space figures: total model parameters against
# gene-space dimension (D), group size (G) and token dimension (d), each with
# the other two shown as a coloured family across facet panels.  Purely
# analytic — needs no corpus data, only matplotlib + numpy.
#
# Adjust the three ranges (MIN MAX COUNT) to explore different regions.

module purge
module load bear-apps/2024a
module load matplotlib/3.9.2-gfbf-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

FIG_DIR=/rds/projects/a/acharjea-multi-omics/Atharva/figures/vocab_tradeoff

cd /rds/homes/a/axw1008/axw1008/src
python scripts/plot_vocab_tradeoff.py \
    --gene-space-range 4000 20000 5 \
    --group-size-range 20 300 6 \
    --d-token-range 32 128 4 \
    --n-layers 2 \
    --out-dir "${FIG_DIR}"

echo ""
echo "Done. Figures written to ${FIG_DIR}"
