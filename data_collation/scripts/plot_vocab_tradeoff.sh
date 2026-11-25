#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --job-name=plot_vocab_tradeoff
#SBATCH --output=logs/plot_vocab_tradeoff_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

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
source ${ELAJ_BASE}/envs/elaj/bin/activate

FIG_DIR=${ELAJ_BASE}/figures/vocab_tradeoff

cd "$ELAJ_REPO/src"
python scripts/plot_vocab_tradeoff.py \
    --gene-space-range 4000 20000 5 \
    --group-size-range 20 300 6 \
    --d-token-range 32 128 4 \
    --n-layers 2 \
    --out-dir "${FIG_DIR}"

echo ""
echo "Done. Figures written to ${FIG_DIR}"
