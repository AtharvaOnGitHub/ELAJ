#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# 71.2 GB measured on job 55687733 (one target, 14 min).
# 100G keeps it under the 128 GB that one of an a100_80 node's
# four GPUs is entitled to; 180G forced the scheduler to hold
# back more than a quarter of a node and the job would not place.
#SBATCH --mem=100G
#SBATCH --job-name=elaj_diag_gen
#SBATCH --output=logs/diag_gen_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Where does the val_nb gap come from? No training, one checkpoint.
#
#   sbatch data_collation/scripts/diagnose_generalisation.sh v1_bottleneck
#   sbatch data_collation/scripts/diagnose_generalisation.sh v1_pool
#   sbatch data_collation/scripts/diagnose_generalisation.sh v1_combined
#
# Optional second argument is a checkpoint path; the default is the run's best.
# Anything after that is passed through.
#
# WHAT IT ANSWERS. train_nb is computed in model.train() and val_nb in
# model.eval(), so the difference between them is not the generalisation gap.
# It mixes different cells, dropout on against off, and DSBN normalising each
# study by its own batch statistics against normalising every study by one
# corpus-wide running average. No run has ever scored TRAINING cells in eval
# mode, so nobody can say which of the three the v-series gap is -- and that
# gap spans 0.0264 to 0.0931 across five arms whose training fit spans 0.0057.
#
# It scores a 2x2 over {train cells, val cells} x {train mode, eval mode},
# which decomposes the gap exactly, then splits the mode switch again into its
# batch-norm and dropout halves.
#
# AND IT CAN ANSWER "NEITHER". The per-study table is there for that: a gap
# concentrated in two or three studies is about the data or the split and not
# about normalisation or the readout.
#
# COST. Eight batches per study per split, eight forward passes over them, no
# backward. Minutes. The two-hour wall is for the corpus load.

set -uo pipefail

RUN="${1:-}"
if [ -z "$RUN" ]; then
    echo "usage: sbatch $0 <experiment name> [checkpoint] [options]" >&2
    exit 2
fi
shift

CKPT="auto"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    CKPT="$1"
    shift
fi
EXTRA=("$@")

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load matplotlib/3.9.2-gfbf-2024a

export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

cd "$ELAJ_REPO/src"
export PYTHONUNBUFFERED=1

OUT="outputs/diagnostics/${RUN}"
echo "Generalisation decomposition: ${RUN}, checkpoint ${CKPT}"
python -u -m scripts.diagnose_generalisation \
    --run "${RUN}" --root . --checkpoint "${CKPT}" \
    --out "${OUT}" ${EXTRA[@]+"${EXTRA[@]}"}

echo "=== done; ${OUT}/generalisation.md holds the table"
