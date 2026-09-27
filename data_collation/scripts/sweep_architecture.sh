#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=06:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_sweep_arch
#SBATCH --output=logs/sweep_arch_%j.log

# Where is the cell signal lost? Asked with random weights, before spending
# 30 h training an arm. Two families, one per card:
#
#   sbatch data_collation/scripts/sweep_architecture.sh aggregation
#   sbatch data_collation/scripts/sweep_architecture.sh group
#
# The optional second argument is the experiment the arms vary from. It
# defaults to v1_combined: the current architecture on the current vocabulary.
# Anything after it is passed through to the script.
#
#   sbatch data_collation/scripts/sweep_architecture.sh group v1_combined --cells 40000
#
# WHAT THIS ANSWERS. The token sweep settled how a gene becomes a token, and
# two measurements since say the constraint has moved to the funnel between
# the group encoders and the cell vector:
#
#   untrained, the token changes were worth +0.211 at the group outputs and
#   only +0.066 by the cell summary;
#   trained, the per-group PCA of the input reaches cross-donor cell-family
#   agreement of 0.944 where the latent keeps 0.9115.
#
# `aggregation` varies the consumer of those 396 group tokens: wider, deeper,
# more heads, or read with order statistics rather than one attention-weighted
# average. `group` varies what is handed over and how much of it: the head
# split, the depth, and an interface of 5, 9 or 17 tokens per group. Both open
# with `base`, so the two cards share a control and read as one table.
#
# WHAT IT CANNOT ANSWER. Anything whose effect is a loss term. DAB, lambda_CCE
# and beta change what training optimises rather than what the architecture can
# hold, so they are settled by training runs and not here.
#
# COST. Six arms over 20,000 cells. The token sweep did thirteen arms in 75
# minutes; grp_pool16 grows the aggregation sequence to 749 positions, so the
# wall is six hours with room to spare. 180G is ample: nothing here fits a
# baseline, so the corpus is never densified the way benchmark_baselines
# densifies it.

set -uo pipefail

FAMILY="${1:-}"
case "$FAMILY" in
    aggregation|group) ;;
    "") echo "usage: sbatch $0 <aggregation|group> [experiment] [options]" >&2
        exit 2 ;;
    *)  echo "unknown family '$FAMILY'; expected 'aggregation' or 'group'" >&2
        exit 2 ;;
esac
shift

RUN="v1_combined"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    RUN="$1"
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
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src
export PYTHONUNBUFFERED=1

OUT_SET=0
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    if [ "$a" = "--out" ]; then OUT_SET=1; fi
done
if [ "$OUT_SET" -eq 0 ]; then
    EXTRA+=(--out "outputs/sweeps/${RUN}_${FAMILY}")
fi

echo "Architecture sweep: family ${FAMILY}, varying from ${RUN}"
echo "Options: ${EXTRA[*]-(none)}"
python -u -m scripts.sweep_architecture --run "${RUN}" --root . \
    --family "${FAMILY}" ${EXTRA[@]+"${EXTRA[@]}"}
