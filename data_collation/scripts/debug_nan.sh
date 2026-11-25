#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --job-name=elaj_debug_nan
#SBATCH --output=logs/debug_nan_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Locates the module that first emits a non-finite value, across the whole
# corpus.
#
#   sbatch data_collation/scripts/debug_nan.sh
#   sbatch data_collation/scripts/debug_nan.sh --only-study duempelmann_2026
#
# Needs the full memory request because EndometriosisDataset reads every h5ad
# into memory during construction, before any subsampling applies — which is
# what killed this on a login node.  Any extra arguments are passed straight
# through to the script.

set -euo pipefail

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

# With no arguments: reproduce the exact batches the failing run saw
# (--match-training with no subsampling).  With arguments: use them verbatim, so
# a previously-working invocation can be replayed exactly rather than merged
# with defaults that cannot be switched off from the command line.
ARGS=(--config configs/experiment/first_run.yaml --root .)
if [ "$#" -gt 0 ]; then
    ARGS+=("$@")
else
    ARGS+=(--match-training --max-cells-per-study 0)
fi

echo "Running: python scripts/debug_nan.py ${ARGS[*]}"
# PYTHONUNBUFFERED: SLURM sends SIGKILL at the walltime limit, and a killed
# process never flushes its stdout buffer.  A job that hits its limit would
# otherwise lose every print() it made — including a fault report it had
# already produced — while stderr survived, which reads as "it hung".
export PYTHONUNBUFFERED=1
python -u scripts/debug_nan.py "${ARGS[@]}"
