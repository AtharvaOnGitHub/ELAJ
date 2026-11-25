#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_bnmode
#SBATCH --output=logs/bnmode_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Is the study separation in the UMAPs a BatchNorm artefact?
#
#   sbatch data_collation/scripts/diagnose_bn_mode.sh configs/experiment/r3_nodab.yaml
#
# Second argument is an optional checkpoint path; the default 'auto' reads the
# run's metrics.csv, finds the epoch with the lowest val_nb, and loads the
# retained checkpoint for that epoch.
#
# Two hours and 180 GB: it builds the val Dataset the same way training does
# (every .h5ad opened, panels held sparse) and then makes two full passes over
# the split plus two UMAP fits. UMAP is single-threaded once seeded, which is
# the slow part.
#
# A GPU it barely uses, for the same reason train.sh needs one: the PyTorch
# module lives in the GPU nodes' software stack and loading it on bbdefault
# fails with "the following module(s) are unknown".
#
# Same module list as train.sh, minus R, which nothing here touches.

set -euo pipefail

CONFIG="${1:-configs/experiment/r3_nodab.yaml}"
CHECKPOINT="${2:-auto}"

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

nvidia-smi
echo ""
echo "Diagnosing ${CONFIG} (checkpoint: ${CHECKPOINT}) ..."

# scikit-learn for the nearest-neighbour search, umap-learn for the figures.
# Checked up front because both are optional dependencies of this project and
# a missing one should be a one-line message now rather than a traceback after
# the val split has been rebuilt.
python -u -c "import sklearn, umap; print('sklearn', sklearn.__version__, '| umap', umap.__version__)"

export PYTHONUNBUFFERED=1
python -u -m scripts.diagnose_bn_mode \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --root .
