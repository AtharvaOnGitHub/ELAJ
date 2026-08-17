#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=04:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_purity
#SBATCH --output=logs/purity_%j.log

# kNN study purity for finished runs, from their best checkpoints.
#
#   sbatch data_collation/scripts/report_purity.sh                  # all of wave R4
#   sbatch data_collation/scripts/report_purity.sh r3_nodab r3_loose
#
# No training. Each run is embedded once through its best checkpoint and
# measured; the expensive part is rebuilding the validation Dataset, which runs
# sharing a corpus do once between them.
#
# Four hours for eight arms: six share the full corpus and build it once,
# r4_eutopic and r4_ectopic each need their own. A GPU because the encoder runs
# on it and because the PyTorch module lives in the GPU nodes' software stack.

set -euo pipefail

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
    ARMS=(r4_tiny_latent r4_thin_groups r4_deep_cell r4_tiny_dab_hi
          r4_eutopic r4_ectopic)
fi

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

# scikit-learn does the nearest-neighbour search. Checked before the split is
# rebuilt so a missing dependency costs a line now, not twenty minutes.
python -u -c "import sklearn; print('sklearn', sklearn.__version__)"

RUN_ARGS=()
for arm in "${ARMS[@]}"; do RUN_ARGS+=(--run "${arm}"); done

echo "Measuring: ${ARMS[*]}"
export PYTHONUNBUFFERED=1
python -u -m scripts.report_purity --root . "${RUN_ARGS[@]}"
