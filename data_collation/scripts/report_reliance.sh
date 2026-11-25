#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --job-name=elaj_reliance
#SBATCH --output=logs/reliance_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# What is the decoder getting from the study embedding, and what from the
# latent? Runs on a SAVED checkpoint, so it answers from runs already paid
# for rather than needing a relaunch.
#
#   sbatch data_collation/scripts/report_reliance.sh m3_studydec --runs-dir outputs/m_series
#   sbatch data_collation/scripts/report_reliance.sh m3_study_raw --runs-dir outputs/m_series
#
# COST. Minutes. No training, no backward, sixteen batches by default. The
# hour is for the corpus load, which is the whole job.

set -uo pipefail

RUN="${1:-}"
if [ -z "$RUN" ]; then
    echo "usage: sbatch $0 <experiment name> [options]" >&2
    exit 2
fi
shift

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

python -u -m scripts.report_reliance --run "${RUN}" --root . \
    --out "outputs/m_series/${RUN}/reliance.json" "$@"
