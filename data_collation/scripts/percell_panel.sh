#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --job-name=elaj_percell
#SBATCH --output=logs/percell_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"
# Per-cell counts of a 16-gene panel for every cell, aligned to the dump
# (src/scripts/percell_panel.py). CPU only, no model. Same corpus build as
# X17 (job 62043453: 11 minutes, 240 G for the train split's peak).
#
#   sbatch data_collation/scripts/percell_panel.sh
set -euo pipefail
CELLS=${ELAJ_REPO}/src/outputs/cells/p1_dab168_kl1
[[ -f "${CELLS}/metadata.csv.gz" ]] || { echo "missing ${CELLS}/metadata.csv.gz" >&2; exit 1; }
module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }
export PYTHONNOUSERSITE=1
VENV=${ELAJ_BASE}/envs/elaj_cpu
[[ -f "${VENV}/bin/activate" ]] || { echo "no CPU venv at ${VENV}" >&2; exit 1; }
source "${VENV}/bin/activate"
cd "$ELAJ_REPO/src"
export PYTHONUNBUFFERED=1
python -u -m scripts.percell_panel \
    --run p1_dab168_kl1 --root . \
    --dump-meta outputs/cells/p1_dab168_kl1/metadata.csv.gz \
    --out outputs/analysis/percell_panel \
    "$@"
