#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --job-name=elaj_x17
#SBATCH --output=logs/x17_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# X17 — pseudobulk per donor and cell family, every gene, so the decoded
# disease genes (X12 step B, Figure 2e) can be checked against MEASURED
# counts. The script's docstring carries the design.
#
#   sbatch data_collation/scripts/disease_genes_expression.sh
#
# NEEDS ONE FILE COPIED UP FIRST. The transferred cell-family labels were
# made locally and exist nowhere on the cluster:
#
#   scp "<ELAJ>/Outputs/cells/p1_dab168_kl1/cell_family_transferred.csv.gz" \
#       <user>@bluebear.bham.ac.uk:${ELAJ_REPO}/src/outputs/cells/p1_dab168_kl1/
#
# NO GPU AND NO CHECKPOINT: corpus build plus one sparse product per split.
# 240 G because train alone is ~1.16M cells and X13 needed 58 GB for 416k;
# splits are built one at a time so that is the peak. X13 took 12 minutes
# for two splits, so 6 hours is generous for three.

set -euo pipefail

CELLS=${ELAJ_REPO}/src/outputs/cells/p1_dab168_kl1
for f in metadata.csv.gz cell_family_transferred.csv.gz; do
    [[ -f "${CELLS}/${f}" ]] || { echo "missing ${CELLS}/${f}; see the scp line at the top of this script" >&2; exit 1; }
done

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1
VENV=${ELAJ_BASE}/envs/elaj_cpu
[[ -f "${VENV}/bin/activate" ]] || {
    echo "no CPU venv at ${VENV}. Run install_benchmark_cpu.sh" >&2; exit 1; }
source "${VENV}/bin/activate"

cd "$ELAJ_REPO/src"
export PYTHONUNBUFFERED=1
python -u -m scripts.disease_genes_expression \
    --run p1_dab168_kl1 --root . \
    --transferred outputs/cells/p1_dab168_kl1/cell_family_transferred.csv.gz \
    --dump-meta outputs/cells/p1_dab168_kl1/metadata.csv.gz \
    --out outputs/analysis/x17_pseudobulk \
    "$@"
