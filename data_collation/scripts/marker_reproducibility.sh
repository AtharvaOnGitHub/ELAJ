#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=08:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --job-name=elaj_x13
#SBATCH --output=logs/x13_%j.log

# X13 — are per-family marker signatures reproducible across studies?
#
#   sbatch data_collation/scripts/marker_reproducibility.sh
#   sbatch data_collation/scripts/marker_reproducibility.sh --permutations 5
#
# Tests the marker half of Marecková's claim that prior endometrial atlases
# lack reproducible marker gene signatures. Uses each study's OWN annotation,
# never the transferred labels, so no signature is a reading of a prediction.
# The script's docstring carries the design and the gene-panel trap.
#
# NO GPU AND NO CHECKPOINT. It builds the corpus matrix and does sparse
# arithmetic; the model is never loaded. bbdefault keeps it off the two-GPU
# cap entirely.
#
# 240 G is benchmark_cpu.sh's figure and the same corpus build sits behind
# both. The counts are CSR over 416,256 cells, which is tens of GB rather
# than hundreds, but the dataset build itself is what peaks.
#
# 8 HOURS, and --permutations is the knob. Each permutation costs a full
# signature pass over every study. 20 is the default; if the log shows a pass
# taking more than ten minutes, resubmit with --permutations 5 and say so in
# the Note, because the null's precision is what that buys.

set -euo pipefail

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1
VENV=/rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj_cpu
[[ -f "${VENV}/bin/activate" ]] || {
    echo "no CPU venv at ${VENV}. Run install_benchmark_cpu.sh" >&2; exit 1; }
source "${VENV}/bin/activate"

cd /rds/homes/a/axw1008/axw1008/src

python -u - <<'PY'
import importlib.util, sys
missing = [p for p in ("numpy", "scipy", "anndata", "h5py", "yaml")
           if importlib.util.find_spec(p) is None]
if missing:
    sys.exit(f"MISSING from elaj_cpu: {', '.join(missing)}")
print("preflight ok")
PY

export PYTHONUNBUFFERED=1
python -u -m scripts.marker_reproducibility \
    --run p1_dab168_kl1 \
    --root . \
    --out outputs/analysis/x13_markers \
    "$@"
