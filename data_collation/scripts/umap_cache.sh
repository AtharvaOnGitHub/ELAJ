#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=elaj_umap
#SBATCH --array=0-8
#SBATCH --output=logs/umap_%A_%a.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Whole-corpus UMAP coordinates for the nine arms that do not have them.
#
#   sbatch data_collation/scripts/umap_cache.sh
#
# WHY THIS EXISTS. Every Supplementary figure that shows a latent needs 2-D
# coordinates, and `outputs/analysis/umaps/<arm>/_umap_cache/global_400000.npy`
# exists for exactly two arms: p1_dab168_kl1 and p1_dab168_kl30, the two
# reported ones. The other four frontier arms and all five ablation arms have
# none, so the ablation UMAPs the Discussion already points at cannot be drawn.
#
# WHY IT IS NOT ON THE LAPTOP, even though plot_cells_umap.py's own docstring
# says "CPU, laptop". One arm is one UMAP fit over 400,000 points plus a k=25
# neighbour search over the same, and `random_state` is set, which makes
# umap-learn single-threaded by design. Nine of those in series is most of a
# working day with the machine unusable for anything else. As an array they
# are one wall-clock task each, in parallel, on a QoS that does not touch the
# bbgpu budget.
#
# NO GPU AND NO MODEL. It reads `outputs/cells/<arm>/latent.npy` and
# `metadata.csv.gz`, which are already dumped. Nothing is re-embedded and no
# checkpoint is loaded, so this cannot change any number already reported.
#
# 64 G, NOT benchmark_cpu.sh's 240. The arithmetic: the latent is
# 1,577,728 x 16 float32 = 101 MB, the expanded metadata frame is a couple of
# GB, and the UMAP graph over 400,000 points is well under 1 GB. 240 G was
# sized for the wide-axis scoring pass that peaked at 201 GB; nothing here
# comes close.
#
# 3 HOURS. A single-threaded UMAP over 400,000 points is the slow part and
# runs in tens of minutes; three hours is the safety margin, not the estimate.
#
# IDEMPOTENT. embed() returns the cache when it already exists and the length
# matches, so re-running costs nothing and cannot overwrite a good cache with
# a different seed.

set -euo pipefail

ARMS=(
  p1_dab7_kl1
  p1_dab7_kl30
  p1_dab336_kl1
  p1_dab336_kl30
  a1_noecs
  a1_nodns
  a1_nophase
  a1_randgroups
  a1_clsonly
)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1
VENV=${ELAJ_BASE}/envs/elaj_cpu
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "no CPU venv at ${VENV}." >&2
    echo "Run: bash data_collation/scripts/install_benchmark_cpu.sh" >&2
    exit 1
fi
source "${VENV}/bin/activate"

cd "$ELAJ_REPO/src"

# PREFLIGHT. umap-learn is imported lazily inside embed(), so without this
# check a missing package would fail an hour in, after the metadata load.
python -u - <<'PY'
import importlib.util, sys
missing = [p for p in ("umap", "pandas", "numpy", "sklearn", "matplotlib")
           if importlib.util.find_spec(p) is None]
if missing:
    sys.exit(f"MISSING from elaj_cpu: {', '.join(missing)}. "
             "pip install umap-learn inside the venv.")
print("preflight ok")
PY

CELLS="outputs/cells/${ARM}"
[[ -d "${CELLS}" ]] || { echo "no dumped cells at ${CELLS}" >&2; exit 2; }

echo "UMAP cache for ${ARM}  (array task ${SLURM_ARRAY_TASK_ID:-0})"
export PYTHONUNBUFFERED=1
python -u -m scripts.plot_cells_umap \
    --cells "${CELLS}" \
    --out "outputs/analysis/umaps/${ARM}"
