#!/bin/bash
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"
# Draws the donor-against-biology frontier from the benchmark CSVs.
#
#   bash data_collation/scripts/plot_frontier.sh
#   bash data_collation/scripts/plot_frontier.sh outputs/benchmarks/custom_dir
#
# NO sbatch, AND THAT IS DELIBERATE. This reads a handful of CSVs and writes
# three PNGs. It opens no .h5ad, loads no model and touches no GPU, so it is
# the same weight as check_donor_overlap.sh and belongs on a login node. Every
# other python entry point in this project goes through sbatch because it
# reads the corpus; this one does not.
#
# WHERE THE BENCHMARKS ARE. Under src/, because benchmark_baselines.sh cds to
# src/ before running and writes to outputs/benchmarks relative to there. The
# default below is right; pass a path only for a directory somewhere else.
#
# Folders named in plot_frontier.py's SOURCES that are absent are reported and
# skipped, so this runs cleanly before the foundation models have scored.

set -euo pipefail

BENCHMARKS="${1:-outputs/benchmarks}"
OUT="${2:-outputs/figures/frontier}"

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load Pillow/10.4.0-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a

export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

cd "$ELAJ_REPO/src"

python -u -m scripts.plot_frontier --benchmarks "${BENCHMARKS}" --out "${OUT}"

echo ""
echo "Figures in src/${OUT}:"
ls -la "${OUT}" 2>/dev/null || true
