#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=08:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --job-name=elaj_continuity
#SBATCH --output=logs/continuity_%j.log

# Does clustering destroy label information the neighbourhoods hold?
#
#   sbatch data_collation/scripts/continuity_test.sh p1_dab168_kl1 \
#       --latents outputs/latents --out outputs/continuity
#
# NO GPU: it reads latents already written to disk and never touches a model,
# so it runs on bbdefault, outside the bbgpu cap of cpu=36,gres/gpu=2, beside
# two training jobs rather than displacing one.
#
# ── IT NEEDS LATENTS, AND NOTHING WRITES THEM BY DEFAULT ──────────────────
# The benchmark computes every latent, scores it and discards it; only UMAP
# coordinates are ever persisted. So the pass that scores wave 3 must also be
# asked to keep them:
#
#   sbatch data_collation/scripts/benchmark_baselines.sh p1_dab168_kl1 \
#       --runs-dir outputs/p_series --fit-split train \
#       --checkpoint <the arm's best epoch> \
#       --reuse-baselines <the inductive baselines csv> \
#       --dump-latent outputs/latents \
#       --out outputs/benchmarks/p1_dab168_kl1
#
# --dump-latent is off by default and changes nothing about scoring. It writes
# one ExternalLatent directory per scored method, which is the same format
# embed_pretrained.py already produces, so the foundation-model latents on
# disk can be pointed at directly and need no GPU pass at all.
#
# THE COMPARISON THAT MATTERS is ELAJ against scVI: both at 16 dimensions,
# both fitted on our training donors, nearly identical kNN recovery of cell
# family (0.986 against 0.987) and six-fold apart on Leiden NMI (0.072 against
# 0.442). If the gap A - B is large for ours and small for scVI's, the
# information in our latent lives at a scale finer than clusters. If the gaps
# match, it does not, and the low NMI is a plain limitation.
#
# Reusing a baselines CSV means scVI is NOT refitted and so is NOT dumped.
# To get its latent the pass has to fit it: drop --reuse-baselines and pass
# --methods pca scvi, which costs the scVI fit (about 16 min measured) and
# the PCA (about 7). That is the one extra cost this test carries.

set -euo pipefail

ARM="${1:?usage: continuity_test.sh <arm> [--latents DIR] [--out DIR]}"
shift || true

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1
VENV=/rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj_cpu
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "no CPU venv at ${VENV}." >&2
    echo "Run: bash data_collation/scripts/install_benchmark_cpu.sh" >&2
    exit 1
fi
source "${VENV}/bin/activate"

cd /rds/homes/a/axw1008/axw1008/src

echo "Arm:     ${ARM}"
echo "Options: $*"
export PYTHONUNBUFFERED=1
python -u -m scripts.continuity_test --run "${ARM}" --root . "$@"
