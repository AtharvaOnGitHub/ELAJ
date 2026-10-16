#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=08:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --job-name=elaj_score_ext
#SBATCH --output=logs/score_ext_%j.log

# Scores ONE pretrained model's latent against the fixed baseline rows.
#
#   sbatch data_collation/scripts/score_external.sh scgpt
#   sbatch data_collation/scripts/score_external.sh scimilarity
#
# NO GPU, WHICH IS THE POINT. The scorer is a kNN search in sklearn, the
# permutation null, and scIB on jax-CPU. benchmark_baselines.sh asks for an
# a100_80 only because scVI and scANVI train on one, and --methods none
# trains nothing. Dropping the card moves this to bbdefault, where several
# can run at once however many GPUs happen to be free.
#
# ONE MODEL PER JOB, DELIBERATELY. On 15 October a single job scored
# geneformer, then died of OOM partway through scgpt's scIB, and the
# second model's four hours were lost. Separate jobs fail separately, and
# three of them finish in the wall time of the slowest rather than the sum.
#
# 240G, NOT THE 128G THAT DIED. Cost scales with the embedding WIDTH, and
# these are wide: 16 dimensions for the fitted baselines against 256 for
# geneformer and 512 for scgpt. Measured on 15 October: geneformer's kNN
# battery 53 min, cross-donor 15 min, scIB 45 min; scgpt the same stages at
# 10, 25 and more than 90 before it was killed. Eight hours covers the
# widest.
#
# The reuse file supplies pca, scvi and scanvi, so nothing is refitted and
# the 1.16M-cell training corpus is never built.

set -euo pipefail

MODEL="${1:?usage: score_external.sh <model name, matching outputs/benchmarks/latents/<name>>}"
RUN="${2:-m6_drain}"
REUSE="${REUSE:-outputs/benchmarks/axis_m6_drain/benchmark.csv}"
LATENT="outputs/benchmarks/latents/${MODEL}"
OUT="outputs/benchmarks/fm_${MODEL}"

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

if [[ ! -f "${LATENT}/manifest.json" ]]; then
    echo "no external latent at ${LATENT}; run embed_many.sh ${MODEL} first" >&2
    exit 1
fi

echo "Model:   ${MODEL}"
echo "Latent:  ${LATENT}"
echo "Reusing: ${REUSE}"
echo "Out:     ${OUT}"
echo ""

export PYTHONUNBUFFERED=1
python -u -m scripts.benchmark_baselines \
    --run "${RUN}" \
    --root . \
    --methods none \
    --reuse-baselines "${REUSE}" \
    --external-latent "${LATENT}" \
    --out "${OUT}"

echo ""
echo "Done. ${OUT}/benchmark.md holds ${MODEL} beside the reused baselines."
