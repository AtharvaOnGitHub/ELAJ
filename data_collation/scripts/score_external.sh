#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
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
# A GPU IS REQUESTED AND BARELY USED, AND THAT IS NOT A MISTAKE. The scorer
# is a kNN search in sklearn, the permutation null and scIB on jax-CPU, so
# nothing here needs a card. But the PyTorch module the elaj venv was built
# against lives in the GPU nodes' software stack, and loading it on bbdefault
# fails with "the following module(s) are unknown" -- which is exactly what
# happened to the scgpt job on 16 October, and exactly what
# diagnose_bn_mode.sh's header already warned about. The computation does not
# need a card; the environment does.
#
# --gres=gpu:1 rather than gpu:a100_80:1, which is the part that was worth
# keeping: any card will do when none of them is used, and the pool of any
# card is far larger than the pool of a100_80s.
#
# ONE MODEL PER JOB, DELIBERATELY. On 15 October a single job scored
# geneformer, then died of OOM partway through scgpt's scIB, and the
# second model's four hours were lost. Separate jobs fail separately, and
# three of them finish in the wall time of the slowest rather than the sum.
#
# SIZED FROM WIDTH, WHICH IS WHAT ACTUALLY DRIVES THE COST. Every stage is a
# neighbour search over 416,256 cells, and that scales badly in the embedding
# dimension. Measured:
#
#   width   model          outcome at 8 h
#     16    the baselines  ~45 min, trivially inside
#    128    SCimilarity    finished comfortably
#    256    Geneformer     finished: kNN 53 min, x-donor 15, scIB 45
#    512    scGPT          TIMED OUT having completed scIB, minutes short
#   high    scFoundation   TIMED OUT still inside the kNN battery after 4 h
#
# So 8 hours covered everything up to 256 and nothing above it. 20 hours is
# sized for scFoundation, which is the widest thing we will score; drop it
# back with --time on the command line for a narrow model.
#
# 240G, not the 128G that died on 15 October in scGPT's scIB battery.
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
