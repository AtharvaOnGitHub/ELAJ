#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --job-name=elaj_embed_many
#SBATCH --output=logs/embed_many_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Embeds our val+test cells with SEVERAL pretrained models in one job.
#
#   sbatch data_collation/scripts/embed_many.sh scfoundation scimilarity
#   sbatch data_collation/scripts/embed_many.sh geneformer scgpt scimilarity
#
# WHY ONE JOB. One queue wait instead of several, which has been the slow
# part this week. Execution is sequential, so this is a trade: it wins when
# the queue is long and loses when cards are free, in which case submit
# embed_pretrained.sh per model instead and let them run concurrently.
#
# WHAT IS NOT SHARED, AND WHY NOT. Each model has its own virtualenv, because
# their dependency sets conflict and keeping them apart is the only reason
# the installs work at all. So each runs as its own process and each builds
# the corpus again, about seven minutes. Merging the venvs to save that is
# not worth the risk; caching the CorpusMatrix to disk would be, if this ever
# runs often enough to matter.
#
# ONE MODEL FAILING MUST NOT COST THE OTHERS. Each is run in a subshell whose
# failure is recorded and stepped over, and the exit status at the end says
# which succeeded. A job that dies on the first of three after queueing for
# hours is the worst outcome available.
#
# SIZED FOR THE LARGEST MEMBER. 240G and 10 hours covers scfoundation, which
# writes a 32 GB dense array and runs its own script over it; scimilarity is
# a forward pass and wants far less. Trim both if scfoundation is not in the
# list.

set -uo pipefail

MODELS_WANTED=("${@:-scimilarity}")
RUN="${RUN:-m6_drain}"

PROCESSED=${ELAJ_BASE}/data/processed
MODELS=${ELAJ_BASE}/models
ENVS=${ELAJ_BASE}/envs
SCRATCH=${ELAJ_BASE}/scratch

repo_for() {
    case "$1" in
        geneformer)   echo "${MODELS}/Geneformer"   ;;
        scgpt)        echo "${MODELS}/scGPT"        ;;
        scimilarity)  echo "${MODELS}/SCimilarity"  ;;
        scfoundation) echo "${MODELS}/scFoundation" ;;
        *) return 1 ;;
    esac
}

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found" >&2; exit 1; }
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""
echo "Models: ${MODELS_WANTED[*]}"
echo "Run:    ${RUN}"
echo ""

SUCCEEDED=()
FAILED=()

for MODEL in "${MODELS_WANTED[@]}"; do
    REPO="$(repo_for "${MODEL}")" || { echo "unknown model '${MODEL}'" >&2
                                       FAILED+=("${MODEL}"); continue; }
    VENV="${ENVS}/fm_${MODEL}"
    OUT="outputs/benchmarks/latents/${MODEL}"
    WORK="${SCRATCH}/${MODEL}_${SLURM_JOB_ID:-manual}"

    echo "════════════════════════════════════════════════════════════════"
    echo "── ${MODEL} ── $(date '+%H:%M:%S')"
    echo "   venv ${VENV}"
    echo "   out  ${OUT}"
    echo "════════════════════════════════════════════════════════════════"

    if [[ ! -d "${VENV}" ]]; then
        echo "no venv at ${VENV}; run install_foundation.sh ${MODEL}" >&2
        FAILED+=("${MODEL}")
        continue
    fi
    mkdir -p "${WORK}"

    # A SUBSHELL PER MODEL, so one venv's activation cannot leak into the
    # next and one model's failure cannot end the job.
    (
        set -e
        source "${VENV}/bin/activate"
        cd "$ELAJ_REPO/src"
        python -u -m scripts.embed_pretrained \
            --run "${RUN}" \
            --model "${MODEL}" \
            --repo "${REPO}" \
            --vocabulary "${PROCESSED}/vocabulary_axis.json" \
            --work "${WORK}" \
            --nproc 8 \
            --batch-size 64 \
            --root . \
            --out "${OUT}"
    )
    if [[ $? -eq 0 ]]; then
        echo "── ${MODEL} OK at $(date '+%H:%M:%S')"
        SUCCEEDED+=("${MODEL}")
    else
        echo "── ${MODEL} FAILED at $(date '+%H:%M:%S'); continuing" >&2
        FAILED+=("${MODEL}")
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "succeeded: ${SUCCEEDED[*]:-none}"
echo "failed:    ${FAILED[*]:-none}"
echo ""
echo "Scratch under ${SCRATCH}/*_${SLURM_JOB_ID:-manual} -- delete it; the"
echo "scfoundation one holds a 32 GB dense array."
if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    printf 'Score with: --external-latent'
    for M in "${SUCCEEDED[@]}"; do printf ' outputs/benchmarks/latents/%s' "$M"; done
    printf '\n'
fi
[[ ${#FAILED[@]} -eq 0 ]]
