#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=12:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_score
#SBATCH --output=logs/score_%j.log

# Scores one finished training arm every way the campaign needs, and puts
# its row in outputs/campaign.csv beside the reference baseline.
#
#   sbatch data_collation/scripts/score_arm.sh c_ref
#   sbatch data_collation/scripts/score_arm.sh c_wide --note "d_gauss 32"
#   sbatch data_collation/scripts/score_arm.sh r8_direction --runs-dir outputs/r8 \
#       --baselines outputs/benchmarks/r8_inductive/benchmark.csv
#
# Four steps, each writing its own directory, in this order:
#   1. validate_latent      -> <run dir>/validation           the thesis battery, figures
#   2. benchmark_baselines  -> outputs/benchmarks/<arm>       ELAJ beside the baselines
#   3. probe_stages         -> outputs/probes/<arm>_stages    where along the encoder
#   4. campaign_row         -> outputs/campaign.csv, .md      the one line that matters
#
# THE BASELINES ARE NOT RE-TRAINED PER ARM. Step 2 reads them from
# --baselines (default outputs/benchmarks/c_baselines/benchmark.csv, the
# control-included corpus fitted on the training donors) and runs only what
# that file lacks. Produce that file once per corpus with
#
#   sbatch data_collation/scripts/benchmark_baselines.sh c_ref --no-elaj \
#       --fit-split train --methods pca scvi scanvi \
#       --out outputs/benchmarks/c_baselines
#
# and every c_ arm reuses it. If the file is absent the baselines are
# trained here, which adds about an hour and produces the same rows.
#
# Flags: --runs-dir DIR      where the run directory lives (filed waves)
#        --baselines CSV     benchmark.csv to reuse (see above)
#        --campaign CSV      default outputs/campaign.csv
#        --note TEXT         free text for the campaign row
#        --groups            full probe, all 45 group encoders (adds ~4 h)
#        --skip-validate     omit step 1
#        --skip-probe        omit step 3
#
# WALLTIME: about 3 h with reused baselines and --no-groups; 12 h leaves
# room for a fresh scVI and the full probe.

set -euo pipefail

ARM=""
RUNS_DIR=""
BASELINES="outputs/benchmarks/c_baselines/benchmark.csv"
CAMPAIGN="outputs/campaign.csv"
NOTE=""
GROUPS=0
DO_VALIDATE=1
DO_PROBE=1
while [ $# -gt 0 ]; do
    case "$1" in
        --runs-dir)   RUNS_DIR="${2:?"$1 needs a value"}"; shift 2 ;;
        --baselines)  BASELINES="${2:?"$1 needs a value"}"; shift 2 ;;
        --campaign)   CAMPAIGN="${2:?"$1 needs a value"}"; shift 2 ;;
        --note)       NOTE="${2:?"$1 needs a value"}"; shift 2 ;;
        --groups)     GROUPS=1; shift ;;
        --skip-validate) DO_VALIDATE=0; shift ;;
        --skip-probe) DO_PROBE=0; shift ;;
        -*) echo "unknown flag $1" >&2; exit 2 ;;
        *)  if [ -n "$ARM" ]; then echo "one arm only; got $ARM and $1" >&2; exit 2; fi
            ARM="$1"; shift ;;
    esac
done
if [ -z "$ARM" ]; then
    echo "Give the experiment name, e.g. c_ref" >&2
    exit 2
fi

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
export PYTHONUNBUFFERED=1

RUNS_FLAG=()
RUN_DIR="outputs/${ARM}"
if [ -n "$RUNS_DIR" ]; then
    RUNS_FLAG=(--runs-dir "$RUNS_DIR")
    RUN_DIR="${RUNS_DIR}/${ARM}"
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "no run directory at src/${RUN_DIR}" >&2
    exit 1
fi

REUSE=()
if [ -f "$BASELINES" ]; then
    REUSE=(--reuse-baselines "$BASELINES")
    echo "Baselines reused from ${BASELINES}"
else
    echo "No baselines at ${BASELINES}; training them here"
fi

echo "== 1. validate_latent: ${ARM}"
if [ "$DO_VALIDATE" -eq 1 ]; then
    python -u -m scripts.validate_latent --run "$ARM" --root . \
        ${RUNS_FLAG[@]+"${RUNS_FLAG[@]}"} --splits val test \
        --out "${RUN_DIR}/validation"
else
    echo "   skipped"
fi

echo "== 2. benchmark_baselines: ${ARM}"
python -u -m scripts.benchmark_baselines --run "$ARM" --root . \
    ${RUNS_FLAG[@]+"${RUNS_FLAG[@]}"} --include-elaj --fit-split train \
    --methods pca scvi scanvi ${REUSE[@]+"${REUSE[@]}"} \
    --out "outputs/benchmarks/${ARM}"

echo "== 3. probe_stages: ${ARM}"
if [ "$DO_PROBE" -eq 1 ]; then
    PROBE_FLAGS=(--no-scib)
    if [ "$GROUPS" -eq 0 ]; then PROBE_FLAGS+=(--no-groups); fi
    python -u -m scripts.probe_stages --run "$ARM" --root . \
        ${RUNS_FLAG[@]+"${RUNS_FLAG[@]}"} "${PROBE_FLAGS[@]}" \
        --out "outputs/probes/${ARM}_stages"
else
    echo "   skipped"
fi

echo "== 4. campaign row: ${ARM}"
NOTE_FLAG=()
if [ -n "$NOTE" ]; then NOTE_FLAG=(--note "$NOTE"); fi
python -u -m scripts.campaign_row --arm "$ARM" \
    --benchmark "outputs/benchmarks/${ARM}/benchmark.csv" \
    --campaign "$CAMPAIGN" ${NOTE_FLAG[@]+"${NOTE_FLAG[@]}"}

echo "== done: ${ARM}"
