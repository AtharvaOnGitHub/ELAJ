#!/bin/bash
#SBATCH --qos=bbgpu
# a100_80, not a100: the bare `a100` constraint resolves to a 40 GB card, and
# run 49817807 measured 18.3 GB in use at batch 128 — enough headroom to matter
# but not enough to raise the batch, which is the cheapest throughput win when
# the GPU is only 70% busy.
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=elaj_train
#SBATCH --output=logs/train_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Trains ELAJ from an experiment YAML.
#
#   sbatch data_collation/scripts/train.sh v1_combined
#   sbatch data_collation/scripts/train.sh configs/experiment/v1_combined.yaml
#   sbatch data_collation/scripts/train.sh src/configs/experiment/v1_combined.yaml
#
# All three work. The bare arm name is the short form; the two paths are
# what you get from tab completion at the repo root and from `cd src` -- this
# script cds to src itself and passes --root ., so a repo-root path used to
# fail with a FileNotFoundError naming a file that plainly exists, which is
# a wasted queue slot for a path prefix.
#
# Defaults to first_run.yaml, which is full_model with batch_size 128.
#
# CPU count feeds the DataLoader workers: StudyPanel keeps counts as CSR and
# densifies one cell per __getitem__, so collation is CPU-bound and starving it
# will idle the GPU.
#
# 16 CORES, NOT 8, MEASURED 6 OCTOBER 2026. The BlueBEAR GPU docs give an
# a100_80 node as 4 GPUs over 72 Ice Lake cores and 512 GB, with the rule
# "at most total_cores / total_GPUs per GPU" -- so one GPU's share is 18
# cores and 128 GB, and 8 cores was using 44% of the entitlement. 16 cores
# gives 15 DataLoader workers instead of 7 and measured 58 min/epoch against
# 62.5 on m5_combined: a 7% saving, not the 20% hoped for, so collation was
# less of the bottleneck than the comment above assumed. Kept because the
# cores are free within the share.
#
# 100 G, NOT 180 G. Four full-length runs peaked at 51.8 to 53.9 GB
# (m3_study_raw 17 epochs, m4_tight16 and m4_edge_starved 19 each, m5 at 2 h),
# so growth over a run is about 2 GB and 180 G was 3.3x the need. More to the
# point 180 G sits ABOVE the 128 GB per-GPU share, which stops the scheduler
# co-locating this job with three other single-GPU jobs -- a backfill blocker
# independent of priority. 100 G is 1.9x the measured peak and under the
# share. See the bluebear-gpu-node-specs note.
#
# WALLTIME: two days, which is the queue maximum. It was 24 h for most of this
# project's history, and that was a self-imposed limit rather than the cluster's
# — every arm of waves R7 and R8 was cancelled by it, and every one of them had
# its best epoch on the LAST epoch it completed, meaning all six were still
# improving when they died. At ~4400 s/epoch two days holds about 39 epochs.
#
# RESUMING. A run the wall killed continues from its newest checkpoint with
#
#   sbatch data_collation/scripts/train.sh configs/experiment/c_ref.yaml --resume
#
# which restores weights, optimizer, global step (so the KL and DAB
# schedules continue rather than restart), the cosine LR state, early
# stopping, and the metrics history. A specific checkpoint:
#   --resume outputs/c_ref/checkpoints/epoch_0011_step_0059994.pt
# The YAML's training.resume_from_checkpoint still works and is overridden
# by the flag.

set -euo pipefail

CONFIG="${1:-configs/experiment/first_run.yaml}"
shift $(( $# > 0 ? 1 : 0 ))
RESUME=()
while [ $# -gt 0 ]; do
    case "$1" in
        --resume)
            if [ $# -gt 1 ] && [[ "$2" != -* ]]; then
                RESUME=(--resume-from "$2"); shift 2
            else
                RESUME=(--resume-from latest); shift
            fi ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load R/4.5.0-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load Pillow/10.4.0-GCCcore-13.3.0
module load R/4.5.0-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a

export R_LIBS_USER=${ELAJ_BASE}/R_libs
export R_MAX_VSIZE=180Gb
export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

cd "$ELAJ_REPO/src"

# Resolve the config against src, accepting an arm name, a src-relative
# path, or a repo-root path. Checked here, after the cd, so the error names
# every form that was tried rather than only the one that was typed.
for candidate in "${CONFIG}" "configs/experiment/${CONFIG}.yaml" \
                 "configs/experiment/${CONFIG}" "../${CONFIG}"; do
    if [ -f "${candidate}" ]; then CONFIG="${candidate}"; break; fi
done
if [ ! -f "${CONFIG}" ]; then
    echo "No experiment config found for '${1:-}'. Tried, from $(pwd):" >&2
    echo "  ${1:-}" >&2
    echo "  configs/experiment/${1:-}.yaml" >&2
    echo "  configs/experiment/${1:-}" >&2
    echo "  ../${1:-}" >&2
    exit 2
fi

nvidia-smi
echo ""
echo "Training with ${CONFIG} ..."
# PYTHONUNBUFFERED: SLURM sends SIGKILL at the walltime limit, and a killed
# process never flushes its stdout buffer.  A job that hits its limit would
# otherwise lose every print() it made — including a fault report it had
# already produced — while stderr survived, which reads as "it hung".
export PYTHONUNBUFFERED=1
if [ ${#RESUME[@]} -gt 0 ]; then echo "Resume: ${RESUME[*]}"; fi
python -u scripts/train.py --config "${CONFIG}" --root . ${RESUME[@]+"${RESUME[@]}"}
