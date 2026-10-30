#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=02:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_dump
#SBATCH --output=logs/dump_%j.log

# Embed every cell and write the latent. Nothing else.
#
#   sbatch data_collation/scripts/dump_cells.sh --runs-dir outputs/a_series a1_noecs
#
# WHY THIS EXISTS AS ITS OWN SCRIPT. validate_latent.sh asks for an A100 and
# twelve hours because its measurement battery is long. The forward pass is
# the ONLY stage that touches CUDA -- everything after it is numpy and
# sklearn. On 28 October 2026 five validate jobs reserved a card for twelve
# hours each, finished the embedding at fifty-one minutes, spent eleven hours
# doing CPU work while holding the GPU, and were killed on the time limit
# before writing their figures. Sixty A100-hours reserved, about four used,
# and the only artefact anyone wanted -- the dump -- had been on disk since
# the first hour.
#
# So: GPU work gets a GPU and a two-hour wall. The measurement battery is a
# separate decision, and figures are drawn locally, never here.
#
# Two hours is the wall because the observed dump time is 51 minutes for
# 1,577,728 cells across train+val+test. It is generous, not tight. A run
# that needs more than two hours has a different problem.

set -euo pipefail

module purge
module load bluebear
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load scikit-learn/1.9.0-gfbf-2024a

ARMS=()
PASSTHRU=()
OUT_GIVEN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dump-cells)
            echo "dump_cells.sh sets --dump-cells itself; pass only the arm." >&2
            exit 2 ;;
        --out)
            OUT_GIVEN=1; PASSTHRU+=("$1" "$2"); shift 2 ;;
        --runs-dir|--splits|--root|--config-dir)
            PASSTHRU+=("$1" "$2"); shift 2 ;;
        --run)
            echo "Pass the arm name without --run; this script adds it." >&2
            exit 2 ;;
        -*)
            PASSTHRU+=("$1"); shift ;;
        *)
            ARMS+=("$1"); shift ;;
    esac
done

if [[ ${#ARMS[@]} -ne 1 ]]; then
    echo "usage: sbatch dump_cells.sh [--runs-dir DIR] [--splits ...] ARM" >&2
    exit 2
fi
ARM="${ARMS[0]}"

cd "${SLURM_SUBMIT_DIR}/src"

# --splits defaults to val in validate_latent.py, which is not what a
# whole-corpus dump means. Set all three unless the caller said otherwise.
if ! printf '%s\n' "${PASSTHRU[@]:-}" | grep -qx -- "--splits"; then
    PASSTHRU+=(--splits train val test)
fi
if [[ ${OUT_GIVEN} -eq 0 ]]; then
    PASSTHRU+=(--out "outputs/validate/${ARM}")
fi

python -u -m scripts.validate_latent \
    --run "${ARM}" --root . \
    --dump-cells "outputs/cells/${ARM}" \
    --dump-only \
    --no-figures \
    "${PASSTHRU[@]}"
