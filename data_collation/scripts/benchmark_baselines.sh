#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=12:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_bench
#SBATCH --output=logs/bench_%j.log

# Scores scVI, scANVI, Harmony, Scanorama and PCA on the cells ELAJ is scored
# on, with ELAJ's own checkpoint in the same table.
#
#   sbatch data_collation/scripts/benchmark_baselines.sh r8_direction \
#       --runs-dir outputs/r8 --out outputs/benchmarks/r8_all
#
#   # a second pass on each method's own HVG set, once `all` has been read:
#   sbatch data_collation/scripts/benchmark_baselines.sh r8_direction \
#       --runs-dir outputs/r8 --gene-policies hvg2000 \
#       --out outputs/benchmarks/r8_hvg2000
#
#   # the fair comparison: baselines fitted on the training donors and
#   # applied to the held-out ones, as ELAJ is. Harmony and Scanorama
#   # cannot and are reported as skipped:
#   sbatch data_collation/scripts/benchmark_baselines.sh r8_direction \
#       --runs-dir outputs/r8 --fit-split train --methods pca scvi scanvi \
#       --out outputs/benchmarks/r8_inductive
#
# The first positional argument is the experiment name; everything else is
# passed through to scripts/benchmark_baselines.py. --include-elaj is on by
# default here because the point of the table is ELAJ's row; pass
# --no-elaj to drop it (for instance to score baselines on a corpus no
# checkpoint exists for).
#
# GPU: scVI and scANVI train on it; the rest are CPU. One A100 is plenty.
# MEMORY: 180 G because --fit-split train holds the ~1.2 M-cell training
# corpus beside the scored one, plus scVI's own copy.
# WALLTIME: 12 h is generous. scVI on ~240k cells is under an hour on an
# A100, scANVI similar, Scanorama the slowest of the CPU methods at tens of
# minutes, and the 200-permutation cross-donor null is the same cost as in
# validate_latent. The script writes the table after every method, so a job
# killed early still leaves everything it finished.
#
# PREREQUISITE: install_baselines.sh, once. A method whose package is absent
# is skipped and named in the table rather than killing the run.

set -euo pipefail

ARMS=()
EXTRA=()
INCLUDE_ELAJ=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-elaj) INCLUDE_ELAJ=0; shift ;;
        --k|--k-cross-donor|--permutations|--seed|--scvi-epochs|--scanvi-label|\
        --fit-split|--pca-components|--scvi-latent|--harmony-pcs|--scanorama-dimred|\
        --out|--root|--config-dir|--runs-dir)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --splits|--methods|--gene-policies)
            # nargs="+": take every following bare word.
            EXTRA+=("$1"); shift
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
                EXTRA+=("$1"); shift
            done ;;
        --run)
            echo "Pass the arm name without --run; this script adds it." >&2; exit 2 ;;
        -*) EXTRA+=("$1"); shift ;;
        *)  ARMS+=("$1"); shift ;;
    esac
done
if [ ${#ARMS[@]} -ne 1 ]; then
    echo "Give exactly one experiment name. Got: ${ARMS[*]-none}" >&2
    exit 2
fi
if [ "$INCLUDE_ELAJ" -eq 1 ]; then
    EXTRA+=("--include-elaj")
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

python -u - <<'PY'
import importlib
print("packages:", ", ".join(
    f"{p}={'ok' if importlib.util.find_spec(p) else 'MISSING'}"
    for p in ("scanpy", "harmonypy", "scanorama", "scvi", "scib_metrics", "scib")))
PY

echo "Benchmarking: ${ARMS[0]}"
echo "Options:      ${EXTRA[*]-(none)}"
export PYTHONUNBUFFERED=1
python -u -m scripts.benchmark_baselines --run "${ARMS[0]}" --root . \
    ${EXTRA[@]+"${EXTRA[@]}"}
