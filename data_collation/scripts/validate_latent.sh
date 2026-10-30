#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=12:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_validate
#SBATCH --output=logs/validate_%j.log

# THIS SCRIPT HOLDS AN A100 FOR UP TO TWELVE HOURS AND NEEDS IT FOR ABOUT ONE.
# Only the forward pass touches CUDA. Stages 3, 4 and 4b are neighbour
# searches in numpy and sklearn and account for roughly nine of those hours
# -- 4b alone was 4h57m on 1.58 M cells. If all you want is the per-cell
# latent, use dump_cells.sh: two-hour wall, GPU released when the embedding
# is done. Reach for THIS script only when the measurement battery is the
# point, and pass --no-figures; figures are drawn locally.
#
# Measured, 28 October 2026, five arms at --splits train val test:
#   dump written        00:51
#   stage 3             02:33
#   stage 4             01:44
#   stage 4b            04:57   <- 41% of the job
#   stage 5 NMI/ARI     01:30   <- retired metric, pass --skip-nmi when it exists
#   stages 6, 7         00:26
# Total beyond the twelve-hour wall: four of the five died in stage 5.

# Does the latent carry anything the model was never given?
#
#   sbatch data_collation/scripts/validate_latent.sh r8_direction
#   sbatch data_collation/scripts/validate_latent.sh --runs-dir outputs/r8 r8_direction
#   sbatch data_collation/scripts/validate_latent.sh --splits val test r8_direction
#
# --splits is what lifts the cross-donor phase test off its floor. On val alone
# only six donors carry a phase label, and at six donors the smallest reachable
# p is 0.10 -- the test cannot answer. val+test gives eighteen held-out donors.
#
# ALWAYS PASS --out WHEN RUNNING MORE THAN ONE SPLIT COMBINATION. The figures
# go where --out points, so two runs sharing a directory silently overwrite
# each other's pictures.
#
# One experiment NAME, not an output path. Flags beginning with "-" pass
# through to validate_latent.py; see report_purity.sh for the same convention.
#
# No training. It embeds the split once from a checkpoint already on disk, so
# it can run against a finished wave while the next one trains.
#
# ripser is optional and only figure 23 (persistent homology) needs it. Install
# it from a LOGIN node, where there is internet:
#     source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate
#     pip install ripser
# Without it that one figure reports the library as missing and the other
# fifteen are unaffected.

set -euo pipefail

ARMS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dims|--k|--out|--root|--config-dir|--runs-dir|--cycle-period|--permutations|\
        --umap-cells|--baseline-cells|--topology-cells|--dump-cells)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --splits)
            # nargs="+": take every following bare word as a split name.
            EXTRA+=("$1"); shift
            while [ $# -gt 0 ] && [[ "$1" != -* ]] && \
                  [[ "$1" == train || "$1" == val || "$1" == test ]]; do
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

python -u -c "import sklearn; print('sklearn', sklearn.__version__)"

echo "Validating: ${ARMS[0]}"
echo "Options:    ${EXTRA[*]-(none)}"
export PYTHONUNBUFFERED=1
python -u -m scripts.validate_latent --run "${ARMS[0]}" --root . \
    ${EXTRA[@]+"${EXTRA[@]}"}
