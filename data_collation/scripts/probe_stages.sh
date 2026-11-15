#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
# 2 DAYS, RAISED FROM 16 HOURS AFTER JOB 61809108 TIMED OUT.
# Measured on that job: 24.0 min median per per-group stage at width 32,
# of which only 5.5 min was the two kNN labels this probe wants -- the
# rest was the study-purity sweep and the disease contrast, which
# group_battery now switches off (stages=("knn",)). The model's own group
# stages are 288 wide against those references' 32, so the surviving kNN
# work scales to roughly 50 min each: 44 of them is about 36 hours.
# 16 would not have fitted even with the fix.
#SBATCH --time=2-00:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_probe
#SBATCH --output=logs/probe_%j.log

# Scores every stage of the encoder, and every group encoder's input and
# output, with the benchmark scorer on the benchmark's cells.
#
#   sbatch data_collation/scripts/probe_stages.sh r8_direction \
#       --runs-dir outputs/r8 --out outputs/probes/r8_stages
#
#   # pipeline stages only, for a quick first look:
#   sbatch data_collation/scripts/probe_stages.sh r8_direction \
#       --runs-dir outputs/r8 --no-groups --out outputs/probes/r8_quick
#
#   # the same architecture at random initialisation, no checkpoint:
#   sbatch data_collation/scripts/probe_stages.sh r8_direction \
#       --untrained --out outputs/probes/r8_untrained
#
#   # the model's own stages, without the two PCA references:
#   sbatch data_collation/scripts/probe_stages.sh v1_pool \
#       --no-groups --no-references --out outputs/probes/v1_pool_quick
#
# The first positional argument is the experiment name; everything else is
# passed through to scripts/probe_stages.py.
#
# GPU: the embedding pass only, a few minutes. The rest is CPU, and the cost
# of a stage is the square of the cell count because the patient-identity
# search is brute force. MEASURED on the v-series 382,464 cells: a 1,408-wide
# stage takes 4 h 26 min (job 55708680, 1 Oct 2026). r8's 238k cells put the
# same stage at just over an hour, which is where the old estimate came from.
#
# With --no-groups there are six stages and three of them are that wide, so
# the whole job wants about 16 hours. Two of the three wide ones are the PCA
# references, and they are scored FIRST, so a job that runs out of wall
# returns the references and nothing about the model: that is exactly how
# jobs 55708680 and 55708681 ended. --no-references drops them and brings the
# job to about 9 hours. The 135 group stages take family and phase kNN only.
#
# The script writes the tables after every stage, so a job killed early keeps
# every row it finished.
#
# MEMORY: 180 G. The first run at 120 G died in scIB's silhouette on a
# 1,440-wide stage; scIB is now off unless --scib is passed, and the
# headroom is for the corpus, the references, and the collected stages
# (about 10 GB together) with room for sklearn's brute-force searches.

set -euo pipefail

ARMS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --k|--k-cross-donor|--permutations|--seed|--out|--root|--config-dir|--runs-dir|\
        --reference-components)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --splits)
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

echo "Probing: ${ARMS[0]}"
echo "Options: ${EXTRA[*]-(none)}"
export PYTHONUNBUFFERED=1
python -u -m scripts.probe_stages --run "${ARMS[0]}" --root . \
    ${EXTRA[@]+"${EXTRA[@]}"}
