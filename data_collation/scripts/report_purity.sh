#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_purity
#SBATCH --output=logs/purity_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# kNN study purity for finished runs, from their best checkpoints.
#
#   sbatch data_collation/scripts/report_purity.sh                  # all of wave R4
#   sbatch data_collation/scripts/report_purity.sh r3_nodab r3_loose
#   sbatch data_collation/scripts/report_purity.sh --dims 4 r5_cce r5_lambda_low
#   sbatch data_collation/scripts/report_purity.sh --runs-dir outputs/r5 r5_cce
#
# --runs-dir is for a wave that has been filed into a subdirectory after it
# finished: the config still records outputs/<arm>/, which stops being where
# the outputs are.
#
# Arms are EXPERIMENT NAMES, not output paths: each resolves to
# configs/experiment/<name>.yaml. The --run flag is added below, so do not pass
# it yourself. Anything beginning with "-" is forwarded to report_purity.py
# verbatim, which is how --dims, --k and --out get through.
#
# No training. Each run is embedded once through its best checkpoint and
# measured; the expensive part is rebuilding the validation Dataset, which runs
# sharing a corpus do once between them.
#
# Four hours for eight arms: six share the full corpus and build it once,
# r4_eutopic and r4_ectopic each need their own. A GPU because the encoder runs
# on it and because the PyTorch module lives in the GPU nodes' software stack.

set -euo pipefail

# Split the command line into arm names and passthrough flags. Without this,
# a flag lands in ARMS, gets an unwanted --run in front of it, and argparse
# fails with "--run: expected one argument" -- which reads like the arm list is
# wrong when the real problem is that flags had nowhere to go.
#
# A flag's VALUE is a bare word, so it cannot be told from an arm name by shape
# alone. The flags that take one are listed explicitly; anything else beginning
# with "-" is assumed to be a bare switch.
ARMS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dims|--k|--out|--root|--config-dir|--runs-dir)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --run)
            echo "Pass arm names without --run; this script adds it." >&2; exit 2 ;;
        -*)
            EXTRA+=("$1"); shift ;;
        *)
            ARMS+=("$1"); shift ;;
    esac
done

if [ ${#ARMS[@]} -eq 0 ]; then
    ARMS=(r4_tiny_latent r4_thin_groups r4_deep_cell r4_tiny_dab_hi
          r4_eutopic r4_ectopic)
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
source ${ELAJ_BASE}/envs/elaj/bin/activate

cd "$ELAJ_REPO/src"

# scikit-learn does the nearest-neighbour search. Checked before the split is
# rebuilt so a missing dependency costs a line now, not twenty minutes.
python -u -c "import sklearn; print('sklearn', sklearn.__version__)"

RUN_ARGS=()
for arm in "${ARMS[@]}"; do RUN_ARGS+=(--run "${arm}"); done

echo "Measuring: ${ARMS[*]}"
echo "Options:   ${EXTRA[*]-(none)}"
export PYTHONUNBUFFERED=1

# ${EXTRA[@]+...} guards the expansion: under `set -u`, bash 3.2 treats an
# empty "${EXTRA[@]}" as an unbound variable and aborts. A --no-tests flag in
# launch_r2.sh once submitted zero jobs and reported success for exactly this.
python -u -m scripts.report_purity --root . \
    "${RUN_ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
