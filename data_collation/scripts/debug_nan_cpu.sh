#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=2:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_debug_nan_cpu
#SBATCH --output=logs/debug_nan_cpu_%j.log

# The CPU-only twin of debug_nan.sh.  No --gres, so it runs on the ordinary
# compute partition rather than icelake-gpua100-shared.
#
#   sbatch data_collation/scripts/debug_nan_cpu.sh
#   sbatch data_collation/scripts/debug_nan_cpu.sh --max-cells-per-study 2000
#
# Two reasons to use it:
#
#   1. It sidesteps the GPU partition entirely, which matters when GPU jobs are
#      being cancelled for reasons outside this project.
#   2. Where the fault appears is itself evidence.  Reproducing on CPU means a
#      numerical fault in the maths; reproducing only on CUDA points at
#      something device-specific, such as an out-of-range gather reading
#      unallocated memory instead of raising.
#
# Slower per batch than the GPU, but the fault is expected on batch 0, so the
# forward pass count is small.  The walltime is set for the dataset load, not
# the compute.

set -euo pipefail

module purge
module load bear-apps/2024a

# bear-apps/2024a carries exactly one PyTorch, and only in the icelake tree
# (/rds/bear-apps/2024a/EL8-ice/modules/all).  There is no CPU-only build, so
# --constraint=icelake above is what makes this work: bbdefault spans several
# architectures and a non-icelake node has no PyTorch at all.  The CUDA build
# runs fine without a GPU — torch.cuda.is_available() simply returns False.
#
# The loop and the availability dump remain so that a node without the module
# reports what it does have, rather than failing with "unknown module" and
# costing another submit to learn anything.
PYTORCH_CANDIDATES=(
    "PyTorch/2.7.1-foss-2024a-CUDA-12.6.0"
)
loaded=""
for candidate in "${PYTORCH_CANDIDATES[@]}"; do
    if module load "${candidate}" >/dev/null 2>&1; then
        loaded="${candidate}"
        break
    fi
done
if [ -z "${loaded}" ]; then
    {
        echo "ERROR: no known PyTorch module on $(hostname). Tried:"
        printf '  %s\n' "${PYTORCH_CANDIDATES[@]}"
        echo "Available on this node:"
        module -t avail PyTorch 2>&1 | sed 's/^/  /'
    } >&2
    exit 1
fi
echo "Loaded ${loaded} on $(hostname)"

module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load matplotlib/3.9.2-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

ARGS=(--config configs/experiment/first_run.yaml --root . --device cpu)
if [ "$#" -gt 0 ]; then
    ARGS+=("$@")
else
    ARGS+=(--match-training --max-cells-per-study 0)
fi

echo "Running: python scripts/debug_nan.py ${ARGS[*]}"
# PYTHONUNBUFFERED: SLURM sends SIGKILL at the walltime limit, and a killed
# process never flushes its stdout buffer.  A job that hits its limit would
# otherwise lose every print() it made — including a fault report it had
# already produced — while stderr survived, which reads as "it hung".
export PYTHONUNBUFFERED=1
python -u scripts/debug_nan.py "${ARGS[@]}"
