#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=08:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --job-name=elaj_bench_cpu
#SBATCH --output=logs/bench_cpu_%j.log

# benchmark_baselines.sh with no GPU, so it runs BESIDE two training jobs
# instead of displacing one.
#
#   sbatch data_collation/scripts/benchmark_cpu.sh p1_dab7_kl1 \
#       --runs-dir outputs/p_series --fit-split train \
#       --reuse-baselines outputs/benchmarks/axis_m6_drain/benchmark.csv \
#       --out outputs/benchmarks/p1_dab7_kl1
#
# Identical arguments to benchmark_baselines.sh -- the parsing is the same
# file, sourced by both.
#
# WHY THIS IS NOT A GPU JOB, AND WHY THAT MATTERS NOW. Measured 19 October:
#
#   sacctmgr show qos bbgpu format=Name,MaxTRESPU
#   bbgpu    cpu=36,gres/gpu=2
#
# Two GPUs is the account's hard ceiling across the whole QoS, and 36 cores is
# the same ceiling restated at the docs' 18-cores-per-GPU entitlement. So
# under bbgpu every scoring job costs one of two training slots. On 19 October
# two benchmark jobs sat on QOSMaxCpuPerUserLimit behind two training runs,
# and no reduction in --cpus-per-task could have started them: the GPU count
# was independently at 2 of 2.
#
# bbdefault is a different QoS and does not touch that budget.
#
# THE COMPUTATION NEVER NEEDED A CARD. It is a sklearn kNN search, the
# permutation null, and scIB on jax-CPU; score_external.sh's header has said
# so since it was written, and asked for --gres=gpu:1 anyway because the
# PyTorch module the elaj venv is built against exists only in the GPU nodes'
# software stack. elaj_cpu removes that reason: its torch is pip's CPU build
# and it loads no BEAR Python modules at all. See install_benchmark_cpu.sh.
#
# 16 CORES AND 240 G, NOT THE GPU SCRIPT'S 8 AND 128. There is no
# per-GPU entitlement to respect here, and the measured wide-axis pass peaked
# at 201 GB, so 240 G is the figure that one actually needs. More cores help:
# the kNN search and the permutation null are both sklearn n_jobs work.
#
# 8 HOURS. The GPU script's 5 covers a full refit; this is sized for a reuse
# pass, which is what --reuse-baselines makes it. A pass that refits scVI and
# scANVI on CPU would be far slower and should stay on benchmark_baselines.sh.
#
# ── VALIDATE BEFORE TRUSTING A ROW ─────────────────────────────────────────
# The rows in SCOREBOARD.md were computed under the GPU venv. This venv is
# pinned to the same package versions (install_benchmark_cpu.sh reads them
# from the GPU venv rather than hardcoding them), but pinned is not the same
# as proven. Re-score one arm that has already been scored and diff:
#
#   sbatch data_collation/scripts/benchmark_cpu.sh m6_drain --no-elaj \
#       --runs-dir outputs/m_series \
#       --reuse-baselines outputs/benchmarks/axis_m6_drain/benchmark.csv \
#       --out outputs/benchmarks/cpuvenv_check
#
#   diff <(sort outputs/benchmarks/axis_m6_drain/benchmark.csv) \
#        <(sort outputs/benchmarks/cpuvenv_check/benchmark.csv)
#
# Expect no differences outside the fit_seconds column. Any difference in a
# `value` means the two stacks do not agree and the CPU venv must not be used
# for rows that sit beside GPU-venv rows in the same table.

set -euo pipefail

# Argument parsing lives in _benchmark_args.sh so the two wrappers cannot
# drift. It sets ARMS and EXTRA from "$@".
#
# RESOLVED FROM THE REPO PATH, NOT FROM BASH_SOURCE. sbatch COPIES the script
# to /var/spool/slurmd/job<id>/slurm_script before running it, so under a job
# ${BASH_SOURCE[0]} is that spooled copy and its dirname holds no sibling
# files. `source "$(dirname "${BASH_SOURCE[0]}")/_benchmark_args.sh"` therefore
# works when the script is run directly and fails inside every sbatch job with
# "No such file or directory" -- which is exactly what job 58007920 did.
#
# The sibling is still preferred when it exists, so a second checkout or a
# local run uses its own copy rather than this one.
REPO=/rds/homes/a/axw1008/axw1008
ARGS="${REPO}/data_collation/scripts/_benchmark_args.sh"
SIBLING="$(dirname "${BASH_SOURCE[0]}")/_benchmark_args.sh"
[[ -f "${SIBLING}" ]] && ARGS="${SIBLING}"
[[ -f "${ARGS}" ]] || { echo "cannot find _benchmark_args.sh (looked at ${SIBLING} and ${REPO}/data_collation/scripts/)" >&2; exit 2; }
source "${ARGS}"

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1
VENV=/rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj_cpu
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "no CPU venv at ${VENV}." >&2
    echo "Run: bash data_collation/scripts/install_benchmark_cpu.sh" >&2
    exit 1
fi
source "${VENV}/bin/activate"

cd /rds/homes/a/axw1008/axw1008/src

python -u - <<'PY'
import importlib
print("packages:", ", ".join(
    f"{p}={'ok' if importlib.util.find_spec(p) else 'MISSING'}"
    for p in ("scanpy", "harmonypy", "scanorama", "scvi", "scib_metrics", "scib")))
import torch
print(f"torch {torch.__version__}, cuda compiled {torch.version.cuda} "
      f"(None is correct), available {torch.cuda.is_available()}")
PY

echo "Benchmarking: ${ARMS[0]}  (CPU, bbdefault, outside the bbgpu cap)"
echo "Options:      ${EXTRA[*]-(none)}"
export PYTHONUNBUFFERED=1
python -u -m scripts.benchmark_baselines --run "${ARMS[0]}" --root . \
    ${EXTRA[@]+"${EXTRA[@]}"}
