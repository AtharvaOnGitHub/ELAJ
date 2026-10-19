#!/bin/bash
# One-time: a CPU-only benchmarking venv that needs no GPU node, so scoring
# stops competing with training for the two-GPU cap.
#
#   bash data_collation/scripts/install_benchmark_cpu.sh
#
# Login node, once. Safe to re-run. FRESH=1 rebuilds from scratch.
#
# WHY THIS EXISTS, MEASURED 19 OCTOBER. `sacctmgr show qos bbgpu` gives
# MaxTRESPU = cpu=36, gres/gpu=2. Those are one limit stated twice: two GPUs
# at the docs' 18-cores-per-GPU entitlement. So the account may hold at most
# TWO GPU jobs at any time, ever. Every scoring job run under bbgpu therefore
# costs a training slot, and with the ablations still to come the whole
# remaining campaign would be serialised in pairs.
#
# The scorer does not need a GPU and never did: it is a sklearn kNN search,
# a permutation null, and scIB on jax-CPU. score_external.sh's own header
# already says so. The only reason it asked for a card is that the PyTorch
# module the elaj venv was built against lives in the GPU nodes' software
# stack, and `module load PyTorch/...-CUDA-12.6.0` on bbdefault fails with
# "the following module(s) are unknown". The computation did not need a card;
# the environment did.
#
# A job on bbdefault sits outside the bbgpu QoS entirely, so it runs BESIDE
# two training jobs rather than displacing one.
#
# NO BEAR PYTHON MODULES, which is install_foundation.sh's approach and not
# install_baselines.sh's. The elaj venv interoperates with cluster modules on
# purpose; this venv must not, because the module that supplies its numpy also
# supplies a CUDA torch that only exists on GPU nodes. Nothing is loaded here
# but a bare python, and pip installs its own CPU torch.
#
# THE PINS ARE READ FROM THE EXISTING elaj VENV, NOT TYPED HERE, and that is
# the whole point of step 1. The benchmark rows this venv produces have to be
# comparable with the m5, m6 and foundation-model rows already in
# SCOREBOARD.md, which were scored under the GPU venv. Different scanpy or
# scib-metrics versions can move scIB numbers, and a table whose rows were
# computed by two different library stacks is not one table. So this script
# activates the GPU venv, records what it actually resolves to, and builds
# the CPU venv to match.
#
# VALIDATE BEFORE TRUSTING IT. After installing, re-score one arm that has
# already been scored and diff the two benchmark.csv files; they should agree
# to the last digit on every row. benchmark_cpu.sh's header gives the command.

set -euo pipefail

ENVS=/rds/projects/a/acharjea-multi-omics/Atharva/envs
GPU_VENV="${ENVS}/elaj"
CPU_VENV="${ENVS}/elaj_cpu"
PINS="${ENVS}/elaj_cpu_pins.txt"

# ── 1. read what the GPU venv resolves to, with its modules loaded ──────────
echo "── reading versions from the GPU venv, so the CPU venv can match ──"
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
source "${GPU_VENV}/bin/activate"

python - "${PINS}" <<'PYEOF'
import importlib, sys
# torch is deliberately absent: the CPU venv installs its own CPU build, and
# matching the CUDA module's 2.7.1 exactly is neither possible nor needed --
# nothing in the scorer runs on the GPU, so torch is only a checkpoint loader.
WANT = ["numpy", "pandas", "scipy", "anndata", "h5py", "matplotlib",
        "sklearn", "scanpy", "scvi", "harmonypy", "scanorama",
        "scib_metrics", "scib", "jax", "yaml", "tqdm", "mudata",
        "zarr", "numcodecs", "xarray", "session_info"]
DIST = {"sklearn": "scikit-learn", "scvi": "scvi-tools",
        "scib_metrics": "scib-metrics", "yaml": "PyYAML",
        "session_info": "session-info"}
lines, missing = [], []
for mod in WANT:
    try:
        m = importlib.import_module(mod)
    except Exception:                                          # noqa: BLE001
        missing.append(mod); continue
    v = getattr(m, "__version__", None)
    if not v:
        try:
            from importlib.metadata import version
            v = version(DIST.get(mod, mod))
        except Exception:                                      # noqa: BLE001
            missing.append(mod); continue
    name = DIST.get(mod, mod)
    lines.append(f"{name}=={v}")
    print(f"  {name:<16} {v}")
if missing:
    print("  not present in the GPU venv, so not pinned: " + ", ".join(missing))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
PYEOF

echo
echo "── pins written to ${PINS} ──"
deactivate

# ── 2. a clean venv with NO modules but a bare python ───────────────────────
module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }
export PYTHONNOUSERSITE=1

if [[ "${FRESH:-0}" == "1" && -d "${CPU_VENV}" ]]; then
    echo "── FRESH=1: removing ${CPU_VENV} ──"
    rm -rf "${CPU_VENV}"
fi
[[ -d "${CPU_VENV}" ]] || python -m venv "${CPU_VENV}"
source "${CPU_VENV}/bin/activate"
pip install --quiet --upgrade pip wheel setuptools

echo
echo "── torch, CPU build: no CUDA runtime, no GPU node needed ──"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo
echo "── everything else, at the GPU venv's versions ──"
# -c as well as -r: anything pulled in transitively is held to the same
# version the GPU venv resolved, not merely the packages named in the file.
pip install -r "${PINS}" -c "${PINS}"

# ── 3. verify by importing, not by pip's opinion ────────────────────────────
echo
echo "── verifying ──"
python - <<'PYEOF'
import importlib
ok = True
for pkg in ("numpy", "pandas", "scipy", "anndata", "h5py", "matplotlib",
            "sklearn", "scanpy", "scvi", "harmonypy", "scanorama", "yaml",
            "tqdm", "torch"):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:<14} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print(f"  {pkg:<14} MISSING ({type(exc).__name__}: {exc})")
for pkg in ("scib_metrics", "scib"):
    try:
        importlib.import_module(pkg); print(f"  {pkg:<14} ok"); break
    except Exception:                                          # noqa: BLE001
        continue
else:
    print("  scib           none (sklearn-only NMI/ARI) -- scIB rows would DIFFER")
    ok = False
import torch
print(f"  torch build    {torch.__version__}")
print(f"  cuda compiled  {torch.version.cuda}  (None is correct for this venv)")
raise SystemExit(0 if ok else 1)
PYEOF

echo
echo "Installed: ${CPU_VENV}"
echo "NEXT: validate it before trusting a row. See benchmark_cpu.sh's header."
