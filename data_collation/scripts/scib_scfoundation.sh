#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --job-name=elaj_scib_scf
#SBATCH --output=logs/scib_scf_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# The one scIB cell scFoundation does not have: silhouette_batch.
#
#   sbatch data_collation/scripts/scib_scfoundation.sh <external-latent-dir>
#
# WHAT IS MISSING AND WHY. scFoundation has nine of its ten scIB metrics on
# the `cell family` label and they are all plausible. silhouette_batch alone
# reads -2.954e+24, which is not a measurement: every other representation
# sits between 0.4649 and 0.8909. The full-split pass was OOM-killed INSIDE
# silhouette_batch after fifteen hours, and that garbage is what it wrote on
# the way down. `figures/scripts/scib_table.py` quarantines it rather than
# shipping it, so the supplementary scorecard currently has one empty cell.
#
# ── NOTHING UNNECESSARY RUNS, which is the whole point ─────────────────────
#
# `--stages scib` runs the scIB battery and nothing else. The default is knn,
# cross-donor, mixing and disease as well, and that default is what made the
# original pass fifteen hours long.
#
# `--external-latent` scores an embedding that is ALREADY ON DISK. No GPU, no
# re-embedding, no model load. This is why the job asks for no card.
#
# `--methods none` fits nothing. The flag's own help names this exact case:
# "Use 'none' for a pass that only scores --external-latent rows beside a
# --reuse-baselines file". So pca, harmony, scanorama, scVI and scANVI are
# copied in from the reuse file rather than refitted.
#
# `--include-elaj` is OMITTED, which is how our own arms stay out of it. They
# already have this metric. (The script's own line 456 mentions a `--no-elaj`
# flag; it does not exist, and this script was briefly written against it.)
#
# ── 360 G, AND THAT IS THE POINT OF THE RERUN ──────────────────────────────
#
# silhouette_batch is what ran out of memory. It computes a silhouette over
# 176,803 cells grouped by batch, and scib's implementation materialises a
# dense pairwise block per group. benchmark_cpu.sh's 240 G is sized for a
# pass whose measured peak was 201 GB and which did NOT complete this stage,
# so matching it would be repeating the experiment. bbdefault icelake nodes
# carry enough for 360 G on one node; if this still dies, the next move is
# subsampling the silhouette rather than more memory, and the supplementary
# should then report the cell count it was computed on.
#
# 6 HOURS. One stage over one representation. The fifteen-hour figure was
# four stages over ten.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat >&2 <<'USAGE'
usage: scib_scfoundation.sh <external-latent-dir> [reuse-benchmark.csv]

<external-latent-dir> is the directory holding scFoundation latent.npy and
manifest.json, written by scripts/embed_pretrained.py. Find it with:

    find /rds -maxdepth 6 -name manifest.json -path "*scfound*" 2>/dev/null

IF IT IS GONE this job cannot run. The latent would have to be recomputed by
embed_pretrained.py on a GPU first, which is a different and much larger job.
Do not point this at a directory for a different model: the row name and the
gene coverage come from the manifest, so a mismatch is silent.
USAGE
    exit 64
fi
EMB="$1"

REUSE="${2:-outputs/benchmarks/wide_full_inductive/benchmark.csv}"

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1
VENV=${ELAJ_BASE}/envs/elaj_cpu
[[ -f "${VENV}/bin/activate" ]] || {
    echo "no CPU venv at ${VENV}. Run install_benchmark_cpu.sh" >&2; exit 1; }
source "${VENV}/bin/activate"

cd "$ELAJ_REPO/src"

[[ -f "${EMB}/latent.npy" ]] || { echo "no latent.npy in ${EMB}" >&2; exit 2; }
[[ -f "${EMB}/manifest.json" ]] || { echo "no manifest.json in ${EMB}" >&2; exit 2; }
[[ -f "${REUSE}" ]] || { echo "no reuse file at ${REUSE}" >&2; exit 2; }

# PREFLIGHT. scib is the package that must be present, and it is the one the
# metric lives in; a missing scib would fail after the latent load.
python -u - <<'PY'
import importlib.util, sys

def has(n): return importlib.util.find_spec(n) is not None

# scib_metrics OR scib, not both. evaluation/scib_metrics.py prefers
# scib_metrics and falls back to scib, and silhouette_batch comes from
# scib_metrics.silhouette_batch on the preferred path. Requiring both is
# what made the first attempt exit when elaj_cpu has scib_metrics and no
# scib, which is a working configuration.
if not (has("scib_metrics") or has("scib")):
    sys.exit("MISSING from elaj_cpu: neither scib_metrics nor scib")
missing = [p for p in ("scanpy", "numpy", "pandas") if not has(p)]
if missing:
    sys.exit(f"MISSING from elaj_cpu: {', '.join(missing)}")
print(f"preflight ok (backend: "
      f"{'scib_metrics' if has('scib_metrics') else 'scib'})")
PY

echo "scIB only, scFoundation only, from ${EMB}"
export PYTHONUNBUFFERED=1
python -u -m scripts.benchmark_baselines \
    --run p1_dab168_kl1 \
    --root . \
    --methods none \
    --stages scib \
    --external-latent "${EMB}" \
    --reuse-baselines "${REUSE}" \
    --out outputs/benchmarks/scib_scfoundation_rerun

echo
echo "Now diff the nine metrics that already existed. They must be identical;"
echo "only silhouette_batch should be new:"
echo "  grep scib- outputs/benchmarks/scib_scfoundation_rerun/benchmark.csv \\"
echo "    | grep scfoundation"
