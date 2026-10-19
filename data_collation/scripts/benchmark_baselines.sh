#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=05:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=elaj_bench
#SBATCH --output=logs/bench_%j.log

# Scores scVI, scANVI, Harmony, Scanorama and PCA on the cells ELAJ is scored
# on, with ELAJ's own checkpoint in the same table.
#
# SIZED FROM SEVEN RECORDED RUNS, not guessed. It asked for 12 hours and
# 256G, and on 9 October 2026 that left a pass sitting on Priority while a
# card stood free.
#
#   run                    memory      wall     baselines
#   m3_study_v2             66.20 GB   3:12:53  refitted
#   r8_inductive           101.88 GB   2:26:37  refitted
#   r8_all                  74.47 GB   2:17:34  refitted
#   m3_study_mixing         68.36 GB   1:44:03  refitted
#   m5_disease              70.92 GB   1:10:54  reused
#   m5_study_v2             66.97 GB   1:03:17  reused
#   m4_tight16_study_v2     65.94 GB   0:58:11  reused
#
# MEMORY IS THE BINDING CONSTRAINT AND 128G IS NOT ARBITRARY. An a100_80 node
# holds 512 GB across four GPUs, so 128G is exactly one GPU's proportional
# share and the job can sit beside three others. At 256G a single-GPU job
# needs half the node's memory and blocks two of them, so it only ever starts
# where that much happens to be free. 128G is 26% above the worst run on
# record.
#
# 5 hours covers the slowest full refit with 56% to spare. A reuse pass needs
# about two: pass --time=02:00:00 for those. Both are overridable on the
# sbatch command line, which beats editing this file for one run.
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
# THE WIDE-AXIS PASS NEEDS ITS OWN SIZING, and the defaults above are wrong
# for it. --baseline-vocabulary doubles the baselines' gene axis to 30,962 and
# builds the corpus TWICE, once per axis, so both numbers move:
#
#   MEASURED, job 57362837, the 30,962-gene inductive pass with pca, scvi
#   and scanvi, --no-elaj:
#
#       three corpus builds (narrow, wide, wide fit)      0:12
#       pca, chunked IncrementalPCA over 2,269 chunks     2:41
#       scvi, 7 epochs                                    1:03
#       scanvi, 7 + 20 epochs                             2:41
#       total                                             6:39
#       peak memory                                     201 GB
#
#   So 360G and 20 hours are both comfortable, with 44% of the memory and
#   two thirds of the wall unused. 240G would fit; 128G would not.
#
#   The estimates these replace were 220-300 GB and 8-10 hours, which were
#   conservative in the right direction. The PCA line predicted 3.0 h against
#   2:41 actual, so the prediction is worth trusting as a hung-or-slow
#   discriminator within two minutes of a job starting.
#
#   sbatch --mem=360G --time=20:00:00 \
#       data_collation/scripts/benchmark_baselines.sh m6_drain --no-elaj \
#       --fit-split train --methods pca scvi scanvi \
#       --baseline-vocabulary /rds/projects/a/acharjea-multi-omics/Atharva/data/processed/vocabulary_axis.json \
#       --out outputs/benchmarks/axis_m6_drain
#
# If the QoS refuses 20 hours, drop scanvi: it cost 6,554 s of the measured
# run and its cell-family score is supervised and not comparable anyway.
#
# --no-elaj, BECAUSE ELAJ'S ROW CANNOT MOVE. It reads the vocabulary its
# checkpoint was trained against whatever the baselines are given, on the same
# cells, in eval mode, with blanking off and the same scorer seed. Embedding it
# again would reproduce the narrow pass by construction and cost 29 minutes to
# learn nothing, so the wide table carries no elaj row and is read beside the
# narrow one.
#
# WHAT TO CHECK INSTEAD, and it is the thing that would void the comparison:
# the log must say 416,256 cells. Without --include-elaj the cell order comes
# from a fresh sampler_order() call rather than from ELAJ's loader pass, and if
# those two disagree the baselines are on a different cell set from the narrow
# table's. The alignment of the two GENE axes is separately checked on every
# row by gene_axis.require_same_cells, which is why the narrow corpus is still
# built when nothing scores against it.
#
# The first positional argument is the experiment name; everything else is
# passed through to scripts/benchmark_baselines.py. --include-elaj is on by
# default here because the point of the table is ELAJ's row; pass
# --no-elaj to drop it (for instance to score baselines on a corpus no
# checkpoint exists for).
#
# GPU: scVI and scANVI train on it; the rest are CPU. One A100 is plenty.
# MEMORY: 256 G. At 180 G, job 55184824 was OOM-killed fitting PCA on the
# 1,195,264-cell training corpus: sc.pp.pca densifies to centre, which is
# 74 GB, on top of two sparse copies at 18 GB each. r8_inductive did the
# same at 784,896 cells and peaked at 101.88 GB, so the headroom went when
# the controls came in and the corpus grew 52%. An A100-80 node holds
# 512 GB across four GPUs, so 256 G for one GPU is half the memory for a
# quarter of the cards: it may queue a little longer on the shared
# partition, and it is well inside the node.
#
# methods.py sizes its own dense-matrix budget from SLURM_MEM_PER_NODE, so
# lowering this does not fail, it switches PCA to IncrementalPCA.
#
# ORIGINAL NOTE: 180 G because --fit-split train holds the ~1.2 M-cell training
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
