#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=00:40:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=elaj_corpus
#SBATCH --output=logs/corpus_%j.log

# Tabulates and plots what the corpus carries in its metadata columns.
#
#   sbatch data_collation/scripts/describe_corpus.sh configs/experiment/s1_circ_low.yaml
#
# No PyTorch, and that is deliberate rather than incidental.  The first version
# of this script copied train.sh's module list, which loads
# PyTorch/2.7.1-foss-2024a-CUDA-12.6.0.  That module exists in the GPU nodes'
# software stack and not on the default partition, so the job died at load time
# with "the following module(s) are unknown" — a message about a CUDA build, on
# a job that never wanted CUDA.
#
# scripts/describe_corpus reads obs frames and draws bar charts.  Its import
# chain reaches endo_model.evaluation, whose __init__ is deliberately torch-free
# and whose two exports import umap and scanpy lazily via importlib, so numpy,
# pandas, anndata, h5py, matplotlib and yaml are the whole requirement.
#
# --constraint=icelake pins the same architecture the GPU nodes use, because the
# 2024a module tree is built per architecture and this is the one every other
# job in this project has been running against.
#
# If the module loads still fail, the reliable fallback is to run it as a GPU
# job with train.sh's exact module list — it wastes a card for five minutes but
# uses a stack known to work.
#
# No --include-excluded. That flag bypasses exclude_studies so a study dropped
# from the corpus still appears, which was useful while deciding whether an
# exclusion was right. zhang_2022 (chorionic villus / decidua) is settled: it is
# not endometrial tissue and is out of scope permanently, so it should not be in
# a figure describing what the atlas contains. Pass the flag by hand if you ever
# need to look at it again.

set -euo pipefail

CONFIG="${1:-configs/experiment/s1_circ_low.yaml}"

module purge
module load bear-apps/2024a
module load SciPy-bundle/2024.05-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load h5py/3.12.1-foss-2024a
module load anndata/0.10.9-foss-2024a
module load matplotlib/3.9.2-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

export PYTHONUNBUFFERED=1
python -u -c "import anndata, pandas, numpy, matplotlib, yaml; print('imports ok')"
python -u -m scripts.describe_corpus --config "${CONFIG}" --root .
