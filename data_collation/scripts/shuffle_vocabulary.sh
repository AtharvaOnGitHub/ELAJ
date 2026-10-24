#!/bin/bash
#SBATCH --job-name=elaj_shuffle_vocab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:0
#SBATCH --mem=8G
#SBATCH --account=acharjea-multi-omics
#SBATCH --qos=bbdefault
#SBATCH --output=logs/shuffle_vocabulary_%j.log

# Builds the shuffled vocabulary the C-1 ablation (a1_randgroups) reads.
#
#   sbatch data_collation/scripts/shuffle_vocabulary.sh
#
# A1's launcher submits this and makes a1_randgroups depend on it, so the
# arm cannot start against a vocabulary that does not exist.
#
# bbdefault AND NO GPU, deliberately. This permutes a 15,496-entry JSON and
# writes it back; it takes seconds. Putting it on bbgpu would spend one of
# the account's two GPU slots on a dictionary shuffle, and those two slots
# are what the training arms are waiting for. It needs no torch, so the
# reason run_tests.sh has to sit on a GPU node does not apply here.
#
# NOT ON THE LOGIN NODE either, small as it is — the rule is the rule, and a
# fifteen-minute slot on bbdefault costs nothing.
#
# IDEMPOTENT BY REFUSAL. If the output exists the script stops rather than
# rewriting it: a1_randgroups may already have trained against the file, and
# silently regenerating it with a different seed would make that run
# unreproducible while leaving every log looking fine. Delete it by hand if
# a rebuild is actually wanted.

set -euo pipefail

VOCAB_IN=/rds/projects/a/acharjea-multi-omics/Atharva/data/processed/vocabulary_v1.json
VOCAB_OUT=/rds/projects/a/acharjea-multi-omics/Atharva/data/processed/vocabulary_v1_shuffled.json
SEED=24

module purge
module load bear-apps/2024a
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
cd /rds/homes/a/axw1008/axw1008/src

if [ ! -f "${VOCAB_IN}" ]; then
    echo "source vocabulary missing: ${VOCAB_IN}" >&2
    exit 1
fi
if [ -f "${VOCAB_OUT}" ]; then
    echo "${VOCAB_OUT} already exists — leaving it alone."
    echo "Delete it by hand if you mean to rebuild; a1_randgroups may have"
    echo "already trained against this exact file."
    exit 0
fi

export PYTHONUNBUFFERED=1
python -u -m scripts.shuffle_vocabulary \
    --in "${VOCAB_IN}" --out "${VOCAB_OUT}" --seed "${SEED}"
