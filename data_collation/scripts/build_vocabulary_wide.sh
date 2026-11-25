#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --job-name=elaj_vocab_wide
#SBATCH --output=logs/build_vocabulary_wide_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# The WIDE vocabulary: every measured gene, not the pathway-annotated half.
#
#   sbatch data_collation/scripts/build_vocabulary_wide.sh
#
# WHY IT EXISTS. build_vocabulary.sh passes --drop-unassigned, which keeps
# 15,495 genes of the 30,962 the studies measure. The filter is there because
# ELAJ's group encoder needs every gene to sit in a coherent pathway group, so
# it is an ELAJ constraint. Handing its output to scVI gives scVI a curated
# gene set it would not have in the field, where a reader selects highly
# variable genes from everything the studies measured.
#
# THIS VOCABULARY IS NOT FOR TRAINING. The unassigned genes land in numbered
# chunks that are groups only in the sense that they fit under the size
# ceiling, and nothing about them is biologically coherent. It exists to be
# passed to benchmark_baselines.py --baseline-vocabulary, where it is read as
# a gene axis and its grouping is never used.
#
# Same gene list and the same corpus as the vocabulary in use, so the two
# differ in exactly one thing: whether the unassigned half was kept. Keep it
# that way, or the comparison acquires a second variable.

export CORPUS=${ELAJ_REPO}/src/configs/data/corpus_c1.yaml
export GENE_LIST=${ELAJ_BASE}/data/processed/gene_list.txt
export VOCAB_OUT=${ELAJ_BASE}/data/processed/vocabulary_wide.json
export DROP_UNASSIGNED=0

# Through `bash`, not `exec` on the path: sbatch READS a job script rather
# than executing it, so nothing in this directory has ever needed the
# executable bit. Invoking the interpreter explicitly makes the mode irrelevant.
exec bash ${ELAJ_REPO}/data_collation/scripts/build_vocabulary.sh
