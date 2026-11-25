#!/bin/bash
#SBATCH --job-name=elaj_vocab_v1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=2:0:0
#SBATCH --mem=120G
#SBATCH --qos=bbdefault
#SBATCH --output=logs/build_vocabulary_v1_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# The v1 vocabulary: the c1 corpus's own gene list, built correctly.
#
# WHY A SECOND SCRIPT AND NOT A FLAG. This one names its output
# vocabulary_v1.json and never touches vocabulary.json, which every c-series
# checkpoint was trained against and which a running job may be reading. The
# base script refuses to overwrite an existing file anyway; this makes the
# intent explicit at the call site.
#
# WHAT CHANGES. The excluded studies now come from corpus_v1.yaml rather than
# a typed list that had drifted from it. Measured by
# scripts/dry_run_vocabulary.py: eleven genes gained, ten lost. The gene
# content is all but identical; the PARTITION moves, because UnionCutStrategy's
# merge order is sensitive to the gene list. That is the reason v runs are
# their own series.
#
# The 500-gene ceiling is now enforced rather than preferred (see
# data/grouping/union_cut.py), so expect one more group than the dry run
# reported: the 503 it produced splits back into a 249 and a 254.

export CORPUS=${ELAJ_REPO}/src/configs/data/corpus_v1.yaml
export GENE_LIST=${ELAJ_BASE}/data/processed/gene_list_v1.txt
export VOCAB_OUT=${ELAJ_BASE}/data/processed/vocabulary_v1.json

# Through `bash`, not `exec` on the path: sbatch READS a job script rather
# than executing it, so nothing in this directory has ever needed the
# executable bit and build_vocabulary.sh does not have it. Invoking the
# interpreter explicitly makes the file mode irrelevant.
exec bash ${ELAJ_REPO}/data_collation/scripts/build_vocabulary.sh
