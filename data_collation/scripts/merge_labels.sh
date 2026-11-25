#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=elaj_merge
#SBATCH --output=logs/merge_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Regroup a label and re-read its geometry, from statistics already written.
#
#   sbatch data_collation/scripts/merge_labels.sh \
#       --stats outputs/validate/p1_dab168_kl1_full --list
#
#   sbatch data_collation/scripts/merge_labels.sh \
#       --stats outputs/validate/p1_dab168_kl1_full \
#       --label "menstrual phase" \
#       --group proliferative=proliferative_early,proliferative,proliferative_late \
#       --group secretory=secretory_early,secretory_early_mid,secretory_mid,secretory_late \
#       --keep periovulatory --keep menstrual
#
# ── SBATCH, NOT THE LOGIN NODE ────────────────────────────────────────────
# An earlier version of this wrapper ran on the login node on the grounds
# that it is "only arithmetic". It is more than that: 2,000 permutations of a
# 254 x 254 matrix is about 130 million operations, and the login nodes are
# shared infrastructure that this project has already been warned about. The
# cost of being wrong about "it is small" falls on everyone else using the
# cluster.
#
# bbdefault, 4 cores, 16 GB, 30 minutes. That is outside the bbgpu cap of
# cpu=36,gres/gpu=2, so it does not displace a GPU job and can sit beside two
# of them.
#
# ── MOST OF THE TIME YOU DO NOT NEED THIS ─────────────────────────────────
# validate_latent.py already applies every grouping declared in
# evaluation/groupings.py and writes the results to grouped_distances.csv in
# the same pass -- including, where a judgement is open, EVERY variant of it
# rather than one chosen. Read that file first. This script is for a grouping
# nobody had thought of when the pass ran.
#
# It needs validate_latent to have finished: it reads level_statistics.csv and
# four companions out of that pass's --out directory, and says so rather than
# guessing if they are absent.

set -euo pipefail

module purge
module load bear-apps/2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

cd "$ELAJ_REPO/src"

export PYTHONUNBUFFERED=1
python -u -m scripts.merge_labels "$@"
