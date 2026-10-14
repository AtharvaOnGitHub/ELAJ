#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=06:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --job-name=elaj_embed_pre
#SBATCH --output=logs/embed_pretrained_%j.log

# Embeds our val+test cells with a pretrained foundation model, zero-shot, and
# writes an ExternalLatent for benchmark_baselines.sh to score.
#
#   sbatch data_collation/scripts/embed_pretrained.sh geneformer
#
# THE MODEL'S OWN VENV, NOT elaj. That is the whole reason this is a separate
# job: Geneformer wants transformers and datasets that no module provides, and
# its resolver must not move scvi-tools' pins under a live campaign. Staged by
# install_geneformer.sh, which runs on a login node because compute nodes have
# no outbound internet.
#
# THE WIDE GENE AXIS, DELIBERATELY. vocabulary.json holds 15,496 genes of the
# 30,962 measured, because UnionCutStrategy drops any gene with no Reactome or
# GO hierarchy entry. vocabulary_axis.json is the full axis, built by
# build_gene_axis.sh, whose groups are meaningless chunks -- which is fine,
# because no baseline reads them. Handing that to a
# competitor gives it our curation, which is the asymmetry we found in the scVI
# comparison on 12 October 2026. Geneformer reads its own 25,424-gene
# vocabulary regardless; the only question is how much of it we offer, and the
# answer is everything we measured. Coverage goes in the manifest.
#
# SIZING, AND IT IS A GUESS. No run on record. 120G covers the val+test corpus
# on the wide axis (the 15,496-gene inductive benchmark peaked at 168.80 GB,
# but that was dominated by the 1.16M-cell FIT corpus, which this job does not
# build) plus the h5ad and the tokenised dataset on disk. Six hours covers
# tokenising 416,256 cells and a forward pass at batch 64; the model itself is
# 41 MB and 10M parameters, so the GPU is not the constraint, tokenisation is.
# Read the log and correct these numbers here afterwards.
#
# THE SCRATCH FILES ARE KEPT. --work holds the h5ad, the tokenised dataset and
# the embedding csv, and when a run goes wrong those three are the evidence.
# They are large; delete them once the manifest looks right.

set -euo pipefail

MODEL="${1:-geneformer}"
RUN="${2:-m6_drain}"
PROCESSED=/rds/projects/a/acharjea-multi-omics/Atharva/data/processed
MODELS=/rds/projects/a/acharjea-multi-omics/Atharva/models
ENVS=/rds/projects/a/acharjea-multi-omics/Atharva/envs

case "${MODEL}" in
    geneformer) REPO="${MODELS}/Geneformer"  ;;
    scgpt)      REPO="${MODELS}/scGPT"        ;;
    scimilarity) REPO="${MODELS}/SCimilarity" ;;
    scfoundation) REPO="${MODELS}/scFoundation" ;;
    *) echo "unknown model '${MODEL}'; known: geneformer scgpt scimilarity scfoundation" >&2; exit 2 ;;
esac
VENV="${ENVS}/fm_${MODEL}"

OUT="outputs/benchmarks/latents/${MODEL}"
# On the project filesystem, not /scratch, which this script has never
# verified exists on BlueBEAR. The tokenised dataset for 416,256 cells at
# input size 2048 is several GB, so it does not belong in the repository.
WORK="/rds/projects/a/acharjea-multi-omics/Atharva/scratch/${MODEL}_${SLURM_JOB_ID:-manual}"

# NO BEAR PYTHON MODULES. The venv built by install_foundation.sh carries its
# own torch and everything else, precisely so nothing on the module path can
# shadow a pip install. Only a bare python is loaded.
module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found" >&2; exit 1; }

export PYTHONNOUSERSITE=1
source "${VENV}/bin/activate"

cd /rds/homes/a/axw1008/axw1008/src

nvidia-smi
echo ""
echo "Model:      ${MODEL}"
echo "Repository: ${REPO}"
echo "Run:        ${RUN}"
echo "Gene axis:  ${PROCESSED}/vocabulary_axis.json"
mkdir -p "${WORK}"
echo "Work:       ${WORK}"
echo "Out:        ${OUT}"
echo ""

export PYTHONUNBUFFERED=1
python -u -m scripts.embed_pretrained \
    --run "${RUN}" \
    --model "${MODEL}" \
    --repo "${REPO}" \
    --vocabulary "${PROCESSED}/vocabulary_axis.json" \
    --work "${WORK}" \
    --nproc 8 \
    --batch-size 64 \
    --root . \
    --out "${OUT}"

echo ""
echo "Done. Score it with:"
echo "  sbatch data_collation/scripts/benchmark_baselines.sh ${RUN} --no-elaj \\"
echo "      --external-latent ${OUT} --methods pca --out outputs/benchmarks/pre_${MODEL}"
