#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=build_vocabulary
#SBATCH --output=logs/build_vocabulary_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Builds data/processed/vocabulary.json from the union of all genes across
# the processed h5ad studies.  Using the union (not intersection) is correct
# because the model uses a DNS sentinel for genes not measured in a given
# study's panel.
#
# The excluded studies are READ FROM THE CORPUS CONFIG rather than typed here.
# They used to be typed, and the two drifted: the vocabulary in use was built
# with wang_2022 excluded while corpus_c1 trains on it, so the 1,791 genes
# only wang measures have no vocabulary row and are never read. Meanwhile
# garcia_alonso_2021_c was excluded from the corpus and not from the build.
# One source of truth removes the whole class of mistake.
#
# WHY EXCLUDE AT ALL. A gene contributed only by an excluded study sits in
# the vocabulary as DNS in every study that actually trains: a wasted
# embedding row, a wasted decoder output, and a dispersion parameter that
# never receives a gradient.
#
# NOTE the larger filter, which is NOT this one. --drop-unassigned below
# discards every gene that matched no curated pathway, which is half the
# transcriptome: 15,495 genes kept of 30,962 measured. That is deliberate,
# because the group encoder needs every gene to sit in a coherent group,
# but it means the vocabulary is the pathway-annotated transcriptome and
# the Methods section has to say so.

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load R/4.5.0-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load Pillow/10.4.0-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a
export R_LIBS_USER=${ELAJ_BASE}/R_libs
export R_MAX_VSIZE=180Gb

export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

export PROCESSED=${ELAJ_BASE}/data/processed
export GENE_LIST=${GENE_LIST:-${PROCESSED}/gene_list.txt}
export VOCAB_OUT=${VOCAB_OUT:-${PROCESSED}/vocabulary.json}

# A vocabulary in use is read by every running job and by every checkpoint
# that was trained against it, so overwriting one silently breaks both.
# Write to a new path, or set OVERWRITE=1 deliberately.
if [[ -e "${VOCAB_OUT}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "REFUSING: ${VOCAB_OUT} exists." >&2
  echo "Write to a new path (VOCAB_OUT=...) or set OVERWRITE=1." >&2
  exit 1
fi
# Which corpus this vocabulary is for. Its exclude_studies is the one that
# counts, so point this at the config the runs actually load.
export CORPUS=${CORPUS:-${ELAJ_REPO}/src/configs/data/corpus_c1.yaml}

# Derived, never typed, and derived so that a FAILURE STOPS THE BUILD.
#
# The first version ran `python - <<EOF` inside $(...), which discards the exit
# status. PyYAML was missing from the module block above, the traceback went to
# stderr, EXCLUDE came out empty, and the build used all thirteen studies while
# printing "Excluding: (nothing)". An exclusion list that cannot be read and a
# corpus that excludes nothing must never look the same.
EXCLUDE_FILE=$(mktemp)
trap 'rm -f "${EXCLUDE_FILE}"' EXIT
if ! python ${ELAJ_REPO}/src/scripts/corpus_exclusions.py \
        "${CORPUS}" > "${EXCLUDE_FILE}"; then
  echo "FATAL: could not read exclude_studies from ${CORPUS}." >&2
  echo "Refusing to build a vocabulary from an unknown study set." >&2
  exit 1
fi
export EXCLUDE=$(tr '\n' ' ' < "${EXCLUDE_FILE}")
echo "Corpus ${CORPUS}"
echo "Excluding: ${EXCLUDE:-(nothing, and the config says so)}"

# ── Step 1: extract gene union from all processed h5ad files ──────────────
echo "Extracting gene union from processed studies ..."
python - <<'PYEOF'
import os
from pathlib import Path
import anndata as ad

processed = Path(os.environ["PROCESSED"])
gene_list_path = Path(os.environ["GENE_LIST"])

excluded = set(os.environ.get("EXCLUDE", "").split())

all_genes: set[str] = set()
for path in sorted(processed.glob("*.h5ad")):
    if path.stem in excluded:
        print(f"  {path.stem}: SKIPPED (in EXCLUDE)")
        continue
    adata = ad.read_h5ad(path, backed="r")
    genes = set(adata.var_names)
    print(f"  {path.stem}: {len(genes):,} genes")
    all_genes.update(genes)
    adata.file.close()

sorted_genes = sorted(all_genes)
gene_list_path.write_text("\n".join(sorted_genes) + "\n")
print(f"\nUnion: {len(sorted_genes):,} unique genes → {gene_list_path}")
PYEOF

# ── Step 2: assign genes to pathway groups → vocabulary.json ──────────────
# gene_sets.json is pre-downloaded on a login node (which has internet access)
# using:
#   python src/scripts/build_vocabulary.py --save-gene-sets src/data/gene_sets.json
# Compute nodes are firewalled, so we read from disk instead.
#
# Group COUNT is the dial on model size: the encoder builds one transformer per
# group, ~25k params at d_token=32.  --min-group-size is the control — it floors
# group size, so the group count falls out as D / (~2 x floor).  A ceiling
# (--cut tree) cannot do this: descending a node promotes all its children at
# once, so most groups land far below the bound.
#
# --max-group-size is a separate concern from model size.  The group encoder
# attends over a group's genes, so activation memory is quadratic in group size:
# GO:0008150 rolled up 3869 genes on the first build, which is 61 GB of
# attention at batch 256 for that one group.  The ceiling trims that tail.
# Keep it at >= 2x --min-group-size: the floor pass accumulates a merged group
# until it clears the floor, so merged groups can reach roughly twice it.
#
# DROP_UNASSIGNED defaults to 1, which is what every c-, m- and v-series
# checkpoint was trained against. Set it to 0 to keep the genes that matched no
# curated pathway, which gives the full measured transcriptome rather than its
# pathway-annotated half: 30,962 genes instead of 15,495. That vocabulary is
# NOT for training, because the group encoder's attention is sized to coherent
# groups and the unassigned chunks are not; it is the gene axis the BASELINES
# read, via benchmark_baselines.py --baseline-vocabulary.
DROP_UNASSIGNED=${DROP_UNASSIGNED:-1}
DROP_FLAG=()
if [[ "${DROP_UNASSIGNED}" == "1" ]]; then
  DROP_FLAG=(--drop-unassigned)
fi

echo ""
echo "Building vocabulary by Reactome -> GO hierarchical roll-up ..."
echo "  drop-unassigned: ${DROP_UNASSIGNED}"
cd "$ELAJ_REPO/src"
python scripts/build_vocabulary.py \
    --gene-list "${GENE_LIST}" \
    --gene-sets-file data/gene_sets.json \
    --hierarchy-dir data/hierarchy \
    --grouping hierarchical \
    --cut union \
    --min-group-size 250 \
    --max-group-size 500 \
    --go-groups 4 \
    "${DROP_FLAG[@]}" \
    --out "${VOCAB_OUT}"

echo ""
echo "Done. Vocabulary written to ${VOCAB_OUT}"
