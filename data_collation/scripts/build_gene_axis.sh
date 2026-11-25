#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --job-name=elaj_gene_axis
#SBATCH --output=logs/build_gene_axis_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Builds the gene axis the BASELINES read: every measured gene, nothing
# dropped.
#
#   sbatch data_collation/scripts/build_gene_axis.sh
#
# WHY NOT build_vocabulary_wide.sh, WHICH WAS SUPPOSED TO DO THIS. It ran with
# DROP_UNASSIGNED=0 and still produced 15,496 genes of the 30,962 in the gene
# list. Job 57175038's log says why:
#
#   union_cut(reactome): 10114 genes -> 28 groups
#   union_cut(go_bp):     5382 genes -> 16 groups
#   ... 15466 dropped (no hierarchy)
#
# UnionCutStrategy discards any gene with no Reactome or GO hierarchy entry,
# which is half the transcriptome, and --drop-unassigned governs a different
# code path. The filter is in the grouping, so the fix is not to pass the
# grouping pipeline a different flag: it is not to use it. The baselines never
# read the groups.
#
# THIS VOCABULARY MUST NEVER TRAIN A MODEL. Its groups are consecutive chunks
# of the gene list, which exist only because GeneVocabulary requires a group
# per gene. ELAJ's group encoder attends within a group, and these groups are
# accidents of file order. Hence the separate name: vocabulary_axis.json, not
# vocabulary_*.json like the ones runs are trained against.
#
# Minutes and 4G: it reads a text file and writes a json. No corpus is opened.

set -euo pipefail

PROCESSED=${ELAJ_BASE}/data/processed

module purge
module load bear-apps/2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

cd "$ELAJ_REPO/src"

python -u -m scripts.build_gene_axis \
    --gene-list "${PROCESSED}/gene_list.txt" \
    --out "${PROCESSED}/vocabulary_axis.json" \
    "$@"

echo ""
echo "Done. Pass it as --baseline-vocabulary to benchmark_baselines.sh,"
echo "or as --vocabulary to embed_pretrained.py."
