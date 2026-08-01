#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --job-name=reprocess_study
#SBATCH --output=logs/reprocess_%x_%j.log

# Re-runs the processing pipeline for a single study, overwriting its processed
# h5ad.  Use after changing that study's block in configs/datasets.yaml.
#
#   sbatch --job-name=duempelmann scripts/reprocess_study.sh duempelmann_2026
#
# The memory request is sized for duempelmann_2026 (466k cells, 25k genes, from
# a 38.9 GB RDS).  Smaller studies will not use it, but a login-node run of this
# is what gets killed — the pipeline holds the full matrix dense at several
# points.
#
# --skip-download: raw files are already on disk.
# --force:         the processed h5ad exists and would otherwise be skipped.

set -euo pipefail

STUDY="${1:?usage: sbatch scripts/reprocess_study.sh <study_name>}"

module purge
module load bear-apps/2024a
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/data_collation

echo "Reprocessing ${STUDY} ..."
python scripts/run_pipeline.py \
    --only "${STUDY}" \
    --skip-download \
    --force

echo ""
echo "Done. Verifying disease_status and disease_ontology_term_id ..."
STUDY="${STUDY}" python - <<'PYEOF'
import os
from pathlib import Path
import anndata as ad

path = Path(
    "/rds/projects/a/acharjea-multi-omics/Atharva/data/processed"
) / f"{os.environ['STUDY']}.h5ad"

adata = ad.read_h5ad(path, backed="r")
print(adata.obs["disease_status"].value_counts(), "\n")
print(adata.obs["disease_ontology_term_id"].value_counts(), "\n")
print(adata.obs.groupby("disease_status", observed=True)["patient_id"].nunique())
adata.file.close()
PYEOF
