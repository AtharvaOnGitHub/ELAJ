#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=check_h5ad
#SBATCH --output=logs/check_h5ad_%j.log

# Usage: sbatch scripts/check_h5ad.sh <study_name>
# Example: sbatch scripts/check_h5ad.sh duempelmann_2026

STUDY="${1:?Usage: sbatch scripts/check_h5ad.sh <study_name>}"
H5AD="/rds/projects/a/acharjea-multi-omics/Atharva/data/processed/${STUDY}.h5ad"

module purge
module load bear-apps/2024a
module load anndata/0.10.9-foss-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

python - <<EOF
import anndata as ad

path = "${H5AD}"
print(f"Loading {path} ...")
adata = ad.read_h5ad(path)
print(adata)

print("\n--- required ELAJ fields ---")
for col in ("patient_id", "source_sample_id",
            "tissue_ontology_term_id", "disease_ontology_term_id"):
    if col in adata.obs.columns:
        vals = adata.obs[col].unique()[:5].tolist()
        print(f"  {col}: {vals}")
    else:
        print(f"  {col}: MISSING")

print("\n--- X matrix ---")
print(f"  nnz={adata.X.nnz}, max={adata.X.max():.1f}, dtype={adata.X.dtype}")
print("\nOK")
EOF
