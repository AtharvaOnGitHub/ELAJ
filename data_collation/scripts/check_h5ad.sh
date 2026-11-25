#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=check_h5ad
#SBATCH --output=logs/check_h5ad_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Usage: sbatch scripts/check_h5ad.sh <study_name>
# Example: sbatch scripts/check_h5ad.sh duempelmann_2026

STUDY="${1:?Usage: sbatch scripts/check_h5ad.sh <study_name>}"
H5AD="${ELAJ_BASE}/data/processed/${STUDY}.h5ad"

module purge
module load bear-apps/2024a
module load anndata/0.10.9-foss-2024a

export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

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
