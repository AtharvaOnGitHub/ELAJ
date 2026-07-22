#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=check_all_studies
#SBATCH --output=logs/check_all_studies_%j.log

module purge
module load bear-apps/2024a
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/data_collation

python - <<'PYEOF'
import sys
from pathlib import Path

import anndata as ad
import numpy as np

PROCESSED = Path("/rds/projects/a/acharjea-multi-omics/Atharva/data/processed")
REQUIRED_COLS = [
    "patient_id",
    "source_sample_id",
    "tissue_ontology_term_id",
    "disease_ontology_term_id",
]

h5ad_files = sorted(PROCESSED.glob("*.h5ad"))
if not h5ad_files:
    print("No .h5ad files found in", PROCESSED)
    sys.exit(1)

print(f"Found {len(h5ad_files)} studies in {PROCESSED}\n")
print("=" * 70)

failures = []
for path in h5ad_files:
    study = path.stem
    study_ok = True
    print(f"\n[{study}]")
    try:
        adata = ad.read_h5ad(path)
        n_cells, n_genes = adata.shape
        print(f"  shape          : {n_cells:,} cells × {n_genes:,} genes")

        # Required columns
        missing = [c for c in REQUIRED_COLS if c not in adata.obs.columns]
        if missing:
            print(f"  MISSING cols   : {missing}")
            study_ok = False
        else:
            print(f"  required cols  : all present")

        # Null check on required columns
        for col in REQUIRED_COLS:
            if col not in adata.obs.columns:
                continue
            n_null = adata.obs[col].isna().sum()
            n_unique = adata.obs[col].nunique()
            if n_null > 0:
                print(f"  WARN {col}: {n_null:,} nulls / {n_unique} unique")
                study_ok = False
            else:
                print(f"  {col}: {n_unique} unique values — {adata.obs[col].unique()[:3].tolist()}")

        # X matrix
        nnz = adata.X.nnz
        x_max = float(adata.X.max())
        print(f"  X              : nnz={nnz:,}, max={x_max:.1f}, dtype={adata.X.dtype}")
        if nnz == 0:
            print(f"  ERROR: X is empty")
            study_ok = False
        if x_max <= 0:
            print(f"  ERROR: X has no positive values")
            study_ok = False

        print(f"  status         : {'OK' if study_ok else 'ISSUES FOUND'}")

    except Exception as exc:
        print(f"  ERROR: {exc}")
        study_ok = False

    if not study_ok:
        failures.append(study)

print("\n" + "=" * 70)
print(f"\nSummary: {len(h5ad_files) - len(failures)}/{len(h5ad_files)} studies passed")
if failures:
    print(f"Failed : {failures}")
    sys.exit(1)
else:
    print("All studies OK")
PYEOF
