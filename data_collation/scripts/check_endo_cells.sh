#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=check_endo_cells
#SBATCH --output=logs/check_endo_cells_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

module purge
module load bear-apps/2024a
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source ${ELAJ_BASE}/envs/elaj/bin/activate

python - <<'PYEOF'
import os
from pathlib import Path
import anndata as ad

PROCESSED = Path(os.environ["ELAJ_BASE"]) / "data" / "processed"

totals = {}
for path in sorted(PROCESSED.glob("*.h5ad")):
    adata = ad.read_h5ad(path)
    counts = adata.obs["disease_ontology_term_id"].value_counts()
    print(f"\n[{path.stem}]  ({adata.n_obs:,} cells total)")
    for term, n in counts.items():
        print(f"  {term}: {n:,}")
        totals[term] = totals.get(term, 0) + n

print("\n" + "=" * 50)
print("CORPUS TOTAL")
print("=" * 50)
grand_total = sum(totals.values())
for term, n in sorted(totals.items(), key=lambda x: -x[1]):
    pct = 100 * n / grand_total
    print(f"  {term}: {n:,}  ({pct:.1f}%)")
print(f"\n  Total: {grand_total:,} cells")
PYEOF
