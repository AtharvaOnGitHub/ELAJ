#!/bin/bash
# The software environment every reported run used on BlueBEAR, the University
# of Birmingham's cluster. Source it in a GPU job (it needs the CUDA module
# tree, which only GPU nodes carry):
#
#   source "$ELAJ_REPO/src/setup_bluebear.sh"
#
# CPU-only jobs cannot load the CUDA PyTorch module; they use a bare Python
# module and their own venv instead (see data_collation/scripts/benchmark_cpu.sh).
# Elsewhere, use setup.sh (conda + pip) instead.

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load R/4.5.0-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load scanpy/1.9.8-foss-2024a
module load h5py/3.12.1-foss-2024a
module load pytest/8.3.3-GCCcore-13.3.0
module load Pillow/10.4.0-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a
export R_LIBS_USER="${ELAJ_BASE:?source site.env first}/R_libs"
export R_MAX_VSIZE=180Gb
export PYTHONNOUSERSITE=1
source "${ELAJ_BASE}/envs/elaj/bin/activate"
