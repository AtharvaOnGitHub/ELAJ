#!/bin/bash
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"
# One-time: a clean venv per foundation model, with NO BEAR Python modules.
#
#   bash data_collation/scripts/install_foundation.sh geneformer
#   bash data_collation/scripts/install_foundation.sh scgpt
#   bash data_collation/scripts/install_foundation.sh scimilarity
#   bash data_collation/scripts/install_foundation.sh scfoundation
#
# Login node, once per model. Safe to re-run. FRESH=1 rebuilds the venv.
#
# WHY NO MODULES. pip cannot uninstall a package that lives in a BEAR module,
# and the module path sits ahead of the venv, so installing a newer regex or
# protobuf changes nothing at import and breaks whatever asked for it. That
# cost 13 October. Here nothing is loaded but a bare python, pip installs its
# own torch (the wheels bundle their CUDA runtime and need only the driver),
# and there is nothing to shadow. install_baselines.sh keeps the module-based
# approach because the elaj venv has to interoperate with our own code.
#
# Each model gets its own venv because their pins conflict with each other.

set -euo pipefail

MODEL="${1:?usage: install_foundation.sh <geneformer|scgpt|scimilarity|scfoundation>}"
MODELS=${ELAJ_BASE}/models
ENVS=${ELAJ_BASE}/envs
VENV="${ENVS}/fm_${MODEL}"

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1

if [[ "${FRESH:-0}" == "1" && -d "${VENV}" ]]; then
    echo "── FRESH=1: removing ${VENV} ──"
    rm -rf "${VENV}"
fi
[[ -d "${VENV}" ]] || python -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install --quiet --upgrade pip wheel setuptools

echo "── torch (pip's own, cu124) ──"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124

echo "── shared: what our corpus code needs ──"
# endo_model.data.dataset and the CorpusMatrix extractor, nothing more.
pip install --quiet "numpy<2" scipy pandas anndata h5py pyyaml tqdm "huggingface_hub>=0.23"

case "${MODEL}" in
  geneformer)
    # V1-10M, Genecorpus-30M, June 2021. transformers 4.x because the V1
    # checkpoint is a plain BertModel and 5.x changed the loading path.
    #
    # WITH ITS DEPENDENCIES, not --no-deps. The first attempt used --no-deps to
    # avoid module shadowing, and in a clean venv there is nothing to shadow,
    # so geneformer's import of matplotlib simply failed. Letting pip resolve
    # its real dependency list ends that whack-a-mole; only numpy and torch are
    # held down, because numpy 2 breaks pyarrow and our own corpus code, and a
    # resolver free to reinstall torch would discard the cu124 build above.
    printf 'numpy<2\ntorch==%s\npyarrow<17\n' \
        "$(python -c 'import torch; print(torch.__version__)')" > "${VENV}/constraints.txt"
    pip install --quiet "transformers>=4.40,<5" "datasets<4" \
        -c "${VENV}/constraints.txt"
    pip install --quiet "${MODELS}/Geneformer" -c "${VENV}/constraints.txt"
    CHECK="geneformer transformers datasets"
    ;;
  scgpt)
    # FROM MAIN, NOT PyPI, AND THAT IS THE WHOLE DIFFICULTY. The released
    # scgpt on PyPI imports torchtext, which was discontinued and is ABI-locked
    # to one torch (0.16.2 <-> torch 2.1.2), so `pip install scgpt` against
    # torch 2.6 either fails or silently downgrades torch and discards the
    # cu124 build. Main dropped torchtext and allows torch>=1.13 with no upper
    # bound.
    #
    # --no-deps, because main's own pin set is hostile and none of it is needed
    # by the one function we call: scvi-tools<1.0 drags an old jax, and
    # pyarrow>=22 wants numpy 2 while our corpus code is tested against 1.26.
    # What embed_data actually imports is installed explicitly below, and the
    # verify step imports that function rather than the package.
    printf 'numpy<2\ntorch==%s\n' \
        "$(python -c 'import torch; print(torch.__version__)')" > "${VENV}/constraints.txt"
    pip install --quiet --no-deps "git+https://github.com/bowang-lab/scGPT.git"
    pip install --quiet -c "${VENV}/constraints.txt" \
        scanpy einops numba scikit-misc umap-learn leidenalg "datasets<4" \
        "pyarrow<17" torchmetrics ipython
    # SATISFY THE IMPORT, NOT THE TREE. scgpt/tasks/__init__.py imports every
    # task, so reaching embed_data drags in modules for perturbation and GRN
    # work we never call. Those are installed --no-deps: present enough to
    # import, without pulling scvi-tools<1.0 and the old jax behind it.
    pip install --quiet --no-deps cell-gears scib || true
    # No flash-attn, deliberately: scGPT lists it in an optional `fa2` group,
    # embed_data is called with use_fast_transformer=False, and building it
    # against a pip torch is a day we do not have.

    # WEIGHTS, AND A PROVENANCE CAVEAT FOR THE METHODS SECTION. scGPT
    # distributes the whole-human checkpoint by Google Drive, which a login
    # node cannot fetch, so this is a THIRD-PARTY HuggingFace mirror and the
    # sha256 of best_model.pt is printed below. If the canonical file is ever
    # checked, that is what it is checked against. Already-present files are
    # left alone, so rsyncing the canonical three from a laptop overrides this.
    # The continual-pretrained checkpoint is NOT fetched: more pretraining data
    # means more overlap with our donors.
    mkdir -p "${MODELS}/scGPT"
    if [[ -f "${MODELS}/scGPT/best_model.pt" ]]; then
        echo "  weights already present; not re-fetching"
    else
        python - "${MODELS}/scGPT" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="MohamedMabrouk/scGPT", repo_type="model",
                         local_dir=sys.argv[1],
                         allow_patterns=["best_model.pt", "vocab.json",
                                         "args.json"])
print(f"  downloaded to {path}")
PYEOF
    fi
    echo "  sha256 of best_model.pt (RECORD THIS for the Methods section):"
    sha256sum "${MODELS}/scGPT/best_model.pt"
    CHECK="scgpt"
    ;;
  scimilarity)
    # Zero-shot cell embedding is what SCimilarity is FOR, so there is a
    # documented call and nothing to improvise: align_dataset to the model's
    # gene order, lognorm_counts, get_embeddings.
    printf 'numpy<2\ntorch==%s\n' \
        "$(python -c 'import torch; print(torch.__version__)')" > "${VENV}/constraints.txt"
    pip install --quiet scimilarity -c "${VENV}/constraints.txt"

    # WHICH ARCHIVE, BECAUSE ZENODO OFFERS THREE AND THEY DIFFER BY 260x.
    #
    #   record 15729925  scimilarity.tar.gz          116.5 MB  encoder only
    #   record 8286452   annotation_model_v1.tar.gz    6.8 GB   API v0.1.0
    #   record 10685499  model_v1.1.tar.gz            30.3 GB   API v0.2.0
    #
    # The 30.3 GB bundle is weights PLUS 23 million reference cell embeddings
    # and kNN graphs, which CellQuery needs and CellEmbedding never touches.
    # Downloading it to compute embeddings was my mistake: at the ~63 kB/s
    # Zenodo was serving on 14 October it was a five-day transfer.
    #
    # So the encoder-only archive is the default. Override with
    # SCIM_URL=... if it turns out to lack something CellEmbedding reads; the
    # 6.8 GB annotation model is the next step up, and it pairs with
    # scimilarity 0.1.x rather than the current release.
    SCIM_URL="${SCIM_URL:-https://zenodo.org/records/15729925/files/scimilarity.tar.gz?download=1}"
    mkdir -p "${MODELS}/SCimilarity"
    if compgen -G "${MODELS}/SCimilarity/*/" > /dev/null; then
        echo "  model directory already present; not re-fetching"
    else
        echo "  fetching ${SCIM_URL}"
        # -C - resumes a partial file rather than starting again, which matters
        # when Zenodo is slow. --speed-limit/--speed-time abandons a transfer
        # stalled under 10 kB/s for two minutes instead of hanging for days.
        curl -L --fail --retry 5 --retry-delay 10 -C - \
             --speed-limit 10240 --speed-time 120 \
             -o "${MODELS}/SCimilarity/model.tar.gz" "${SCIM_URL}"
        tar -xzf "${MODELS}/SCimilarity/model.tar.gz" -C "${MODELS}/SCimilarity"
        rm -f "${MODELS}/SCimilarity/model.tar.gz"
    fi
    echo "  extracted tree (CellEmbedding needs the encoder and a gene order):"
    find "${MODELS}/SCimilarity" -maxdepth 3 | head -30
    CHECK="scimilarity"
    ;;
  scfoundation)
    # THEIR ENTRY POINT IS A SCRIPT, so the repository is cloned for
    # get_embedding.py and OS_scRNA_gene_index.19264.tsv rather than pip
    # installed. Nothing of theirs is imported; we run it as documented.
    printf 'numpy<2\ntorch==%s\n' \
        "$(python -c 'import torch; print(torch.__version__)')" > "${VENV}/constraints.txt"
    pip install --quiet -c "${VENV}/constraints.txt" \
        einops local-attention scanpy

    mkdir -p "${MODELS}"
    if [[ -d "${MODELS}/scFoundation/.git" ]]; then
        echo "  repository present; fetching"
        git -C "${MODELS}/scFoundation" pull --quiet || true
    else
        git clone --quiet --depth 1 \
            https://github.com/biomap-research/scFoundation.git \
            "${MODELS}/scFoundation"
    fi

    # WEIGHTS, AND THE SAME PROVENANCE CAVEAT AS scGPT. The canonical link is
    # a SharePoint folder, which needs a browser session and so cannot be
    # fetched from a login node. genbio-ai/scFoundation mirrors models.ckpt
    # (1.43 GB, ungated) on HuggingFace. The sha256 is printed below and
    # belongs in the Methods section. An existing file is left alone, so
    # rsyncing the canonical checkpoint overrides this.
    CKPT_DIR="${MODELS}/scFoundation/model/models"
    mkdir -p "${CKPT_DIR}"
    if [[ -f "${CKPT_DIR}/models.ckpt" ]]; then
        echo "  checkpoint already present; not re-fetching"
    else
        python - "${CKPT_DIR}" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
path = hf_hub_download("genbio-ai/scFoundation", "models.ckpt",
                       local_dir=sys.argv[1])
print(f"  downloaded to {path}")
PYEOF
    fi
    echo "  sha256 of models.ckpt (RECORD THIS for the Methods section):"
    sha256sum "${CKPT_DIR}/models.ckpt"
    CHECK="scfoundation"
    ;;
  *) echo "unknown model '${MODEL}'; known: geneformer scgpt scimilarity scfoundation" >&2; exit 2 ;;
esac

echo
echo "── verifying (imports, not pip's opinion) ──"
python - "$CHECK" "${MODELS}/scFoundation" <<'PYEOF'
import importlib, sys
ok = True
for pkg in ("numpy", "pandas", "anndata", "torch", *sys.argv[1].split()):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:<16} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print(f"  {pkg:<16} MISSING ({type(exc).__name__}: {exc})")
# THE ENTRY POINT, NOT JUST THE PACKAGE. `import scgpt` succeeding says
# nothing about whether scgpt.tasks.embed_data can be reached: that module
# pulls in the model, tokeniser and preprocessor, and a missing dependency
# three levels down is exactly what --no-deps risks.
if "scgpt" in sys.argv[1].split():
    # Walked in stages, because "No module named X" does not say WHERE in the
    # chain X was wanted, and --no-deps makes that the only question worth
    # answering. Each stage names the next package to add.
    for label, target in (("scgpt", "scgpt"),
                          ("scgpt.model", "scgpt.model"),
                          ("scgpt.tokenizer", "scgpt.tokenizer"),
                          ("scgpt.preprocess", "scgpt.preprocess"),
                          ("scgpt.tasks", "scgpt.tasks"),
                          ("scgpt.tasks.cell_emb", "scgpt.tasks.cell_emb")):
        try:
            importlib.import_module(target)
            print(f"  {label:<20} ok")
        except Exception as exc:                               # noqa: BLE001
            ok = False
            print(f"  {label:<20} FAILS ({type(exc).__name__}: {exc})")
            break
    else:
        from scgpt.tasks import embed_data            # noqa: F401
        print("  embed_data           importable")
if "scfoundation" in sys.argv[1].split():
    # NOT an import: their entry point is a script, so what matters is that
    # the three files exist and the checkpoint is not a 1 kB error page.
    import pathlib
    model_dir = pathlib.Path(sys.argv[2]) / "model"
    for rel, floor in (("get_embedding.py", 1_000),
                       ("OS_scRNA_gene_index.19264.tsv", 100_000),
                       ("models/models.ckpt", 100_000_000)):
        path = model_dir / rel
        if not path.is_file():
            print(f"  {rel:<34} MISSING")
            ok = False
        elif path.stat().st_size < floor:
            print(f"  {rel:<34} TOO SMALL ({path.stat().st_size} bytes)")
            ok = False
        else:
            print(f"  {rel:<34} ok ({path.stat().st_size / 1e6:.1f} MB)")
if "scimilarity" in sys.argv[1].split():
    try:
        from scimilarity import CellEmbedding                  # noqa: F401
        from scimilarity.utils import align_dataset, lognorm_counts  # noqa: F401
        print("  scimilarity      CellEmbedding and utils importable")
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print(f"  scimilarity      UNREACHABLE ({type(exc).__name__}: {exc})")
if "geneformer" in sys.argv[1].split():
    try:
        from geneformer import EmbExtractor, TranscriptomeTokenizer  # noqa: F401
        print("  geneformer       EmbExtractor and tokeniser importable")
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print(f"  geneformer       UNREACHABLE ({type(exc).__name__}: {exc})")

import torch
print(f"  cuda available   {torch.cuda.is_available()}  (False on a login node is normal)")
print(f"  torch cuda build {torch.version.cuda}")
raise SystemExit(0 if ok else 1)
PYEOF

echo
echo "Done. Environment: ${VENV}"
