#!/bin/bash
# Push a locally downloaded model checkpoint to the HPC. Runs on YOUR MAC.
#
#   bash data_collation/scripts/push_weights.sh scfoundation ~/Downloads/models.ckpt
#   bash data_collation/scripts/push_weights.sh scgpt ~/Downloads/scGPT_human
#
# ONLY NEEDED WHEN THE LOGIN NODE CANNOT FETCH THE WEIGHTS ITSELF. Both
# scGPT and scFoundation distribute canonically through links that need a
# browser session -- Google Drive and SharePoint -- and install_foundation.sh
# works around that with third-party HuggingFace mirrors. A mirror is fine for
# getting results and is recorded by sha256, but if you want CANONICAL
# provenance in the paper, download from the authors' own link and push it
# here: the install scripts leave an existing checkpoint alone, so this
# overrides the mirror.
#
# Second argument is a FILE for scfoundation (models.ckpt) or a DIRECTORY for
# scgpt (best_model.pt, vocab.json, args.json together).

set -euo pipefail

MODEL="${1:?usage: push_weights.sh <scfoundation|scgpt|geneformer|scimilarity> <local path>}"
SOURCE="${2:?second argument: the local file or directory to push}"
REMOTE=axw1008@bear-pg-login06.bear.cluster
MODELS=/rds/projects/a/acharjea-multi-omics/Atharva/models

case "${MODEL}" in
    scfoundation) DEST="${MODELS}/scFoundation/model/models/" ;;
    scgpt)        DEST="${MODELS}/scGPT/" ;;
    geneformer)   DEST="${MODELS}/Geneformer/" ;;
    scimilarity)  DEST="${MODELS}/SCimilarity/" ;;
    *) echo "unknown model '${MODEL}'" >&2; exit 2 ;;
esac

if [[ ! -e "${SOURCE}" ]]; then
    echo "no such path: ${SOURCE}" >&2
    exit 1
fi

echo "── sha256 before transfer (RECORD THIS) ──"
if [[ -d "${SOURCE}" ]]; then
    find "${SOURCE}" -type f -exec shasum -a 256 {} \;
else
    shasum -a 256 "${SOURCE}"
fi

echo
echo "── pushing to ${REMOTE}:${DEST} ──"
ssh "${REMOTE}" "mkdir -p '${DEST}'"
# A trailing slash on a directory source copies its CONTENTS, which is what
# scgpt wants; a file source copies the file. --partial so a dropped
# connection on 1.4 GB resumes rather than restarting.
rsync -avh --progress --partial "${SOURCE}$([[ -d "${SOURCE}" ]] && echo /)" \
    "${REMOTE}:${DEST}"

echo
echo "── sha256 on the cluster, which must match above ──"
ssh "${REMOTE}" "cd '${DEST}' && sha256sum \$(basename '${SOURCE}') 2>/dev/null || sha256sum *"
