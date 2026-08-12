"""Download the structural hierarchy files (login-node, has internet).

Gene sets are NOT downloaded here — they are reused from the existing
gene_sets.json.  Only the small structural files are needed:

    ReactomePathwaysRelation.txt   parent<TAB>child edges
    ReactomePathways.txt           id<TAB>name<TAB>species  (labels)
    go-basic.obo                   GO ontology (is_a / part_of)
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_REACTOME_BASE = "https://reactome.org/download/current"
_GO_OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"

HIERARCHY_FILES = {
    "ReactomePathwaysRelation.txt": f"{_REACTOME_BASE}/ReactomePathwaysRelation.txt",
    "ReactomePathways.txt": f"{_REACTOME_BASE}/ReactomePathways.txt",
    "go-basic.obo": _GO_OBO_URL,
}


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading %s → %s", url, dest.name)
    req = urllib.request.Request(url, headers={"User-Agent": "ELAJ-hierarchy-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    logger.info("  saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def save_hierarchy_files(out_dir: str | Path) -> None:
    """Download all hierarchy structure files into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, url in HIERARCHY_FILES.items():
        _download(url, out / filename)
    logger.info("Hierarchy files saved to %s", out)
