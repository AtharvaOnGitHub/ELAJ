from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, Optional, Set

from src.utils.logging import get_logger

logger = get_logger(__name__)

_HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)


class HGNCTable:
    """HGNC complete gene set providing approved symbol lookup.

    Downloads hgnc_complete_set.txt from EBI FTP on first use and caches it
    locally. Builds a mapping from alias and previous symbols (and Ensembl IDs)
    to the current approved HGNC symbol, enabling O(1) lookup for any variant.

    Conflicts — where an alias maps to more than one approved symbol — are
    silently excluded from the alias map so they fall through to mygene.info.
    """

    def __init__(self, cache_path: str) -> None:
        self._cache = Path(cache_path)
        self._approved: Set[str] = set()
        self._alias_map: Dict[str, str] = {}
        self._ensure_loaded()

    def lookup(self, name: str) -> Optional[str]:
        """Return the current approved HGNC symbol for name, or None."""
        if name in self._approved:
            return name
        return self._alias_map.get(name)

    def is_approved(self, name: str) -> bool:
        return name in self._approved

    # ── private ────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._cache.exists():
            self._download()
        self._parse()

    def _download(self) -> None:
        self._cache.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading HGNC complete set → %s", self._cache)
        try:
            subprocess.run(
                [
                    "curl", "-L", "-f", "-s",
                    "-o", str(self._cache), _HGNC_URL,
                ],
                check=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if self._cache.exists():
                self._cache.unlink()
            raise RuntimeError(
                f"Failed to download HGNC complete set from {_HGNC_URL}"
            ) from exc
        logger.info("HGNC complete set downloaded (%d bytes)", self._cache.stat().st_size)

    def _parse(self) -> None:
        conflicts: Set[str] = set()

        with open(self._cache, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                symbol = row.get("symbol", "").strip()
                if not symbol:
                    continue
                self._approved.add(symbol)

                # Alias and previous symbols → approved symbol
                for col in ("alias_symbol", "prev_symbol"):
                    for alias in row.get(col, "").split("|"):
                        alias = alias.strip()
                        if not alias or alias in self._approved:
                            continue
                        if alias in self._alias_map and self._alias_map[alias] != symbol:
                            conflicts.add(alias)
                        elif alias not in conflicts:
                            self._alias_map[alias] = symbol

                # Ensembl gene ID → approved symbol
                ensg = row.get("ensembl_gene_id", "").strip()
                if ensg and ensg not in self._approved:
                    if ensg in self._alias_map and self._alias_map[ensg] != symbol:
                        conflicts.add(ensg)
                    elif ensg not in conflicts:
                        self._alias_map[ensg] = symbol

        for c in conflicts:
            self._alias_map.pop(c, None)

        logger.info(
            "HGNC table loaded: %d approved symbols, %d alias/prev/Ensembl mappings "
            "(%d ambiguous conflicts excluded)",
            len(self._approved),
            len(self._alias_map),
            len(conflicts),
        )
