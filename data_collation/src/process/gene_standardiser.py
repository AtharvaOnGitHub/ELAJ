from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import mygene
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GeneStandardiser:
    """Map adata.var.index to HGNC approved gene symbols.

    Resolution order for each gene name:
      1. Manual alias config (configs/gene_aliases.yaml) — highest priority
      2. mygene.info lookup (symbol / alias fields, human only)
      3. Identity — if the name already looks like a valid HGNC symbol
      4. Flag for manual review and drop from this study

    Genes mapped to None in the alias config are intentionally dropped.
    Unresolvable genes are written to gene_mapping_issues.csv.
    """

    def __init__(self, aliases_path: str, issues_csv_path: str) -> None:
        self._aliases: Dict[str, Optional[str]] = self._load_aliases(aliases_path)
        self._issues_path = Path(issues_csv_path)
        self._mg = mygene.MyGeneInfo()

    def standardise(self, adata: ad.AnnData, study_name: str) -> ad.AnnData:
        raw_names: List[str] = list(adata.var_names)
        mapping = self._build_mapping(raw_names, study_name)

        kept_old = [g for g in raw_names if mapping.get(g) is not None]
        kept_new = [mapping[g] for g in kept_old]

        n_dropped = len(raw_names) - len(kept_old)
        if n_dropped:
            logger.info(
                "%s: dropped %d genes (unresolvable or explicitly excluded)",
                study_name,
                n_dropped,
            )

        adata = adata[:, kept_old].copy()
        adata.var_names = kept_new
        adata.var_names_make_unique()
        return adata

    # ── private ────────────────────────────────────────────────────────────────

    def _build_mapping(
        self, raw_names: List[str], study_name: str
    ) -> Dict[str, Optional[str]]:
        mapping: Dict[str, Optional[str]] = {}
        needs_lookup: List[str] = []

        for name in raw_names:
            if name in self._aliases:
                mapping[name] = self._aliases[name]
            else:
                needs_lookup.append(name)

        if needs_lookup:
            resolved = self._mygene_lookup(needs_lookup)
            issues: List[str] = []
            for name in needs_lookup:
                hgnc = resolved.get(name)
                if hgnc is not None:
                    mapping[name] = hgnc
                elif self._looks_like_hgnc(name):
                    mapping[name] = name
                else:
                    mapping[name] = None
                    issues.append(name)

            if issues:
                self._write_issues(issues, study_name)
                logger.warning(
                    "%s: %d genes could not be mapped to HGNC symbols — "
                    "see %s for the full list",
                    study_name,
                    len(issues),
                    self._issues_path,
                )

        return mapping

    def _mygene_lookup(self, names: List[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        try:
            hits = self._mg.querymany(
                names,
                scopes="symbol,alias",
                fields="symbol",
                species="human",
                returnall=False,
                verbose=False,
            )
            for hit in hits:
                if "notfound" not in hit and "symbol" in hit:
                    result[hit["query"]] = hit["symbol"]
        except Exception as exc:
            logger.warning("mygene.info lookup failed: %s — falling back to identity", exc)
        return result

    def _looks_like_hgnc(self, name: str) -> bool:
        return bool(name and name[0].isupper() and not name.startswith("ENSG"))

    def _write_issues(self, genes: List[str], study_name: str) -> None:
        self._issues_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._issues_path.exists()
        with open(self._issues_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["study_name", "raw_gene_name"])
            if write_header:
                writer.writeheader()
            for gene in genes:
                writer.writerow({"study_name": study_name, "raw_gene_name": gene})

    @staticmethod
    def _load_aliases(path: str) -> Dict[str, Optional[str]]:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p) as f:
            raw = yaml.safe_load(f)
        return dict(raw.get("aliases") or {})
