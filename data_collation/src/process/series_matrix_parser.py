from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

_STRIP_QUOTES = re.compile(r'^"|"$')


class SeriesMatrixParser:
    """Parse a GEO series matrix .txt.gz file without GEOparse.

    Reads the file line by line and collects per-sample metadata from:
      - !Sample_geo_accession  → GSM IDs (column order)
      - !Sample_title          → 'title' column
      - !Sample_source_name_ch1 → 'source_name' column
      - !Sample_characteristics_ch1 (all rows) → one column per key

    Each sample may have a different set of characteristics; the staggered
    format (different keys at the same row index across samples) is handled
    by splitting each cell on ': ' independently.

    Returns a DataFrame indexed by GSM accession ('source_sample_id').
    Returns an empty DataFrame if the file cannot be parsed.
    """

    def parse(self, path: Path) -> pd.DataFrame:
        gsm_ids: List[str] = []
        titles: List[str] = []
        source_names: List[str] = []
        char_rows: List[List[str]] = []

        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line.startswith("!Sample_geo_accession"):
                        gsm_ids = self._parse_values(line)
                    elif line.startswith("!Sample_title\t"):
                        titles = self._parse_values(line)
                    elif line.startswith("!Sample_source_name_ch1"):
                        source_names = self._parse_values(line)
                    elif line.startswith("!Sample_characteristics_ch1"):
                        char_rows.append(self._parse_values(line))
                    elif line.startswith("!series_matrix_table_begin"):
                        break
        except Exception as exc:
            logger.warning("Failed to read series matrix %s: %s", path.name, exc)
            return pd.DataFrame()

        if not gsm_ids:
            logger.warning("No !Sample_geo_accession found in %s", path.name)
            return pd.DataFrame()

        n = len(gsm_ids)
        sample_data: List[Dict[str, str]] = [{} for _ in range(n)]

        for i in range(n):
            if i < len(titles):
                sample_data[i]["title"] = titles[i]
            if i < len(source_names):
                sample_data[i]["source_name"] = source_names[i]

        for row_values in char_rows:
            for i, val in enumerate(row_values):
                if i >= n or not val:
                    continue
                if ": " in val:
                    key, value = val.split(": ", 1)
                    sample_data[i][key.strip()] = value.strip()

        df = pd.DataFrame(sample_data, index=gsm_ids)
        df.index.name = "source_sample_id"
        logger.info(
            "%s: parsed %d samples, %d columns from series matrix",
            path.name,
            len(df),
            len(df.columns),
        )
        return df

    # ── private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_values(line: str) -> List[str]:
        """Return tab-separated values from a series matrix row, stripped of quotes."""
        parts = line.split("\t")[1:]
        return [_STRIP_QUOTES.sub("", p.strip()) for p in parts]
