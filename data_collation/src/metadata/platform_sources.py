"""Fetch per-sample platform metadata from the archives that hold it.

Nothing in the corpus records which machine or which chemistry produced a cell,
and that is the largest single source of technical variation between studies:
10x 3' v2 and v3 differ several-fold in genes detected per cell, and the
CellRanger reference decides which genes exist at all.

Three archives, three shapes:

  GEO           SOFT text, `!Sample_*` lines, one block per GSM
  ArrayExpress  SDRF, a tab-separated table with named protocol columns
  CELLxGENE     `assay_ontology_term_id` already inside the h5ad

**Raw text is kept verbatim.** Chemistry is not a field anywhere — it is a
sentence inside a free-text protocol, written differently by every lab
("Next GEM 3' v3.1", "Chromium Single Cell 3' Reagent Kit v2", "5' scRNA-seq").
Normalising during collection would bake one reading of that sentence into the
record with no way to check it later. So this writes what the archive says, and
a separate pass decides what it means.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

GEO_SOFT = ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            "?acc={acc}&targ={targ}&form=text&view=brief")
SDRF = "https://www.ebi.ac.uk/biostudies/files/{acc}/{acc}.sdrf.txt"

# The GEO fields worth keeping. instrument_model and platform_id are structured;
# the two protocol fields are prose and are where chemistry actually hides.
GEO_FIELDS = (
    "title", "instrument_model", "library_strategy", "library_selection",
    "description", "extract_protocol_ch1", "data_processing", "characteristics_ch1",
)


@dataclass
class SampleRecord:
    """One sample's platform metadata, as the archive stated it."""

    study: str
    accession: str
    sample_id: str
    fields: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str:
        return self.fields.get(key, "")


def fetch(url: str, retries: int = 3, pause: float = 2.0) -> str:
    """GET a URL as text, retrying transient failures.

    NCBI throttles; a bare failure here would silently drop a whole study from
    the table, so a failed fetch raises rather than returning empty.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:  # noqa: PERF203
            last = exc
            logger.warning("  fetch failed (%s), retry %d/%d",
                           exc, attempt + 1, retries)
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def parse_geo_samples(soft: str, study: str, accession: str) -> list[SampleRecord]:
    """Split a SOFT document into one record per GSM.

    A `!Sample_*` line can repeat — characteristics especially — so repeats are
    joined with ' | ' rather than overwriting, which would silently keep only
    the last of several donor attributes.
    """
    records: list[SampleRecord] = []
    current: SampleRecord | None = None

    for line in soft.splitlines():
        if line.startswith("^SAMPLE"):
            current = SampleRecord(study=study, accession=accession,
                                   sample_id=line.split("=", 1)[1].strip())
            records.append(current)
            continue
        if current is None or not line.startswith("!Sample_"):
            continue
        key, _, value = line[len("!Sample_"):].partition("=")
        key, value = key.strip(), value.strip()
        if key not in GEO_FIELDS:
            continue
        current.fields[key] = (
            f"{current.fields[key]} | {value}" if key in current.fields else value
        )
    return records


def geo_series_platform(accession: str) -> str:
    """The series-level GPL id, which names the sequencer."""
    soft = fetch(GEO_SOFT.format(acc=accession, targ="self"))
    match = re.search(r"!Series_platform_id\s*=\s*(\S+)", soft)
    return match.group(1) if match else ""


def from_geo(study: str, accession: str) -> list[SampleRecord]:
    """Every sample in a GEO series, with its platform fields."""
    logger.info("GEO %s (%s)", accession, study)
    records = parse_geo_samples(
        fetch(GEO_SOFT.format(acc=accession, targ="gsm")), study, accession
    )
    platform = geo_series_platform(accession)
    for record in records:
        record.fields["series_platform_id"] = platform
    logger.info("  %d samples", len(records))
    return records


def from_arrayexpress(study: str, accession: str) -> list[SampleRecord]:
    """Every row of an ArrayExpress SDRF, keeping the protocol columns.

    The SDRF is one row per assay with free-form column names, so the columns
    are matched by keyword rather than by exact name — they differ between
    submissions.
    """
    logger.info("ArrayExpress %s (%s)", accession, study)
    lines = fetch(SDRF.format(acc=accession)).splitlines()
    if not lines:
        return []

    header = lines[0].split("\t")
    wanted = {
        i: name for i, name in enumerate(header)
        if re.search(r"instrument|platform|library construction|protocol|"
                     r"technology|sequencing|source name|assay name",
                     name, re.I)
    }

    records = []
    for row in lines[1:]:
        cells = row.split("\t")
        if len(cells) < len(header):
            continue
        fields = {header[i].strip(): cells[i].strip() for i in wanted}
        records.append(SampleRecord(
            study=study, accession=accession,
            sample_id=cells[0].strip(), fields=fields,
        ))
    logger.info("  %d assays", len(records))
    return records
