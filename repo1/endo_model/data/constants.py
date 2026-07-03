"""Global constants for the ELAJ data pipeline.

These values are used throughout the codebase.  Import from here rather than
defining magic numbers inline.
"""

DNS_SENTINEL: int = -1
"""Sentinel stored in the raw count tensor wherever a gene was not measured.
A gene with DNS_SENTINEL was not in the study's sequencing panel — it is
epistemologically distinct from a measured zero expression value.
DNS positions are excluded from reconstruction loss but included in encoding."""

N_TISSUE_LEVELS: int = 4
"""Four-level tissue hierarchy: compartment → organ → tissue_type → microsite."""

N_DISEASE_STATUSES: int = 4
"""Integer codes: 0=eutopic, 1=ectopic, 2=control, 3=other."""

DISEASE_STATUS_TO_INT: dict[str, int] = {
    "eutopic": 0,
    "ectopic": 1,
    "control": 2,
    "other": 3,
}

INT_TO_DISEASE_STATUS: dict[int, str] = {v: k for k, v in DISEASE_STATUS_TO_INT.items()}
