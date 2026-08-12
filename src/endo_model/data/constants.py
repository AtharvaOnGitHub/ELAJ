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

TISSUE_LEVEL_SIZES: tuple[int, ...] = (8, 16, 32, 16)
"""Category count allowed at each tissue level, in level order.

These size the embedding tables in TissueHierarchyEncoder, so they are also the
valid range of the corresponding obs columns.  Kept here, with the other data
contracts, rather than beside the tables: preflight_check validates the corpus
against them and must not import torch to do it."""

N_DISEASE_STATUSES: int = 5
"""Integer codes: 0=eutopic, 1=ectopic, 2=control, 3=adenomyosis, 4=other."""

DISEASE_STATUS_TO_INT: dict[str, int] = {
    "eutopic": 0,
    "ectopic": 1,
    "control": 2,
    "adenomyosis": 3,
    "other": 4,
}
"""String -> integer code for the disease embedding table.

Must stay in step with VALID_STATUSES in
data_collation/src/process/metadata_attacher.py, which is what decides the
strings that survive into the h5ad files.  A status valid there but missing
here is silently re-coded to "other" at load time.

"adenomyosis" is its own code rather than "other" because wang_2022
(GSE218044, MONDO:0010888) is a disease contrast, not an unclassifiable
leftover -- collapsing it into "other" would put it in the same bucket as
tissue that is simply out of scope.  Note this renumbers "other" from 3 to 4:
h5ad files written before this change carry strings, not codes, so they are
unaffected, but any checkpoint trained before it is not loadable."""

INT_TO_DISEASE_STATUS: dict[int, str] = {v: k for k, v in DISEASE_STATUS_TO_INT.items()}
