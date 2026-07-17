from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import yaml


class DataSource(str, Enum):
    GEO = "geo"
    CELLXGENE = "cellxgene"
    REPRODUCTIVE_ATLAS = "reproductive_atlas"
    FIGSHARE = "figshare"
    MANUAL = "manual"   # pre-placed file, no download attempted


class LoaderType(str, Enum):
    H5_PER_SAMPLE = "h5_per_sample"
    MTX_PER_SAMPLE = "mtx_per_sample"
    H5AD = "h5ad"
    RDS = "rds"


@dataclass
class OntologyConfig:
    """Per-study config for mapping series-matrix values to ontology term IDs
    (tissue_ontology_term_id or disease_ontology_term_id) or disease_status.

    Resolution order per sample:
      1. cfg.column → cfg.value_map  (primary)
      2. cfg.fallback_column → cfg.fallback_value_map  (if primary misses)
      3. cfg.default  (ultimate fallback)

    For disease_status only: if cfg.override_column value is in cfg.override_map,
    the override takes precedence over steps 1-3.
    """

    column: Optional[str] = None
    value_map: Dict[str, str] = field(default_factory=dict)
    default: Optional[str] = None
    fallback_column: Optional[str] = None
    fallback_value_map: Dict[str, str] = field(default_factory=dict)
    override_column: Optional[str] = None
    override_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "OntologyConfig":
        return cls(
            column=d.get("column"),
            value_map=dict(d.get("value_map") or {}),
            default=d.get("default"),
            fallback_column=d.get("fallback_column"),
            fallback_value_map=dict(d.get("fallback_value_map") or {}),
            override_column=d.get("override_column"),
            override_map=dict(d.get("override_map") or {}),
        )


@dataclass
class DatasetConfig:
    study_name: str
    source: DataSource
    loader: LoaderType
    tissue_types: List[str]
    has_cycle_phase: bool

    # GEO
    accession: Optional[str] = None
    supplementary_metadata: Optional[str] = None  # filename within suppl files
    rds_file: Optional[str] = None

    # CELLxGENE
    dataset_id: Optional[str] = None

    # Figshare
    figshare_token: Optional[str] = None

    note: Optional[str] = None

    # Per-study ontology and status config
    tissue_ontology: Optional[OntologyConfig] = None
    disease_ontology: Optional[OntologyConfig] = None
    status_config: Optional[OntologyConfig] = None

    # Cells whose tissue_ontology_term_id is in this list are dropped post-metadata
    exclude_tissue_ontology_terms: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        def _ont(key: str) -> Optional[OntologyConfig]:
            raw = d.get(key)
            return OntologyConfig.from_dict(raw) if raw else None

        return cls(
            study_name=d["study_name"],
            source=DataSource(d["source"]),
            loader=LoaderType(d["loader"]),
            tissue_types=list(d.get("tissue_types", [])),
            has_cycle_phase=bool(d.get("has_cycle_phase", False)),
            accession=d.get("accession"),
            supplementary_metadata=d.get("supplementary_metadata"),
            rds_file=d.get("rds_file"),
            dataset_id=d.get("dataset_id"),
            figshare_token=d.get("figshare_token"),
            note=d.get("note"),
            tissue_ontology=_ont("tissue_ontology"),
            disease_ontology=_ont("disease_ontology"),
            status_config=_ont("status_config"),
            exclude_tissue_ontology_terms=list(d.get("exclude_tissue_ontology_terms") or []),
        )


def load_dataset_configs(path: str) -> List[DatasetConfig]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return [DatasetConfig.from_dict(d) for d in raw.get("datasets", [])]
