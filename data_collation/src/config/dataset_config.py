from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

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

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
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
        )


def load_dataset_configs(path: str) -> List[DatasetConfig]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return [DatasetConfig.from_dict(d) for d in raw.get("datasets", [])]
