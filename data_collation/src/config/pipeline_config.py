from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    gene_aliases: str = "configs/gene_aliases.yaml"
    hgnc_table: str = "data/hgnc_complete_set.txt"
    patient_id_map: str = "data/processed/patient_id_map.csv"
    gene_mapping_issues: str = "data/processed/gene_mapping_issues.csv"
    studies_csv: str = "data/processed/studies.csv"
    pipeline_log: str = "data/processed/pipeline.log"


@dataclass
class QCConfig:
    min_umi: int = 200
    min_genes: int = 100
    max_mt_fraction: float = 0.25
    min_cells_per_gene: int = 3
    run_scrublet: bool = True
    scrublet_threshold: Optional[float] = None


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        # Expand ${VAR} in path values, so site paths live in site.env.
        if isinstance(raw, dict) and isinstance(raw.get("paths"), dict):
            raw["paths"] = {k: os.path.expandvars(v) if isinstance(v, str) else v
                            for k, v in raw["paths"].items()}
        return cls(
            paths=PathsConfig(**raw.get("paths", {})),
            qc=QCConfig(**raw.get("qc", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )

    def resolve(self, root: str = ".") -> "PipelineConfig":
        """Return a copy with all paths resolved relative to root."""
        r = Path(root)
        p = self.paths
        return PipelineConfig(
            paths=PathsConfig(
                raw_dir=str(r / p.raw_dir),
                processed_dir=str(r / p.processed_dir),
                gene_aliases=str(r / p.gene_aliases),
                hgnc_table=str(r / p.hgnc_table),
                patient_id_map=str(r / p.patient_id_map),
                gene_mapping_issues=str(r / p.gene_mapping_issues),
                studies_csv=str(r / p.studies_csv),
                pipeline_log=str(r / p.pipeline_log),
            ),
            qc=self.qc,
            logging=self.logging,
        )
