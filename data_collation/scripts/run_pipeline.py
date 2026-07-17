#!/usr/bin/env python3
"""Data collation pipeline entry point.

Usage:
    python scripts/run_pipeline.py [--config configs/pipeline.yaml]
                                   [--datasets configs/datasets.yaml]
                                   [--only STUDY_NAME [STUDY_NAME ...]]
                                   [--skip-download]
                                   [--force]

Options:
    --only          Process only the named studies (by study_name)
    --skip-download Assume raw files are already present; skip all downloads
    --force         Re-process even if the output .h5ad already exists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `import src.*` from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.dataset_config import DataSource, LoaderType, load_dataset_configs
from src.config.pipeline_config import PipelineConfig
from src.download.atlas import ReproductiveAtlasDownloader
from src.download.cellxgene import CellxGeneDownloader
from src.download.figshare import FigshareDownloader
from src.download.geo import GEODownloader
from src.load.h5_loader import H5PerSampleLoader
from src.load.h5ad_loader import H5adLoader
from src.load.mtx_loader import MTXPerSampleLoader
from src.load.rds_loader import RDSLoader
from src.process.gene_standardiser import GeneStandardiser
from src.process.metadata_attacher import MetadataAttacher
from src.process.qc import QCFilter
from src.process.raw_extractor import RawCountExtractor
from src.utils.logging import configure_logging, get_logger
from src.utils.patient_registry import PatientRegistry
from src.write.h5ad_writer import H5adWriter
from src.write.manifest import ManifestWriter

logger = get_logger(__name__)

# ── downloader registry ────────────────────────────────────────────────────────
_DOWNLOADERS = {
    DataSource.GEO: GEODownloader,
    DataSource.CELLXGENE: CellxGeneDownloader,
    DataSource.REPRODUCTIVE_ATLAS: ReproductiveAtlasDownloader,
    DataSource.FIGSHARE: FigshareDownloader,
    DataSource.MANUAL: None,   # no-op: user places files manually
}

# ── loader registry ───────────────────────────────────────────────────────────
_LOADERS = {
    LoaderType.H5_PER_SAMPLE: H5PerSampleLoader,
    LoaderType.MTX_PER_SAMPLE: MTXPerSampleLoader,
    LoaderType.H5AD: H5adLoader,
    LoaderType.RDS: RDSLoader,
}


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).parent.parent
    cfg = PipelineConfig.from_yaml(args.config).resolve(str(root))
    configure_logging(cfg.logging.level, cfg.paths.pipeline_log)

    datasets = load_dataset_configs(args.datasets)
    if args.only:
        datasets = [d for d in datasets if d.study_name in args.only]
        if not datasets:
            logger.error("No datasets matched --only filter: %s", args.only)
            sys.exit(1)

    registry = PatientRegistry(cfg.paths.patient_id_map)
    standardiser = GeneStandardiser(cfg.paths.gene_aliases, cfg.paths.gene_mapping_issues, cfg.paths.hgnc_table)
    attacher = MetadataAttacher(registry)
    qc_filter = QCFilter(cfg.qc)
    extractor = RawCountExtractor()
    writer = H5adWriter()
    manifest = ManifestWriter()

    processed_dir = Path(cfg.paths.processed_dir)
    raw_dir_base = Path(cfg.paths.raw_dir)

    completed: list = []

    for dataset in datasets:
        study_name = dataset.study_name
        out_path = processed_dir / f"{study_name}.h5ad"

        if out_path.exists() and not args.force:
            logger.info("[%s] output already exists, skipping (use --force to re-run)", study_name)
            completed.append(dataset)
            continue

        logger.info("=" * 60)
        logger.info("[%s] starting", study_name)

        raw_dir = raw_dir_base / (dataset.accession or dataset.dataset_id or study_name)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Download ──────────────────────────────────────────────────────
        if not args.skip_download:
            downloader_cls = _DOWNLOADERS.get(dataset.source)
            if downloader_cls is None:
                logger.info("[%s] source=manual — expecting pre-placed files in %s", study_name, raw_dir)
            else:
                try:
                    downloader_cls().download(dataset, raw_dir)
                except Exception as exc:
                    logger.error("[%s] download failed: %s — skipping", study_name, exc)
                    continue

        # ── 2. Load ──────────────────────────────────────────────────────────
        loader_cls = _LOADERS[dataset.loader]
        try:
            adata = loader_cls().load(dataset, raw_dir)
        except Exception as exc:
            logger.error("[%s] load failed: %s — skipping", study_name, exc)
            continue

        # ── 3. Extract raw counts ────────────────────────────────────────────
        adata = extractor.extract(adata)

        # ── 4. Standardise gene names ────────────────────────────────────────
        adata = standardiser.standardise(adata, study_name)

        # ── 5. Attach metadata ───────────────────────────────────────────────
        adata = attacher.attach(adata, dataset, raw_dir)

        # ── 6. QC ────────────────────────────────────────────────────────────
        adata = qc_filter.filter(adata, study_name)

        # ── 7. Write .h5ad ───────────────────────────────────────────────────
        writer.write(adata, out_path)

        completed.append(dataset)
        logger.info("[%s] done", study_name)

    # ── Save registry & manifest ─────────────────────────────────────────────
    registry.save()
    manifest.write(completed, processed_dir, Path(cfg.paths.studies_csv))
    logger.info("Pipeline complete. %d / %d studies processed.", len(completed), len(datasets))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ELAJ data collation pipeline")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--datasets", default="configs/datasets.yaml")
    p.add_argument("--only", nargs="+", metavar="STUDY")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
