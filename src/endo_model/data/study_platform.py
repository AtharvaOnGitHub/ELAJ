"""Study name -> the technical covariates that produced its cells.

What this is for
----------------
Nothing in the model currently tells the decoder how a cell was made. Study
identity enters only through DSBN, in the *encoder*, so the latent has to carry
the technical signal itself if reconstruction is to use it — and four waves of
adversarial training have shown how badly that goes.

Giving the decoder the technical context inverts that. A decoder that knows the
cell came from 10x 3' v3 on a NovaSeq can reconstruct it from a latent that
never encoded the chemistry, which removes the reason to store it. This is the
scVI pattern, and `TissueHierarchyEncoder` is already the same shape.

Why a static table and not a scrape
-----------------------------------
The values were assembled once, by hand, from GEO protocols, an ArrayExpress
SDRF and the papers' methods sections, because no two archives record this the
same way and one of them is wrong. `data_collation/scripts/fetch_platform_metadata.py`
collects the raw evidence; this file is the adjudicated result. Re-deriving it
at training time would mean re-litigating those judgements on every run.

Two tables, sample first
------------------------
`study_platform.csv` carries one row per study. `sample_platform.csv` overrides
it for the studies that are not uniform, keyed on `source_sample_id` — the same
id the collation pipeline writes into obs, so the join is exact rather than
inferred. Only lin_2023 needs it today. Most studies are uniform and listing
every sample would be noise.

Judgements worth knowing:

* **lin_2023** — ten samples on a NovaSeq 6000, three on a HiSeq 2000 in a
  later submission batch. Split per sample; the study row is only a fallback.
* **fonseca_2023** — checked per sample: all 51 GSMs are NovaSeq 6000, so the
  sequencer needs no split. Chemistry cannot be split at all — the Nature
  Genetics methods say "3' Reagent Kits User Guide (v2 or v3)" and every GEO
  sample carries identical protocol text, so no source resolves it.
* **marecko_2024** — the SDRF says `10x 3' v1` for all 103 assays; the paper
  says `Chromium Single Cell 3' Kit v.3.0 and v.3.1`. The paper wins. The same
  submission splits 35 scRNA-seq against 68 snRNA-seq, so its material is
  genuinely mixed; splitting it needs its assay ids checked against
  source_sample_id, which has not been done.
* **li_2025** — BD Rhapsody, a microwell assay, not a droplet one. It is the
  only non-10x study in the corpus.
* **garcia_alonso_2021_b** — read from its own h5ad: obs.assay is 10x 3' v3 for
  9,654 cells and v2 for 1,075, so it is genuinely mixed. Per-cell resolution
  sits in obs and would beat this table if collation carried it through.

Codes, not names, reach the model. They are written out in the CSV rather than
derived from sorted unique values, because a derived code would silently
renumber the moment a study is added and every checkpoint trained against the
old numbering would be quietly wrong.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CODE_COLUMNS = ("assay_code", "chemistry_code", "sequencer_code", "material_code")
"""The integer columns handed to the model, in the order the encoder reads
them. 0 is reserved for unknown in every one."""

UNKNOWN = 0


@dataclass(frozen=True)
class PlatformRecord:
    """One study's technical covariates, as adjudicated."""

    study: str
    accession: str
    sequencer: str
    chemistry: str
    assay_family: str
    material: str
    codes: tuple[int, ...]
    source: str
    confidence: str
    note: str

    @property
    def is_known(self) -> bool:
        return any(c != UNKNOWN for c in self.codes)


class StudyPlatform:
    """The lookup, loaded from `configs/data/study_platform.csv`.

    Args:
        path: The CSV. Its `study` column must match the names in studies.csv.
    """

    def __init__(self, path: Path | str, sample_path: Path | str | None = None) -> None:
        self.path = Path(path)
        self.sample_path = Path(sample_path) if sample_path else None
        self._by_study: dict[str, PlatformRecord] = {}
        # (study, source_sample_id) -> codes, for the studies that are not
        # uniform. Only populated where a split is evidenced; a study absent
        # from this file resolves through _by_study, which is the common case.
        self._by_sample: dict[tuple[str, str], tuple[int, ...]] = {}
        self._load()
        if self.sample_path:
            self._load_samples()

    def _load(self) -> None:
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                study = row["study"].strip()
                self._by_study[study] = PlatformRecord(
                    study=study,
                    accession=row["accession"].strip(),
                    sequencer=row["sequencer"].strip(),
                    chemistry=row["chemistry"].strip(),
                    assay_family=row["assay_family"].strip(),
                    material=row["material"].strip(),
                    codes=tuple(int(row[c]) for c in CODE_COLUMNS),
                    source=row["source"].strip(),
                    confidence=row["confidence"].strip(),
                    note=row["note"].strip(),
                )
        known = sum(1 for r in self._by_study.values() if r.is_known)
        logger.info(
            "Study platform: %d studies from %s, %d with codes, %d unknown",
            len(self._by_study), self.path.name, known,
            len(self._by_study) - known,
        )

    def _load_samples(self) -> None:
        with self.sample_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["study"].strip(), row["source_sample_id"].strip())
                self._by_sample[key] = tuple(int(row[c]) for c in CODE_COLUMNS)
        studies = {s for s, _ in self._by_sample}
        logger.info(
            "Sample platform: %d samples across %d studies from %s",
            len(self._by_sample), len(studies), self.sample_path.name,
        )

    def codes_for_sample(self, study: str, sample_id: str) -> tuple[int, ...]:
        """One sample's codes, falling back to its study's.

        Sample first because a study is not always uniform: lin_2023 ran ten
        samples on a NovaSeq 6000 and three on a HiSeq 2000, and collapsing
        that to one "mixed" code throws away a distinction the archive states
        per sample. Study second because most studies are uniform and listing
        every sample would be noise.
        """
        codes = self._by_sample.get(
            (str(study).strip(), str(sample_id).strip())
        )
        return codes if codes is not None else self.codes_for(study)

    def codes_for(self, study: str) -> tuple[int, ...]:
        """One study's covariate codes. An unlisted study reads as all-unknown.

        Unknown rather than an error: a study added to studies.csv before
        someone has read its methods should train, with its technical context
        marked absent, not halt the run.
        """
        record = self._by_study.get(str(study).strip())
        if record is None:
            logger.warning(
                "Study %r is not in %s — its technical covariates read as "
                "unknown. Add a row once its platform is confirmed.",
                study, self.path.name,
            )
            return (UNKNOWN,) * len(CODE_COLUMNS)
        return record.codes

    def record_for(self, study: str) -> PlatformRecord | None:
        return self._by_study.get(str(study).strip())

    def sizes(self) -> tuple[int, ...]:
        """Embedding table width per code column: one more than the largest
        code seen, so index `max` is valid and 0 stays reserved for unknown.

        Spans both tables. A sample-level code outside the study-level range
        would index past the end of an embedding built from the study table
        alone, which is an IndexError on CPU and a silent read of unallocated
        memory on CUDA.
        """
        both = [r.codes for r in self._by_study.values()] + list(self._by_sample.values())
        return tuple(
            max((c[i] for c in both), default=0) + 1
            for i in range(len(CODE_COLUMNS))
        )

    def describe(self) -> str:
        """The table as one block, for a log or a thesis appendix."""
        head = (f"{'study':<22}{'sequencer':<28}{'chemistry':<24}"
                f"{'material':<24}{'confidence'}")
        lines = [head, "-" * len(head)]
        for record in sorted(self._by_study.values(), key=lambda r: r.study):
            lines.append(
                f"{record.study[:21]:<22}{record.sequencer[:27]:<28}"
                f"{record.chemistry[:23]:<24}{record.material[:23]:<24}"
                f"{record.confidence}"
            )
        return "\n".join(lines)
