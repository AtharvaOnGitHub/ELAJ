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

Three of those judgements are worth knowing:

* **marecko_2024** — the SDRF says `10x 3' v1` for all 103 assays; the paper
  says `Chromium Single Cell 3' Kit v.3.0 and v.3.1`. The paper wins. The same
  submission splits 35 scRNA-seq against 68 snRNA-seq, so its material is
  genuinely mixed and a per-study code cannot separate the two.
* **li_2025** — BD Rhapsody, a microwell assay, not a droplet one. It is the
  only non-10x study in the corpus.
* **fonseca_2023** — GEO records `3'` with no version across all 51 samples,
  and the Nature Genetics methods are not in the paper folder. Encoded as an
  explicit "version unstated" category rather than guessed at.

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

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._by_study: dict[str, PlatformRecord] = {}
        self._load()

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
        code seen, so index `max` is valid and 0 stays reserved for unknown."""
        return tuple(
            max((r.codes[i] for r in self._by_study.values()), default=0) + 1
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
