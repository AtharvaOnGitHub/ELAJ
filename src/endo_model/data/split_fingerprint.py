"""Which patients a split held, recorded so a checkpoint cannot be scored
against a different split in silence.

The train/val/test boundary is a patient wall: no donor appears on both sides.
The wall is not stored anywhere, though. It is recomputed from the corpus every
time a dataset is built, by stratifying patients over study and tissue. So the
wall moves whenever the thing it stratifies on moves, and a fix to the corpus is
exactly that kind of change. Commit 482dc69 gave garcia_alonso its real tissue
levels, the strata changed, and the split changed with them:

    before the fix   study 6 validation = 20,002 cells, study 7 = 1,528
    after the fix    study 6 validation =  5,027 cells, study 7 = 1,792

Study 7 gained validation cells, and they came from patients that had been on
the training side. `c1_combined` was trained before the fix and scored after it,
so some of the cells it was scored on were cells it had learnt. Every number
that run produced is optimistic by an unknown amount.

Nothing about that failure is visible in the output. A contaminated score is a
plausible number in the right range, only better, which is the direction that
gets believed. There is no downstream check that can catch it, because a
representation that has memorised a cell and a representation that generalises
to it look the same from the outside. The only place to catch it is here, by
writing down which patients were held out and refusing a mismatch.

The digest covers the patient ids and the split name, not the cells. Cell counts
shift for reasons that are not leakage: a changed gene vocabulary drops cells to
different depths, and a column repair can make a cell usable that was not. What
must not change is the set of donors on the far side of the wall.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PAYLOAD_KEY = "split_fingerprint"
DIGEST_LENGTH = 16


class SplitMismatch(RuntimeError):
    """A checkpoint was about to be scored against a split it did not train on."""


@dataclass(frozen=True)
class SplitFingerprint:
    """The identity of one split: its name, its donors, and a digest of both."""

    split: str
    n_patients: int
    digest: str

    @classmethod
    def of(cls, dataset) -> "SplitFingerprint":
        """Fingerprint a built dataset.

        Reads `patient_ids` and `split`, which every dataset the trainer and the
        scorers build exposes.
        """
        return cls.of_patients(dataset.split, dataset.patient_ids)

    @classmethod
    def of_patients(cls, split: str, patients) -> "SplitFingerprint":
        """Fingerprint a split from its donor ids directly.

        Separate from `of` so a fingerprint can be built in a test, or from a
        record, without a dataset.
        """
        unique = sorted({str(patient) for patient in patients})
        if not unique:
            raise ValueError(
                f"split {split!r} has no patient ids, so it cannot be "
                f"fingerprinted; scoring it could not be checked"
            )
        body = "\n".join([str(split), *unique]).encode("utf-8")
        return cls(split=str(split), n_patients=len(unique),
                   digest=hashlib.sha256(body).hexdigest()[:DIGEST_LENGTH])

    def as_record(self) -> dict:
        """The form stored in a checkpoint payload."""
        return {"split": self.split, "n_patients": self.n_patients,
                "digest": self.digest}

    @classmethod
    def from_record(cls, record: dict) -> "SplitFingerprint":
        """Rebuild a fingerprint from a stored record."""
        return cls(split=str(record["split"]),
                   n_patients=int(record["n_patients"]),
                   digest=str(record["digest"]))

    def matches(self, other: "SplitFingerprint") -> bool:
        return self.digest == other.digest

    def describe(self) -> str:
        return f"{self.split}: {self.n_patients} donors, {self.digest}"


class SplitGuard:
    """Refuses to score a checkpoint against a split it did not train on."""

    def __init__(self, allow_unrecorded: bool = True) -> None:
        self.allow_unrecorded = bool(allow_unrecorded)

    def check(self, payload: dict, dataset) -> SplitFingerprint | None:
        """Compare the split in hand against the one the checkpoint recorded.

        Returns the recorded fingerprint, or None when the checkpoint predates
        the record. Raises `SplitMismatch` when the two disagree.
        """
        here = SplitFingerprint.of(dataset)
        record = (payload or {}).get(PAYLOAD_KEY)
        if not record:
            return self._unrecorded(here)
        there = SplitFingerprint.from_record(record)
        if not here.matches(there):
            raise SplitMismatch(self._mismatch_message(here, there))
        logger.info("Split provenance OK, %s", here.describe())
        return there

    def _unrecorded(self, here: SplitFingerprint) -> None:
        """A checkpoint saved before fingerprints existed."""
        message = (
            f"This checkpoint records no split fingerprint, so the split in "
            f"hand ({here.describe()}) cannot be confirmed to be the one it "
            f"trained on. If the corpus or the tissue columns changed since it "
            f"ran, its scores may include cells it trained on."
        )
        if not self.allow_unrecorded:
            raise SplitMismatch(message)
        logger.warning("UNVERIFIED SPLIT. %s", message)
        return None

    @staticmethod
    def _mismatch_message(here: SplitFingerprint,
                          there: SplitFingerprint) -> str:
        return (
            f"Split mismatch: this checkpoint trained with {there.describe()}, "
            f"and the split built here is {here.describe()}. Scoring it would "
            f"measure it partly on cells it learnt, which reads as a better "
            f"model and not as an error. Re-train on the current split, or "
            f"score with the split the checkpoint was trained on."
        )
