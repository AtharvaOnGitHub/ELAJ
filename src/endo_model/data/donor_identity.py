"""One person, one donor id, however many studies name them.

THE DEFECT THIS CLOSES. `garcia_alonso_2021_b` is a strict subset of `_a`:
all fifteen `source_sample_id` values shared, every one of its 10,729
barcodes present in `_a`, and ZERO `patient_id` values in common, because
CELLxGENE mints a donor UUID per dataset. Fifteen people carried thirty
identifiers.

`assign_splits` already forwards an already-assigned patient so they keep one
split, but it keys on the id string, so two ids for one person slip past it
and are split independently. Seven of the fifteen landed on opposite sides of
the wall, putting 5,026 cells into training AND into scoring, and leaving
8,905 more in held-out splits belonging to donors the model had partly seen.

THE RULE. Two donor ids that share a `source_sample_id` are the same person.
Sharing is transitive: if A and B share a sample and B and C share another,
all three are one donor. Every alias resolves to one canonical id, the
splitter sees one donor, and the person lands in exactly one split.

WHY SAMPLE ID AND NOT CELL BARCODE. A sample is the material taken from a
person on one occasion, so two datasets describing it are describing that
person. Barcodes would also work for an exact duplicate like garcia, but
would miss a study that re-processed the same sample into different cells.

TRAIN WINS, AND IT COSTS NOTHING. Because each canonical donor is assigned
once, there is no precedence rule to apply afterwards: a person cannot be in
train under one alias and in test under another. Cells that were about to be
orphaned by such a conflict simply follow their donor.

Pure bookkeeping over id strings. No numpy, no torch, no I/O.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DonorAliases:
    """Resolves the donor ids that name one person to a single canonical id.

    Built from `(patient_id, sample_id)` pairs across every study, so it sees
    the whole corpus before deciding anything. Two ids sharing any sample are
    one donor, and that relation is closed transitively.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._by_sample: dict[str, str] = {}

    def add(self, patient_id: str, sample_id: str | None) -> None:
        """Record that `patient_id` contributed `sample_id`.

        A missing or blank sample id links nothing: a study without the
        column keeps every donor distinct, which is the safe default.
        """
        patient = str(patient_id)
        self._find(patient)
        if sample_id is None:
            return
        sample = str(sample_id).strip()
        if not sample or sample.lower() in ("nan", "none", "na", ""):
            return
        seen = self._by_sample.get(sample)
        if seen is None:
            self._by_sample[sample] = patient
        else:
            self._union(seen, patient)

    def add_study(self, patient_ids, sample_ids=None) -> None:
        """One study's per-cell ids. `sample_ids` None links nothing."""
        if sample_ids is None:
            for patient in patient_ids:
                self.add(patient, None)
            return
        for patient, sample in zip(patient_ids, sample_ids):
            self.add(patient, sample)

    def canonical(self, patient_id: str) -> str:
        """The single id this person is known by."""
        return self._find(str(patient_id))

    def resolve_all(self, patient_ids) -> list[str]:
        """`canonical` over a sequence, in order."""
        return [self._find(str(p)) for p in patient_ids]

    def aliased(self) -> dict[str, list[str]]:
        """`{canonical: [every id for that person]}`, for ids with more than one.

        The report, and the thing to log: a corpus with no aliases should say
        so out loud rather than leaving the reader to assume it was checked.
        """
        groups: dict[str, list[str]] = defaultdict(list)
        for patient in self._parent:
            groups[self._find(patient)].append(patient)
        return {k: sorted(v) for k, v in groups.items() if len(v) > 1}

    def describe(self) -> str:
        """One line for the run log."""
        groups = self.aliased()
        if not groups:
            return ("Donor identity: no two donor ids share a sample; every "
                    "id is its own person.")
        merged = sum(len(v) for v in groups.values())
        return (f"Donor identity: {merged} donor ids resolve to "
                f"{len(groups)} people. Those people are split once, so no "
                f"one can be in train under one id and held out under "
                f"another.")

    # ── union-find over id strings ─────────────────────────────────────────
    #
    # Transitivity is the whole reason this is union-find rather than a dict:
    # three ids chained through two different samples are still one person,
    # and a flat mapping would resolve only the pairs it saw directly.

    def _find(self, patient: str) -> str:
        root = self._parent.setdefault(patient, patient)
        while root != self._parent[root]:
            self._parent[root] = self._parent[self._parent[root]]
            root = self._parent[root]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        # The lexicographically smaller id wins, so the canonical name is
        # stable across runs and the split is reproducible.
        first, second = sorted((ra, rb))
        self._parent[second] = first
