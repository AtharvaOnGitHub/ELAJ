"""Rewrite a vocabulary with its genes reassigned to groups at random.

    python -m scripts.shuffle_vocabulary --in  vocabulary_v1.json \
                                         --out vocabulary_v1_shuffled.json

CPU, seconds. This is the only artefact the C-1 ablation needs.

WHAT C-1 ASKS. The architecture's claim is that the Reactome and GO grouping
carries biology: attention inside a pathway is cheap BECAUSE the genes in a
pathway are the ones worth comparing. The cost argument of the Methods holds
for any partition of the vocabulary -- 45 groups of the same sizes cost the
same arithmetic whatever is in them. So the cost half of the claim is not in
question and cannot be ablated. What can be ablated is the biology half: if a
random partition of the same shape trains to the same numbers, the pathway
annotations were doing nothing and the architecture is a cost trick rather
than an inductive bias.

WHAT IS AND IS NOT HELD FIXED. Group SIZES are preserved exactly, gene by
gene, which keeps the parameter count, the bucketing, the padding, the
attention cost and the effective gene space all identical to the base. Each
gene keeps its `global_idx`, so the count matrix is untouched and no data has
to be rebuilt. The ONLY thing that changes is which genes sit in a group
together. That is the single variable, and it is the one the claim is about.

WHY NOT SHUFFLE THE GENE IDENTITIES INSTEAD. Permuting identity embeddings
would break the correspondence between a column of the count matrix and the
gene it belongs to, which breaks reconstruction rather than ablating a claim.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger("shuffle_vocabulary")


def shuffle_groups(raw: dict[str, dict], seed: int) -> dict[str, dict]:
    """Reassign genes to groups at random, preserving every group's size.

    The permutation is applied to the LIST of group slots rather than to the
    genes, so sizes are exact by construction rather than by rejection: a slot
    list holding each group name as many times as that group has members is
    shuffled once and dealt back out.
    """
    genes = sorted(raw)                       # sorted so the seed is the only
    rng = np.random.default_rng(seed)         # source of variation

    slots: list[str] = []
    for gene in genes:
        slots.append(raw[gene]["group"])
    rng.shuffle(slots)

    out: dict[str, dict] = {}
    position = Counter()
    for gene, group in zip(genes, slots):
        entry = dict(raw[gene])               # copy: global_idx and the rest
        entry["group"] = group                # survive untouched
        entry["group_idx"] = position[group]
        position[group] += 1
        out[gene] = entry
    return out


def sizes(raw: dict[str, dict]) -> dict[str, int]:
    c = Counter(entry["group"] for entry in raw.values())
    return dict(sorted(c.items()))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dest", required=True)
    p.add_argument("--seed", type=int, default=24)
    args = p.parse_args()

    raw = json.loads(Path(args.src).read_text(encoding="utf-8"))
    out = shuffle_groups(raw, args.seed)

    before, after = sizes(raw), sizes(out)
    assert before == after, "group sizes changed; the ablation would not be one variable"
    assert {g: raw[g]["global_idx"] for g in raw} == \
           {g: out[g]["global_idx"] for g in out}, "a global_idx moved"

    moved = sum(1 for g in raw if raw[g]["group"] != out[g]["group"])
    logger.info("%s genes, %d groups, sizes identical", f"{len(raw):,}", len(before))
    logger.info("%s genes changed group (%.1f%%)", f"{moved:,}", 100 * moved / len(raw))
    # A permutation that happened to leave most genes put would be a weak
    # ablation, so the number is printed rather than assumed.
    Path(args.dest).write_text(json.dumps(out), encoding="utf-8")
    logger.info("wrote %s", args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
