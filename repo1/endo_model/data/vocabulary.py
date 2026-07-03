"""Gene vocabulary: loads vocabulary.json and provides fast bidirectional lookups.

vocabulary.json schema
----------------------
{
    "VEGFA": {
        "global_idx":  1847,     # unique integer 0 … vocab_size-1
        "group":       "angiogenesis",
        "group_idx":   23,       # position within the group's embedding table
        "chromosome":  "6p21.1"
    },
    ...
}

The vocabulary is IMMUTABLE at training time.  Modify and rebuild it via
scripts/build_vocabulary.py if the gene panel or group assignments change.
"""

import json
import random
from pathlib import Path
from typing import Optional


class GeneVocabulary:
    """Immutable mapping between HGNC gene symbols and their numeric indices.

    After construction all lookups are O(1) (plain dict access).
    """

    def __init__(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, dict] = json.load(f)
        self._build_from_raw(raw)

    @classmethod
    def from_dict(cls, vocab_dict: dict[str, dict]) -> "GeneVocabulary":
        """Construct directly from a Python dict — used in tests."""
        obj = cls.__new__(cls)
        obj._build_from_raw(vocab_dict)
        return obj

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _build_from_raw(self, raw: dict[str, dict]) -> None:
        self._gene_to_entry: dict[str, dict] = raw

        # Reverse mapping: global index → gene symbol
        self._idx_to_gene: dict[int, str] = {
            entry["global_idx"]: gene for gene, entry in raw.items()
        }

        # Build per-group sorted gene lists (sorted by group_idx ascending)
        groups: dict[str, list[tuple[int, str]]] = {}
        for gene, entry in raw.items():
            group = entry["group"]
            groups.setdefault(group, []).append((entry["group_idx"], gene))

        self._groups: dict[str, list[str]] = {
            group: [gene for _, gene in sorted(members)]
            for group, members in groups.items()
        }

        # Pre-compute global index lists per group (same order as _groups)
        self._group_global_idxs: dict[str, list[int]] = {
            group: [raw[gene]["global_idx"] for gene in genes]
            for group, genes in self._groups.items()
        }

        self._group_sizes: dict[str, int] = {
            group: len(genes) for group, genes in self._groups.items()
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._gene_to_entry)

    @property
    def group_names(self) -> list[str]:
        return list(self._groups.keys())

    @property
    def group_sizes(self) -> dict[str, int]:
        return self._group_sizes

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def __contains__(self, gene: str) -> bool:
        return gene in self._gene_to_entry

    def gene_to_index(self, gene: str) -> int:
        """Global vocabulary index for a gene symbol. KeyError if not in vocab."""
        return self._gene_to_entry[gene]["global_idx"]

    def index_to_gene(self, idx: int) -> str:
        """Gene symbol for a global vocabulary index. KeyError if not in vocab."""
        return self._idx_to_gene[idx]

    def gene_to_group(self, gene: str) -> str:
        return self._gene_to_entry[gene]["group"]

    def gene_to_group_index(self, gene: str) -> int:
        """Within-group index (position in the group's embedding table)."""
        return self._gene_to_entry[gene]["group_idx"]

    def gene_to_chromosome(self, gene: str) -> str:
        return self._gene_to_entry[gene]["chromosome"]

    def genes_in_group(self, group: str) -> list[str]:
        """Gene symbols for a group, ordered by group_idx."""
        return self._groups[group]

    def global_indices_for_group(self, group: str) -> list[int]:
        """Global vocabulary indices for all genes in a group, ordered by group_idx."""
        return self._group_global_idxs[group]

    def __repr__(self) -> str:
        return (
            f"GeneVocabulary(vocab_size={self.vocab_size}, "
            f"n_groups={len(self._groups)})"
        )


# ---------------------------------------------------------------------------
# Synthetic vocabulary generator (for tests and CI)
# ---------------------------------------------------------------------------


def generate_synthetic_vocabulary(
    n_genes: int = 500,
    n_groups: int = 5,
    seed: int = 24,
) -> dict[str, dict]:
    """Return a deterministic vocabulary dict compatible with GeneVocabulary.

    Genes are named GENE0000 … GENE{n_genes-1} and assigned round-robin
    to groups named group_0 … group_{n_groups-1}.

    Args:
        n_genes:  Total number of genes in the vocabulary.
        n_groups: Number of biological groups.
        seed:     Random seed — kept constant for reproducible test fixtures.
    """
    rng = random.Random(seed)
    group_names = [f"group_{i}" for i in range(n_groups)]

    # Track within-group counters separately
    group_counters: list[int] = [0] * n_groups

    # Shuffle gene-to-group assignment so groups are not contiguous by global idx
    gene_idxs = list(range(n_genes))
    rng.shuffle(gene_idxs)

    vocab: dict[str, dict] = {}
    for global_idx in range(n_genes):
        gene = f"GENE{global_idx:04d}"
        # round-robin on the shuffled order so group sizes are equal
        assignment_pos = gene_idxs[global_idx] % n_groups
        group = group_names[assignment_pos]
        within_idx = group_counters[assignment_pos]
        group_counters[assignment_pos] += 1
        vocab[gene] = {
            "global_idx": global_idx,
            "group": group,
            "group_idx": within_idx,
            "chromosome": f"{(global_idx % 22) + 1}p{global_idx % 100}",
        }

    return vocab


def save_vocabulary(vocab: dict[str, dict], path: str) -> None:
    """Write a vocabulary dict to a JSON file, creating parent dirs if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
