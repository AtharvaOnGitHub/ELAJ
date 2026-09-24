"""Print a corpus config's `exclude_studies`, one per line.

    python scripts/corpus_exclusions.py configs/data/corpus_v1.yaml

WHY THIS EXISTS. `build_vocabulary.sh` has to know which studies the corpus
drops, and typing that list into the shell script is what let the vocabulary
and the corpus disagree: the build excluded wang_2022, which the corpus trains
on, and kept garcia_alonso_2021_c, which it does not. Deriving it from the
config removes the drift.

WHY IT REFUSES TO PRINT AN EMPTY LIST WHEN THE KEY IS PRESENT. The first
attempt at deriving it ran inside `$(...)`, which discards the exit status.
PyYAML was missing from the job script's module block, the traceback went to
stderr, EXCLUDE came out empty, and the build silently used all thirteen
studies: a third wrong vocabulary, announced as "Excluding: (nothing)" four
lines below its own traceback.

A list that cannot be read is not the same as a corpus that excludes nothing,
and the only safe behaviour is to fail rather than to guess. The caller checks
the exit status as well, so the two guards are independent.
"""

from __future__ import annotations

import sys

KEY = "exclude_studies"


def parse(text: str) -> list[str]:
    """The study names under `exclude_studies`, in file order.

    Raises ValueError if the key is present but yields no name, which is the
    case where returning [] would silently build the wrong vocabulary. A key
    that is absent returns [], because a corpus that drops nothing is a real
    and harmless configuration.
    """
    import yaml

    config = yaml.safe_load(text) or {}
    if not isinstance(config, dict):
        raise ValueError(f"expected a mapping at the top level of the config, "
                         f"got {type(config).__name__}")
    if KEY not in config:
        return []

    value = config[KEY]
    if value is None:
        raise ValueError(
            f"{KEY} is present but empty. Delete the key to mean 'exclude "
            f"nothing'; an empty value is more often an editing accident."
        )
    if not isinstance(value, list):
        raise ValueError(f"{KEY} must be a list, got {type(value).__name__}")
    if not value:
        return []            # `[]` is a deliberate statement, unlike a null

    names = [str(name).strip() for name in value if str(name).strip()]
    if not names:
        raise ValueError(
            f"{KEY} is present but no study name could be read from it. "
            f"Refusing to report an empty list."
        )
    return names


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: corpus_exclusions.py <corpus.yaml>", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        for name in parse(f.read()):
            print(name)


if __name__ == "__main__":
    main()
