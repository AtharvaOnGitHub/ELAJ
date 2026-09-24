"""`exclude_studies` is derived from the corpus config, and must never fail open.

The vocabulary in use was built from a typed exclusion list that had drifted
from the corpus config, so it dropped wang_2022 (which the corpus trains on)
and kept garcia_alonso_2021_c (which it does not). Deriving the list fixes the
drift, and introduced a worse failure on the first attempt: the derivation ran
inside `$(...)`, which discards the exit status, PyYAML was missing, and the
build used all thirteen studies while printing "Excluding: (nothing)".

So the property under test is not only "parses the list" but "**refuses to
report an empty list when the key is present**". An exclusion list that cannot
be read and a corpus that excludes nothing must never look the same.
"""

from pathlib import Path

import pytest

from scripts.corpus_exclusions import parse  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "data"


class TestTheRealConfigs:
    @pytest.mark.parametrize("name,expected", [
        ("corpus_c1.yaml", ["zhang_2022", "garcia_alonso_2021_c"]),
        ("corpus_v1.yaml", ["zhang_2022", "garcia_alonso_2021_c"]),
    ])
    def test_the_c1_and_v1_corpora(self, name, expected):
        assert parse((CONFIGS / name).read_text()) == expected

    def test_v1_matches_c1(self):
        """The v1 corpus differs from c1 in vocabulary_path and nothing else.
        If the exclusions ever diverge, the rebuild is answering a different
        question from the one it was built for."""
        assert (parse((CONFIGS / "corpus_v1.yaml").read_text())
                == parse((CONFIGS / "corpus_c1.yaml").read_text()))

    def test_every_corpus_config_parses(self):
        """A config this cannot read would stop a build, so none may exist."""
        for path in sorted(CONFIGS.glob("corpus*.yaml")):
            parse(path.read_text())


class TestItFailsRatherThanGuessing:
    def test_a_present_but_empty_key_raises(self):
        with pytest.raises(ValueError, match="present but empty"):
            parse("exclude_studies:\nvocabulary_path: /x.json\n")

    def test_a_list_of_blanks_raises(self):
        with pytest.raises(ValueError, match="no study name"):
            parse("exclude_studies:\n  - ''\n  - '  '\n")

    def test_a_scalar_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            parse("exclude_studies: zhang_2022\n")

    def test_a_non_mapping_config_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            parse("- just\n- a\n- list\n")

    def test_invalid_yaml_raises_rather_than_returning_nothing(self):
        import yaml

        with pytest.raises(yaml.YAMLError):
            parse("exclude_studies:\n  - a\n   bad indent: [\n")


class TestAbsentIsNotAnError:
    def test_no_key_means_exclude_nothing(self):
        """A corpus that drops no study is a real configuration, unlike a key
        that is present and unreadable."""
        assert parse("vocabulary_path: /x.json\n") == []

    def test_an_empty_config_is_empty(self):
        assert parse("") == []


class TestShapes:
    def test_comments_and_blank_lines_are_ignored(self):
        text = (
            "exclude_studies:\n"
            "  # zhang is out of scope\n"
            "  - zhang_2022\n"
            "\n"
            "  - garcia_alonso_2021_c\n"
            "include_disease_statuses:\n  - control\n"
        )
        assert parse(text) == ["zhang_2022", "garcia_alonso_2021_c"]

    def test_an_inline_list_works(self):
        assert parse("exclude_studies: [a, b]\n") == ["a", "b"]

    def test_an_explicitly_empty_inline_list_is_allowed(self):
        assert parse("exclude_studies: []\n") == []

    def test_the_following_key_is_not_swallowed(self):
        text = "exclude_studies:\n  - a\ninclude_disease_statuses:\n  - control\n"
        assert parse(text) == ["a"]

    def test_order_is_preserved(self):
        assert parse("exclude_studies:\n  - b\n  - a\n  - c\n") == ["b", "a", "c"]
