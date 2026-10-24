"""Tests for kd.data parsing helpers."""

import pytest

from kd.data import clean_reasoning, parse_gold

# A realistic GSM8K answer field: steps with calculator annotations, then the '#### N' line.
GSM8K_ANSWER = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
    "#### 72"
)


class TestParseGold:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("#### 72", "72"),
            (GSM8K_ANSWER, "72"),
            ("some steps\n#### 0", "0"),
            ("#### 72\n", "72"),  # trailing newline
            ("####    72   ", "72"),  # padding around the number
            ("#### -5", "-5"),  # negative answer
            ("#### 3.5", "3.5"),  # decimal answer
            ("a\n#### 1\nb\n#### 2", "2"),  # last marker wins
            ("####72", "72"),  # no space after the marker
        ],
    )
    def test_returns_number_after_last_marker(self, answer: str, expected: str) -> None:
        assert parse_gold(answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("#### 1,000", "1000"),  # thousands separator stripped
            ("#### $10", "10"),  # currency symbol stripped
        ],
    )
    def test_normalizes_formatted_numbers(self, answer: str, expected: str) -> None:
        """Delegated to kd.scoring.normalize_number; encodes the contract parse_gold expects."""
        assert parse_gold(answer) == expected

    @pytest.mark.parametrize(
        "answer",
        [
            "",
            "no marker here",
            "Natalia sold 48/2 = 24 clips in May.",
            "### 72",  # three hashes, not four
        ],
    )
    def test_raises_when_marker_missing(self, answer: str) -> None:
        with pytest.raises(ValueError, match="####"):
            parse_gold(answer)

    @pytest.mark.parametrize(
        "answer",
        [
            "#### ",  # nothing after the marker
            "#### twelve",
            "#### n/a",
        ],
    )
    def test_raises_when_gold_not_a_number(self, answer: str) -> None:
        with pytest.raises(ValueError, match="not a number"):
            parse_gold(answer)


class TestCleanReasoning:
    def test_drops_marker_line_and_annotations(self) -> None:
        assert clean_reasoning(GSM8K_ANSWER) == (
            "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether."
        )

    def test_removes_every_annotation(self) -> None:
        answer = "a <<1+1=2>> b <<2*2=4>> c\n#### 4"
        assert clean_reasoning(answer) == "a  b  c"

    def test_keeps_step_structure(self) -> None:
        answer = "step one\nstep two\nstep three\n#### 9"
        assert clean_reasoning(answer) == "step one\nstep two\nstep three"

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("  steps  \n#### 7", "steps"),  # surrounding whitespace stripped
            ("steps\n\n\n#### 7", "steps"),  # blank lines before the marker
            ("#### 7", ""),  # nothing but the answer line
            ("steps\n#### 7\ntrailing", "steps"),  # text after the marker is dropped too
        ],
    )
    def test_strips_whitespace(self, answer: str, expected: str) -> None:
        assert clean_reasoning(answer) == expected

    def test_without_marker_returns_cleaned_body(self) -> None:
        assert clean_reasoning("just <<1+1=2>>steps") == "just steps"

    def test_splits_on_last_marker_only(self) -> None:
        assert clean_reasoning("a\n#### 1\nb\n#### 2") == "a\n#### 1\nb"

    def test_leaves_unclosed_annotation_alone(self) -> None:
        assert clean_reasoning("a <<1+1=2 b\n#### 2") == "a <<1+1=2 b"
