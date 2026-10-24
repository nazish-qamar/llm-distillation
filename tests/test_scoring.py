"""Tests for kd.scoring answer extraction and metrics."""

import pytest

from kd.scoring import accuracy, extract_answer, is_correct, normalize_number


class TestNormalizeNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("72", "72"),
            ("0", "0"),
            ("  42  ", "42"),  # surrounding whitespace
            ("-5", "-5"),
            ("1,000", "1000"),  # thousands separator
            ("1,234,567", "1234567"),
            ("$18", "18"),  # currency symbol
            ("$18.00", "18"),  # trailing zeros dropped
            ("18.50", "18.5"),
            ("1,000.50", "1000.5"),  # commas and decimals together
            ("3.", "3"),  # trailing period
            ("1000000", "1000000"),  # stays plain, not '1E+6'
            ("-0", "0"),  # negative zero collapses
            ("+5", "5"),
        ],
    )
    def test_canonicalizes_numbers(self, raw: str, expected: str) -> None:
        assert normalize_number(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "twelve",
            "n/a",
            "12abc",
            "--5",
            ".",
            "$",
            "1 000",  # internal space is not stripped
        ],
    )
    def test_returns_none_for_non_numbers(self, raw: str) -> None:
        assert normalize_number(raw) is None

    @pytest.mark.parametrize("raw", ["NaN", "nan", "Infinity", "inf", "-inf"])
    def test_returns_none_for_non_finite(self, raw: str) -> None:
        """Decimal() accepts these spellings, so is_finite() has to reject them."""
        assert normalize_number(raw) is None

    def test_accepts_scientific_notation(self) -> None:
        """Decimal() parses exponents; documents that this is not rejected."""
        assert normalize_number("1e3") == "1000"


class TestExtractAnswer:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("#### 72", "72"),
            ("Natalia sold 24 clips.\n#### 72", "72"),
            ("#### 1,000", "1000"),
            ("#### $18.00", "18"),
            ("####72", "72"),  # no space after the marker
            ("#### 72.", "72"),  # trailing period
            ("#### 1\n#### 2", "2"),  # last marker wins
        ],
    )
    def test_prefers_the_marker(self, text: str, expected: str) -> None:
        assert extract_answer(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The answer is \\boxed{42}.", "42"),
            ("\\boxed{1,000}", "1000"),
            ("\\boxed{1} then \\boxed{2}", "2"),  # last box wins
            ("\\boxed{42} and I also wrote 7 here", "42"),  # beats the trailing number
        ],
    )
    def test_falls_back_to_boxed(self, text: str, expected: str) -> None:
        assert extract_answer(text) == expected

    def test_marker_beats_boxed(self) -> None:
        assert extract_answer("\\boxed{1}\n#### 2") == "2"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("so 3 then 4 then 5", "5"),
            ("I spent 5 dollars and got 3.5 back", "3.5"),
            ("the temperature is -5", "-5"),
            ("total: 1,000", "1000"),
        ],
    )
    def test_falls_back_to_last_number(self, text: str, expected: str) -> None:
        assert extract_answer(text) == expected

    def test_marker_without_a_number_returns_none(self) -> None:
        """The marker is authoritative: if it holds no number, nothing else counts."""
        assert extract_answer("The answer is 5\n#### unknown") is None

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "no numbers here",
            "#### unknown",  # marker present, nothing numeric anywhere
            "\\boxed{n/a}",
        ],
    )
    def test_returns_none_when_no_number_found(self, text: str) -> None:
        assert extract_answer(text) is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("#### .5", "0.5"),  # marker tier
            ("#### -.5", "-0.5"),  # negative, no leading digit
            ("\\boxed{.25}", "0.25"),  # boxed tier
            ("so the answer is .75", "0.75"),  # last-number tier
        ],
    )
    def test_keeps_bare_leading_dot_decimals(self, text: str, expected: str) -> None:
        """A decimal with no leading digit must not lose its integer part."""
        assert extract_answer(text) == expected


class TestIsCorrect:
    @pytest.mark.parametrize(
        ("pred_text", "gold"),
        [
            ("#### 72", "72"),
            ("steps\n#### 72", "72"),
            ("#### 72", "72.00"),  # gold is normalized too
            ("#### 1000", "1,000"),  # formatting differences don't matter
            ("The answer is \\boxed{18}", "18"),
            ("I think it is 18", "18"),  # last-number fallback still scores
        ],
    )
    def test_true_when_answers_match(self, pred_text: str, gold: str) -> None:
        assert is_correct(pred_text, gold) is True

    @pytest.mark.parametrize(
        ("pred_text", "gold"),
        [
            ("#### 71", "72"),
            ("#### -72", "72"),  # sign matters
            ("no numbers here", "72"),  # nothing extractable
            ("", "72"),
            ("#### 72", "twelve"),  # unscoreable gold is False, not an error
            ("#### .5", "5"),  # regression: '.5' must not be read as '5'
        ],
    )
    def test_false_when_answers_differ_or_are_missing(self, pred_text: str, gold: str) -> None:
        assert is_correct(pred_text, gold) is False


class TestAccuracy:
    def test_all_correct(self) -> None:
        assert accuracy(["#### 1", "#### 2"], ["1", "2"]) == 1.0

    def test_none_correct(self) -> None:
        assert accuracy(["#### 1", "#### 2"], ["9", "9"]) == 0.0

    def test_partial(self) -> None:
        preds = ["#### 1", "#### 9", "#### 3", "no answer"]
        assert accuracy(preds, ["1", "2", "3", "4"]) == 0.5

    def test_returns_fraction_not_integer_division(self) -> None:
        assert accuracy(["#### 1", "#### 9", "#### 9"], ["1", "2", "3"]) == pytest.approx(1 / 3)

    def test_single_example(self) -> None:
        assert accuracy(["#### 1"], ["1"]) == 1.0

    def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            accuracy(["#### 1"], ["1", "2"])

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            accuracy([], [])
