"""Extract and score final numeric answers from model output."""

import re
from decimal import Decimal, InvalidOperation

_ANSWER_MARKER = "####"
# Optional minus sign, digits with optional thousands commas, optional decimal part.
_NUMBER = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def normalize_number(s: str) -> str | None:
    """Canonical string form of a number: '1,000' -> '1000', '$18.00' -> '18'.

    Returns None if the string isn't a finite number.
    """
    cleaned = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite():  # Decimal accepts 'NaN' and 'Infinity'
        return None
    if value == value.to_integral_value():
        return str(int(value))  # '18.00' -> '18', avoids '1E+3' style output
    return format(value.normalize(), "f")  # '18.50' -> '18.5'


def _first_number(text: str) -> str | None:
    match = _NUMBER.search(text)
    return normalize_number(match.group()) if match else None


def _last_number(text: str) -> str | None:
    matches: list[str] = _NUMBER.findall(text)
    return normalize_number(matches[-1]) if matches else None


def extract_answer(text: str) -> str | None:
    """Find the final answer in model output.

    Priority: '#### N' marker, then '\\boxed{N}', then the last number in the text.
    """
    if _ANSWER_MARKER in text:
        # Marker is authoritative: present but non-numeric means None, no fallthrough.
        return _first_number(text.rsplit(_ANSWER_MARKER, 1)[1])

    boxed: list[str] = _BOXED.findall(text)
    if boxed:
        answer = _first_number(boxed[-1])
        if answer is not None:
            return answer

    return _last_number(text)


def is_correct(pred_text: str, gold: str) -> bool:
    """True if the answer extracted from pred_text equals the gold answer."""
    pred = extract_answer(pred_text)
    return pred is not None and pred == normalize_number(gold)


def accuracy(preds: list[str], golds: list[str]) -> float:
    """Fraction of predictions whose extracted answer matches the gold."""
    if len(preds) != len(golds):
        raise ValueError(f"Length mismatch: {len(preds)} preds vs {len(golds)} golds")
    if not preds:
        raise ValueError("Cannot compute accuracy of an empty list")
    return sum(is_correct(p, g) for p, g in zip(preds, golds, strict=True)) / len(preds)
