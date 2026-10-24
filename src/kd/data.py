"""Load GSM8K into clean, typed examples."""

import re
from dataclasses import dataclass
from typing import Literal

from datasets import load_dataset

from kd.scoring import normalize_number

Split = Literal["train", "test"]

_ANSWER_MARKER = "####"
_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    reasoning: str  # step-by-step solution, annotations and final-answer line removed
    gold: str  # normalized final number


def parse_gold(answer: str) -> str:
    """Return the normalized number after the last '####' marker."""
    if _ANSWER_MARKER not in answer:
        raise ValueError(f"No '{_ANSWER_MARKER}' marker in answer: {answer[-80:]!r}")
    raw = answer.rsplit(_ANSWER_MARKER, 1)[1].strip()
    gold = normalize_number(raw)
    if gold is None:
        raise ValueError(f"Gold answer is not a number: {raw!r}")
    return gold


def clean_reasoning(answer: str) -> str:
    """Drop the '#### N' line and calculator annotations like '<<48/2=24>>'."""
    body = answer.rsplit(_ANSWER_MARKER, 1)[0]
    body = _CALC_ANNOTATION.sub("", body)
    return body.strip()


def load_gsm8k(split: Split) -> list[Example]:
    """Load one GSM8K split as a list of Examples."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples: list[Example] = []
    for i, row in enumerate(ds):
        answer = str(row["answer"])
        examples.append(
            Example(
                id=f"{split}-{i}",
                question=str(row["question"]).strip(),
                reasoning=clean_reasoning(answer),
                gold=parse_gold(answer),
            )
        )
    return examples
