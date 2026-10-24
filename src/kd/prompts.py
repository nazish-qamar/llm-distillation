"""Prompt and training-target formatting for GSM8K."""

from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "showing your reasoning. End your response with a final line of the form "
    "'#### <number>', where <number> is the final numeric answer only."
)


def build_messages(question: str) -> list[ChatMessage]:
    """Chat messages for inference: system instructions plus the question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def format_target(reasoning: str, gold: str) -> str:
    """The assistant response a student is trained to produce."""
    return f"{reasoning.strip()}\n#### {gold}"


def build_training_messages(question: str, reasoning: str, gold: str) -> list[ChatMessage]:
    """Full conversation for SFT: prompt plus the target assistant turn."""
    return [
        *build_messages(question),
        {"role": "assistant", "content": format_target(reasoning, gold)},
    ]
