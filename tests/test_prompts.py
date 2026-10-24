"""Tests for kd.prompts message and target formatting."""

from kd.prompts import build_messages, format_target


def test_build_messages_returns_system_then_user() -> None:
    messages = build_messages("What is 2+2?")
    assert len(messages) == 2
    assert [m["role"] for m in messages] == ["system", "user"]


def test_format_target_ends_with_answer_line() -> None:
    assert format_target("Step 1...", "18").endswith("#### 18")
