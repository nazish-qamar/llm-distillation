# llm-distillation

Knowledge distillation of an LLM on [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
grade-school math problems: a **Qwen2.5-7B-Instruct** teacher produces
step-by-step solutions, and a **Qwen2.5-0.5B-Instruct** student is fine-tuned on
them. The question is how much of the teacher's reasoning ability transfers into
a model 14x smaller.

This is a learning project. GPU work runs on RunPod; run tracking goes to
Databricks MLflow.

## Status

Implemented and tested:

| Module | What it does |
| --- | --- |
| `src/kd/data.py` | Loads GSM8K into typed `Example`s (id, question, cleaned reasoning, normalized gold) |
| `src/kd/prompts.py` | The shared system prompt, inference messages, and SFT training targets |
| `src/kd/scoring.py` | Answer extraction from model output, plus `is_correct` / `accuracy` |

Not written yet: teacher generation, student fine-tuning, evaluation entry
points, and MLflow logging. `configs/` and `scripts/` are empty placeholders,
and the `kd` console script is still a stub that prints `Hello from kd!`.

## Requirements

- Python 3.12 (pinned: `requires-python = ">=3.12,<3.13"`, `.python-version` is `3.12`)
- [uv](https://docs.astral.sh/uv/)

## Setup after cloning

### 1. Dependencies

```bash
uv sync                 # base + dev dependencies
uv sync --extra gpu     # adds torch, transformers, trl, vllm (GPU boxes only)
```

The `gpu` extra is deliberately optional so a laptop checkout stays small. Note
that a plain `uv sync` on a machine where the extra is already installed will
uninstall it again; pass `--extra gpu` to keep it.

### 2. Environment variables

Secrets come from the environment, never from tracked files:

```bash
cp .env.example .env    # then fill in the values; .env is gitignored
```

### 3. Git hooks and gitleaks

Git hooks are not copied by `git clone`, and `gitleaks` is not a Python
dependency, so both need one manual step per clone:

```bash
# Enable the pre-commit secret scan (.githooks/pre-commit)
git config core.hooksPath .githooks

# Install gitleaks, which that hook runs
curl -sSL -O https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
tar xzf gitleaks_8.30.1_linux_x64.tar.gz
install -m 755 gitleaks ~/.local/bin/gitleaks
```

Other platforms and the release checksums are on the
[gitleaks releases page](https://github.com/gitleaks/gitleaks/releases).
The hook finds `gitleaks` on `PATH` or in `~/.local/bin`.

Verify both steps:

```bash
git config --get core.hooksPath   # -> .githooks
gitleaks version                  # -> 8.30.1
```

Without the first step the hook never runs and commits are not scanned. Without
the second the hook fails closed: it blocks the commit and tells you to install
gitleaks. To bypass it for one commit, use `git commit --no-verify`.

Rules live in `.gitleaks.toml`, which extends the gitleaks defaults with one
custom Hugging Face rule — the built-in `hf_` pattern matches letters only, so
it misses any real token containing a digit.

## Development

```bash
uv run pytest             # 104 tests
uv run ruff check .
uv run ruff format .
uv run mypy src tests     # strict mode
```

To scan the full history for secrets, not just staged changes:

```bash
gitleaks git --redact --verbose
```

## Data notes

`load_gsm8k("train")` gives 7473 examples and `load_gsm8k("test")` gives 1319.
Each `Example` has its calculator annotations (`<<48/2=24>>`) and the `#### N`
answer line stripped from `reasoning`, with the final number normalized into
`gold`.

Two invariants the tests pin down:

- **The `####` marker is authoritative.** If a completion has the marker but no
  number after it, `extract_answer` returns `None`. It does not fall back to
  numbers in the reasoning, which would otherwise score malformed output as
  correct and inflate student accuracy.
- **Round-trip.** For every example in both splits,
  `is_correct(format_target(reasoning, gold), gold)` is `True`, so a student that
  reproduces a training target exactly is always scored correct.
