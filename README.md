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

`scripts/` holds the pod tooling: `setup_pod.sh` (bootstrap), `verify_pod.py`
(readiness checks) and `smoke_test.py` (end-to-end chain test). See
[Starting a pod](#starting-a-pod).

Not written yet: teacher generation, student fine-tuning, and evaluation entry
points. MLflow logging exists only in the smoke test so far. `configs/` is an
empty placeholder, and the `kd` console script is still a stub that prints
`Hello from kd!`.

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
uv run mypy src tests scripts   # strict mode
```

Any `uv run` that needs the GPU libraries must pass `--extra gpu`, e.g.
`uv run --extra gpu python scripts/verify_pod.py`. A plain `uv run` syncs the
environment to the default dependency set and **uninstalls** torch, vllm,
transformers and trl.

To scan the full history for secrets, not just staged changes:

```bash
gitleaks git --redact --verbose
```

## Starting a pod

Pods are disposable and have no network volume, so this runs from scratch each
time. Set these six variables in the RunPod **pod config** — the pod does not
read `.env`, which is local-only:

| Variable | Value in the pod config |
| --- | --- |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
| `DATABRICKS_TOKEN` | `{{ RUNPOD_SECRET_DATABRICKS_TOKEN }}` |
| `DATABRICKS_HOST` | plain value, e.g. `https://dbc-xxxxxxxx-xxxx.cloud.databricks.com` |
| `MLFLOW_TRACKING_URI` | plain value: `databricks` |
| `MLFLOW_EXPERIMENT_NAME` | plain value, e.g. `/Users/you@example.com/kd-gsm8k` |
| `HF_HUB_REPO_PREFIX` | plain value, e.g. `your-hf-username` |

The two secrets go in RunPod's secret store and are referenced with the
`{{ RUNPOD_SECRET_name }}` syntax so the values never appear in the pod config
itself. `setup_pod.sh` checks that all six are set and names any that are
missing, without printing values.

Then clone and bootstrap:

```bash
cd /workspace
git clone https://github.com/nazish-qamar/llm-distillation.git
cd llm-distillation
bash scripts/setup_pod.sh
```

`setup_pod.sh` is idempotent — re-run it after a `git push` to pick up new
code. It checks for a GPU and the six variables, points `HF_HOME` and
`UV_CACHE_DIR` at `/workspace` (the ~60GB volume disk, rather than the ~30GB
container disk), installs `uv` and `tmux` if missing, fast-forwards the repo,
runs `uv sync --extra gpu --locked`, and finishes with
`scripts/verify_pod.py`, exiting non-zero if the pod is not usable.

Run long jobs under `tmux` so they survive an SSH disconnect.

### Smoke test

Once `verify_pod.py` reports `Pod is ready.`, run the end-to-end check:

```bash
uv run --extra gpu --locked python scripts/smoke_test.py
```

The `--extra gpu` is required here too — a plain `uv run` would uninstall torch
and vllm before the script even starts.

It puts five GSM8K test examples through `Qwen/Qwen2.5-0.5B-Instruct` with vLLM
(greedy, `max_tokens=512`), scores them with `kd.scoring`, and logs one MLflow
run named `smoke-test` to the experiment in `MLFLOW_EXPERIMENT_NAME`, with the
full completions as an `outputs.jsonl` artifact. It ends with a summary block
giving the accuracy and the MLflow run id.

What it proves is that the chain is wired up — `kd.data` loading, `kd.prompts`
through the model's own chat template, generation, `kd.scoring` extraction, and
a run reaching Databricks. It says nothing useful about model quality: at 0.5B
and untuned, a low accuracy is expected. **Success is exit code 0 and a run id
you can open in MLflow, not a high score.**

A `finish=length` on any example means the completion hit `max_tokens` and was
cut off before its `#### N` line, so it scores wrong for a reason unrelated to
the model's reasoning. The script warns when that happens.

Verified on an A40 pod: `verify_pod.py` 13/13 PASS, and the smoke test ran end
to end at 2/5 accuracy with every completion finishing on `stop`, with the run
logged to Databricks.

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
