"""Smoke-test the full distillation chain on a GPU pod.

Run with:

    uv run --extra gpu --locked python scripts/smoke_test.py

Five GSM8K test examples through Qwen2.5-0.5B-Instruct with vLLM, scored with
kd.scoring, logged as one MLflow run. This proves the pipeline is wired up
end to end; it says nothing useful about model quality at this size.

Every step runs inside main() under the __main__ guard: vLLM can start worker
processes that re-import this module, and anything at module scope would then
be executed again in each worker.
"""

import json
import os
import subprocess
import sys
from typing import Any

from kd.data import load_gsm8k
from kd.prompts import build_messages
from kd.scoring import accuracy, extract_answer, is_correct

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_EXAMPLES = 5
TEMPERATURE = 0.0
MAX_TOKENS = 512
# Pinned rather than left at Qwen's 32768: the KV cache is sized from this, and
# the default risks an OOM on a smaller pod GPU for no benefit when prompts are
# a few hundred tokens and completions are capped at MAX_TOKENS.
MAX_MODEL_LEN = 2048
RUN_NAME = "smoke-test"
_PREVIEW_CHARS = 300


def _fail(message: str) -> int:
    """Print one clear line and return the exit code. No traceback."""
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def _git_commit() -> str:
    """The HEAD commit, for tagging the run. 'unknown' rather than raising."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def main() -> int:
    # 1. Tracking-URI guard, same as scripts/verify_pod.py. Env only, no
    # network. With MLFLOW_TRACKING_URI unset, mlflow silently falls back to a
    # local store and the run would never reach Databricks.
    import mlflow

    uri = str(mlflow.get_tracking_uri())
    if not uri.startswith("databricks"):
        return _fail(
            f"MLFLOW_TRACKING_URI resolves to {uri!r}, not databricks. "
            "The run would be written to a local store. Set it and re-run."
        )
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "")
    if not experiment:
        return _fail("MLFLOW_EXPERIMENT_NAME is not set.")

    # 2. GPU libraries. Imported here, deliberately before any MLflow network
    # call, so a machine without them fails on this clear message instead of a
    # confusing auth error -- and so this path stays testable off a pod.
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        return _fail(
            f"{exc.name or 'a GPU library'} is not installed. Run this on a GPU pod with: "
            "uv run --extra gpu --locked python scripts/smoke_test.py"
        )

    # 3. Data, straight from kd.data: cheap and locally cached.
    examples = load_gsm8k("test")[:N_EXAMPLES]
    print(f"torch {torch.__version__}, vllm {vllm.__version__}")
    print(f"{len(examples)} examples from the GSM8K test split\n")

    # 4. Resolve the experiment before the model loads, so bad credentials fail
    # in seconds rather than after minutes of GPU work.
    mlflow.set_experiment(experiment)

    # 5. Model.
    llm = LLM(model=MODEL, max_model_len=MAX_MODEL_LEN)

    # 6. Prompts: kd.prompts owns the prompt content, the tokenizer owns the
    # chat formatting. The tokenizer comes off the loaded LLM so it is
    # guaranteed to be the one the model is running with.
    tokenizer = llm.get_tokenizer()
    prompts = [
        str(
            tokenizer.apply_chat_template(
                build_messages(example.question),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        for example in examples
    ]

    # 7. Greedy decoding in one batched call.
    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    outputs = llm.generate(prompts, sampling)

    # 8. Score and report. is_correct/extract_answer/accuracy all come from
    # kd.scoring -- no comparison or extraction logic is repeated here.
    rows: list[dict[str, Any]] = []
    completions: list[str] = []
    for example, output in zip(examples, outputs, strict=True):
        completion = str(output.outputs[0].text)
        # "length" means MAX_TOKENS truncated the completion before it could
        # reach its '#### N' line, so it scores wrong for a reason that has
        # nothing to do with the model's reasoning. Worth seeing per example.
        finish_reason = str(output.outputs[0].finish_reason or "unknown")
        completions.append(completion)
        rows.append(
            {
                "id": example.id,
                "question": example.question,
                "gold": example.gold,
                "extracted": extract_answer(completion),
                "correct": is_correct(completion, example.gold),
                "finish_reason": finish_reason,
                "completion": completion,
            }
        )

    for row in rows:
        preview = " ".join(str(row["completion"]).split())[:_PREVIEW_CHARS]
        print(
            f"id={row['id']}  gold={row['gold']}  extracted={row['extracted']}  "
            f"correct={'yes' if row['correct'] else 'no'}  finish={row['finish_reason']}"
        )
        print(f"  {preview}\n")

    acc = accuracy(completions, [example.gold for example in examples])
    n_correct = sum(1 for row in rows if row["correct"])
    print(f"accuracy: {acc:.2f} ({n_correct}/{len(rows)})")

    truncated = sum(1 for row in rows if row["finish_reason"] == "length")
    if truncated:
        print(f"warning: {truncated} completion(s) hit max_tokens={MAX_TOKENS} and were cut off")

    # 9. One run, logged after generation so a crash mid-generation leaves no
    # half-written run behind.
    gpu_name = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "unknown"
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param("model", MODEL)
        mlflow.log_param("n_examples", len(examples))
        mlflow.log_param("temperature", TEMPERATURE)
        mlflow.log_param("max_tokens", MAX_TOKENS)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.set_tag("gpu_name", gpu_name)
        mlflow.set_tag("torch_version", str(torch.__version__))
        mlflow.set_tag("vllm_version", str(vllm.__version__))
        mlflow.log_text("\n".join(json.dumps(row) for row in rows), "outputs.jsonl")
        run_id = str(run.info.run_id)

    # Reprinted together at the very end: MLflow's own logging output comes
    # between the accuracy line above and here, so without this the score is
    # buried above the chatter.
    print("\n--- summary ---")
    print(f"accuracy:      {acc:.2f} ({n_correct}/{len(rows)})")
    print(f"mlflow run id: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
