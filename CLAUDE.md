This is a learning project on LLM knowledge distillation. Claude Code implements,
but I review every change.

- Plan before editing. Explain the approach, then wait for my approval.
- When you make a change, explain the non-obvious lines: why this way, and what breaks otherwise.
- Never claim something works unless you ran it and saw the output.
- Never put tokens or secrets in files. Secrets come from environment variables.
- Stack: uv, ruff, strict mypy, pytest. GPU code runs on RunPod; tracking goes to Databricks MLflow.