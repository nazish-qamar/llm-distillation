"""Verify that a RunPod GPU pod is actually ready to run distillation jobs.

Run via:

    uv run --extra gpu --locked python scripts/verify_pod.py

The `--extra gpu` is required: a plain `uv run` syncs the environment to the
default dependency set and would uninstall torch, vllm, transformers and trl.

Prints a PASS/WARN/FAIL table and exits non-zero if anything FAILs. Low disk
space is a WARN, not a FAIL. No secret value is ever printed.
"""

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Status = Literal["PASS", "WARN", "FAIL"]

MIN_FREE_GIB = 20
WORKSPACE = "/workspace"
CONTAINER_DISK = "/"
_GIB = 1024**3
# Values of these are scrubbed from all output. Presence is checked by
# setup_pod.sh; this script only needs them for redaction.
_SECRET_ENV_VARS = ("HF_TOKEN", "DATABRICKS_TOKEN")
_DETAIL_MAX_LEN = 160


@dataclass(frozen=True)
class Result:
    name: str
    status: Status
    detail: str


def _redact(text: str) -> str:
    """Replace any secret environment value that appears in text.

    Nothing here prints a secret deliberately; this is the second layer, for
    library exceptions that might embed a token in a URL or error message.
    """
    for var in _SECRET_ENV_VARS:
        value = os.environ.get(var, "")
        if len(value) >= 8:  # ignore trivially short values to avoid mangling text
            text = text.replace(value, "***")
    return text


def _parse_major_minor(text: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", text.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _driver_cuda_version() -> str | None:
    """Highest CUDA version the installed driver supports, per nvidia-smi."""
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", proc.stdout)
    return match.group(1) if match else None


def check_torch_import() -> Result:
    import torch

    return Result("torch import", "PASS", f"version {torch.__version__}")


def check_cuda_available() -> Result:
    import torch

    if not torch.cuda.is_available():
        return Result("cuda available", "FAIL", "torch.cuda.is_available() is False")
    return Result("cuda available", "PASS", f"{torch.cuda.device_count()} device(s)")


def check_gpu() -> Result:
    import torch

    if not torch.cuda.is_available():
        return Result("gpu", "FAIL", "no CUDA device")
    props = torch.cuda.get_device_properties(0)
    return Result("gpu", "PASS", f"{props.name}, {props.total_memory / _GIB:.1f} GiB VRAM")


def check_gpu_compute() -> Result:
    """Ground truth: run a real op on the device.

    Version strings can agree while the stack is still broken, so this is the
    check that actually decides whether the pod is usable.
    """
    import torch

    if not torch.cuda.is_available():
        return Result("gpu compute", "FAIL", "no CUDA device")
    value = (torch.ones(2, device="cuda") * 2).sum().item()
    if value != 4:
        return Result("gpu compute", "FAIL", f"expected 4, got {value}")
    return Result("gpu compute", "PASS", f"tiny op on device returned {value}")


def check_cuda_versions() -> Result:
    """Diagnostic for the most common pod failure: torch built for a newer CUDA.

    Severity is tiered because minor-version forward compatibility normally
    holds. `gpu compute` above is the real verdict.
    """
    import torch

    torch_cuda = torch.version.cuda
    name = "cuda build vs driver"
    if torch_cuda is None:
        return Result(name, "FAIL", "torch has no CUDA build (CPU-only wheel installed)")
    driver_cuda = _driver_cuda_version()
    if driver_cuda is None:
        return Result(name, "WARN", f"torch cuda {torch_cuda}; driver ceiling unreadable")

    torch_version = _parse_major_minor(str(torch_cuda))
    driver_version = _parse_major_minor(driver_cuda)
    detail = f"torch cuda {torch_cuda} vs driver ceiling {driver_cuda}"
    if torch_version is None or driver_version is None:
        return Result(name, "WARN", f"{detail} - could not parse one of them")
    if torch_version[0] > driver_version[0]:
        return Result(name, "FAIL", f"{detail} - driver major too old, needs an upgrade")
    if torch_version > driver_version:
        return Result(name, "WARN", f"{detail} - newer minor, usually forward compatible")
    return Result(name, "PASS", detail)


def check_vllm() -> Result:
    import vllm

    return Result("vllm import", "PASS", f"version {vllm.__version__}")


def check_transformers() -> Result:
    import transformers

    return Result("transformers import", "PASS", f"version {transformers.__version__}")


def check_trl() -> Result:
    import trl

    return Result("trl import", "PASS", f"version {trl.__version__}")


def check_hf_auth() -> Result:
    """Confirm the HF token works, and surface its role.

    Step 11 pushes the distilled model, which needs a write token, so the role
    is worth seeing now. A read token still passes here.
    """
    from huggingface_hub import whoami

    info = whoami()
    user = str(info.get("name") or "<unknown>")
    # Same traversal huggingface_hub uses internally (_login.py), which
    # tolerates either key being absent.
    access_token_info = (info.get("auth") or {}).get("accessToken") or {}
    role = access_token_info.get("role")
    return Result("hf auth", "PASS", f"{user} (role: {role if role else 'unknown'})")


def check_mlflow_auth() -> Result:
    """Read-only reachability check. Creating runs is step 7, not this script.

    The tracking URI is asserted *before* a client is built. With
    MLFLOW_TRACKING_URI unset, mlflow silently falls back to a local store and
    a reachability check would pass while nothing reaches Databricks -- and
    merely constructing the client creates the local store on disk.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    name = "mlflow auth"
    uri = str(mlflow.get_tracking_uri())
    if not uri.startswith("databricks"):
        return Result(name, "FAIL", f"tracking uri is {uri!r}, not databricks: would log locally")

    experiments = MlflowClient().search_experiments(max_results=1)
    return Result(name, "PASS", f"{uri}, {len(experiments)} experiment(s) returned")


def _check_disk(name: str, path: str) -> Result:
    if not os.path.isdir(path):
        return Result(name, "FAIL", f"{path} does not exist")
    free_gib = shutil.disk_usage(path).free / _GIB
    detail = f"{free_gib:.1f} GiB free on {path}"
    if free_gib < MIN_FREE_GIB:
        return Result(name, "WARN", f"{detail} - below {MIN_FREE_GIB} GiB")
    return Result(name, "PASS", detail)


def check_workspace_disk() -> Result:
    return _check_disk("workspace disk", WORKSPACE)


def check_container_disk() -> Result:
    return _check_disk("container disk", CONTAINER_DISK)


def check_hf_home() -> Result:
    """HF_HOME must be on the volume disk, or model weights fill the container."""
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        return Result("hf_home", "FAIL", "HF_HOME is not set")
    if hf_home != WORKSPACE and not hf_home.startswith(f"{WORKSPACE}/"):
        return Result("hf_home", "FAIL", f"{hf_home} is not under {WORKSPACE}")
    return Result("hf_home", "PASS", hf_home)


_CHECKS: tuple[tuple[str, Callable[[], Result]], ...] = (
    ("torch import", check_torch_import),
    ("cuda available", check_cuda_available),
    ("gpu", check_gpu),
    ("gpu compute", check_gpu_compute),
    ("cuda build vs driver", check_cuda_versions),
    ("vllm import", check_vllm),
    ("transformers import", check_transformers),
    ("trl import", check_trl),
    ("hf auth", check_hf_auth),
    ("mlflow auth", check_mlflow_auth),
    ("workspace disk", check_workspace_disk),
    ("container disk", check_container_disk),
    ("hf_home", check_hf_home),
)


def _run(name: str, check: Callable[[], Result]) -> Result:
    """Run one check, turning any exception into a FAIL row.

    Broad except on purpose: one unavailable library or unreachable service
    must not abort the table or dump a traceback at someone bringing up a pod.
    """
    try:
        return check()
    except Exception as exc:
        message = _redact(f"{type(exc).__name__}: {exc}").splitlines()[0]
        return Result(name, "FAIL", message[:_DETAIL_MAX_LEN])


def main() -> int:
    results = [_run(name, check) for name, check in _CHECKS]
    width = max(len(result.name) for result in results)

    print(f"{'CHECK'.ljust(width)}  STATUS  DETAIL")
    print(f"{'-' * width}  ------  {'-' * 44}")
    for result in results:
        print(f"{result.name.ljust(width)}  {result.status:<6}  {_redact(result.detail)}")

    failed = [result for result in results if result.status == "FAIL"]
    warned = [result for result in results if result.status == "WARN"]
    passed = len(results) - len(failed) - len(warned)
    print(f"\n{len(results)} checks: {passed} PASS, {len(warned)} WARN, {len(failed)} FAIL")

    if failed:
        print("\nPod is NOT ready. Failing checks:")
        for result in failed:
            print(f"  - {result.name}: {_redact(result.detail)}")
        return 1
    print("\nPod is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
