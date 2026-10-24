#!/usr/bin/env bash
#
# Bootstrap a fresh RunPod GPU pod for llm-distillation.
#
#   cd /workspace/llm-distillation && bash scripts/setup_pod.sh
#
# Safe to re-run: every step is guarded, so a second run is a no-op.
#
# There is deliberately no `set -x` in this script. It reads secret environment
# variables (HF_TOKEN, DATABRICKS_TOKEN) and tracing would print them into the
# pod log, which is exactly what we are trying to avoid.

set -euo pipefail

# Resolve the repo root from this script's own location rather than hardcoding
# /workspace/llm-distillation, so a clone elsewhere still works.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

readonly HF_CACHE_PATH="/workspace/hf-cache"
readonly UV_CACHE_PATH="/workspace/uv-cache"
readonly BASHRC="$HOME/.bashrc"
readonly BASHRC_MARKER="# >>> llm-distillation pod env >>>"
readonly BASHRC_END="# <<< llm-distillation pod env <<<"

step() { printf "\n==> %s\n" "$1"; }
die() { printf "ERROR: %s\n" "$1" >&2; exit 1; }

# --------------------------------------------------------------------------
step "Checking for an NVIDIA GPU"
# --------------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi not found. Is this a GPU pod? (expected the PyTorch/CUDA template)"
printf "nvidia-smi found\n"

# --------------------------------------------------------------------------
step "Checking required environment variables"
# --------------------------------------------------------------------------
required_vars=(
    HF_TOKEN
    DATABRICKS_HOST
    DATABRICKS_TOKEN
    MLFLOW_TRACKING_URI
    MLFLOW_EXPERIMENT_NAME
    HF_HUB_REPO_PREFIX
)
missing=()
for name in "${required_vars[@]}"; do
    # Indirect expansion reads the value only to test whether it is empty.
    # The value itself is never printed, logged, or written anywhere.
    if [[ -z "${!name:-}" ]]; then
        missing+=("$name")
    fi
done
if ((${#missing[@]} > 0)); then
    printf "ERROR: these environment variables are unset or empty:\n" >&2
    printf "  - %s\n" "${missing[@]}" >&2
    printf "\nSet them in the RunPod pod config (secrets as {{ RUNPOD_SECRET_name }}),\n" >&2
    printf "then restart the pod. See the 'Starting a pod' section of README.md.\n" >&2
    exit 1
fi
printf "all %d present (values not shown)\n" "${#required_vars[@]}"

# --------------------------------------------------------------------------
step "GPU and driver"
# --------------------------------------------------------------------------
nvidia-smi
# The CUDA version in the nvidia-smi header is the highest CUDA the driver
# supports, not the toolkit that is installed. verify_pod.py compares it
# against the CUDA version torch was built for.
driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)"
if [[ -n "$driver_cuda" ]]; then
    printf "\nDriver supports CUDA up to: %s\n" "$driver_cuda"
else
    printf "\nCould not parse a CUDA version from the nvidia-smi header.\n"
fi

# --------------------------------------------------------------------------
step "Pointing caches at /workspace"
# --------------------------------------------------------------------------
# Model weights and wheels are tens of GB. /workspace is the ~60GB volume
# disk; the container disk is only ~30GB and would fill up first.
export HF_HOME="$HF_CACHE_PATH"
export UV_CACHE_DIR="$UV_CACHE_PATH"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
printf "HF_HOME=%s\nUV_CACHE_DIR=%s\n" "$HF_HOME" "$UV_CACHE_DIR"

# --------------------------------------------------------------------------
step "Persisting cache paths to $BASHRC"
# --------------------------------------------------------------------------
# Guarded by a marker so re-running appends nothing. An interactive SSH shell
# needs these too, not just this script's process.
if grep -qF "$BASHRC_MARKER" "$BASHRC" 2>/dev/null; then
    printf "marker already present, leaving %s unchanged\n" "$BASHRC"
else
    cat >>"$BASHRC" <<EOF
$BASHRC_MARKER
export HF_HOME="$HF_CACHE_PATH"
export UV_CACHE_DIR="$UV_CACHE_PATH"
export PATH="\$HOME/.local/bin:\$PATH"
$BASHRC_END
EOF
    printf "appended to %s\n" "$BASHRC"
fi

# --------------------------------------------------------------------------
step "Installing uv if missing"
# --------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
    printf "already installed: %s\n" "$(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    hash -r # forget the cached "not found" lookup
    command -v uv >/dev/null 2>&1 ||
        die "uv installer finished but uv is still not on PATH"
    printf "installed: %s\n" "$(uv --version)"
fi

# --------------------------------------------------------------------------
step "Installing tmux if missing"
# --------------------------------------------------------------------------
# Training and generation runs take hours. Without tmux an SSH disconnect
# kills the job and the pod keeps billing for nothing.
if command -v tmux >/dev/null 2>&1; then
    printf "already installed: %s\n" "$(tmux -V)"
else
    sudo_cmd=()
    if [[ "$(id -u)" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 ||
            die "tmux is missing, this user is not root, and sudo is unavailable"
        sudo_cmd=(sudo)
    fi
    # `env` carries DEBIAN_FRONTEND through sudo, which strips the environment.
    "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update -qq
    "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
    printf "installed: %s\n" "$(tmux -V)"
fi

# --------------------------------------------------------------------------
step "Updating the repo (fast-forward only)"
# --------------------------------------------------------------------------
# --ff-only so a diverged pod checkout fails loudly instead of creating a
# merge commit nobody will ever see.
git pull --ff-only

# --------------------------------------------------------------------------
step "Syncing dependencies"
# --------------------------------------------------------------------------
# --locked asserts uv.lock is current and installs exactly it. If the lockfile
# has drifted from pyproject.toml this fails here, rather than the pod quietly
# re-resolving to a dependency set that was never tested.
uv sync --extra gpu --locked

# --------------------------------------------------------------------------
step "Verifying the pod"
# --------------------------------------------------------------------------
# --extra gpu is required, not cosmetic: a plain `uv run` syncs the environment
# to the default dependency set, which would uninstall torch, vllm,
# transformers and trl before the script even starts.
#
# exec so this script exits with the verifier's status: a pod that fails
# verification must not look like a successful bootstrap.
exec uv run --extra gpu --locked python scripts/verify_pod.py
