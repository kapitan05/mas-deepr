#!/usr/bin/env bash
# One-time setup of a GPU machine (Google Cloud VM or RunPod pod) for
# scripts/run_queue.py. Safe to rerun: every step skips what's already done.
#
#   curl -LsSf https://raw.githubusercontent.com/kapitan05/mas-deepr/<branch>/scripts/vm/setup_vm.sh -o setup_vm.sh
#   BRANCH=<branch> bash setup_vm.sh
#
# or, from an existing clone:   bash scripts/vm/setup_vm.sh
#
# Then copy your .env into the project folder (it is never in git) and rerun
# this script once more -- the last step checks which keys are present.
#
# Settings (environment variables):
#   REPO_URL   default https://github.com/kapitan05/mas-deepr.git
#   BRANCH     default main
#   BASE_DIR   default /workspace if it exists (RunPod's persistent volume),
#              else $HOME
#
# Installs: system packages (git, tmux, build tools), uv, Python 3.12 venv
# .venv-gpu with the project + openpipe-art[backend]==0.5.20 (wandb>=0.30
# forced over ART's pin), Chromium for crawl4ai, ART's managed vLLM runtime,
# the Qwen3-4B-Instruct-2507 weights and the training/eval datasets. The GPU
# driver is NOT installed here -- use an image that ships it (Google Cloud
# "Deep Learning VM" images, any RunPod PyTorch template).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/kapitan05/mas-deepr.git}"
BRANCH="${BRANCH:-main}"
if [[ -z "${BASE_DIR:-}" ]]; then
  if [[ -d /workspace ]]; then BASE_DIR=/workspace; else BASE_DIR="$HOME"; fi
fi
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
ART_VERSION="0.5.20"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*"; }
SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi

# --- 1. system packages ------------------------------------------------------
say "System packages"
if ! command -v git >/dev/null || ! command -v tmux >/dev/null \
   || ! command -v curl >/dev/null || ! command -v gcc >/dev/null; then
  $SUDO apt-get update -y
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y \
    git tmux curl ca-certificates build-essential
else
  echo "git, tmux, curl, gcc already installed"
fi

# --- 2. uv -------------------------------------------------------------------
say "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

# --- 3. code -----------------------------------------------------------------
# Run from inside a clone -> use it; otherwise clone into $BASE_DIR/mas-deepr.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../../pyproject.toml" && -d "$SCRIPT_DIR/../../.git" ]]; then
  PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
  say "Code: using the clone at $PROJECT_DIR (branch $(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD))"
else
  PROJECT_DIR="$BASE_DIR/mas-deepr"
  if [[ -d "$PROJECT_DIR/.git" ]]; then
    say "Code: updating $PROJECT_DIR to origin/$BRANCH"
    git -C "$PROJECT_DIR" fetch origin "$BRANCH"
    git -C "$PROJECT_DIR" checkout "$BRANCH"
    git -C "$PROJECT_DIR" pull --ff-only origin "$BRANCH"
  else
    say "Code: cloning $REPO_URL ($BRANCH) into $PROJECT_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
  fi
fi
cd "$PROJECT_DIR"

# Same cache location scripts/run_queue.py uses (ART defaults to /tmp,
# which is wiped on reboot).
export ART_MEGATRON_CACHE_ROOT="$PROJECT_DIR/.cache/art"
export HF_HOME="$ART_MEGATRON_CACHE_ROOT/huggingface"
mkdir -p "$ART_MEGATRON_CACHE_ROOT"

# --- 4. Python environment ---------------------------------------------------
say "Python 3.12 environment .venv-gpu"
PY="$PROJECT_DIR/.venv-gpu/bin/python"
if [[ ! -x "$PY" ]]; then
  uv venv .venv-gpu --python 3.12
fi
OVERRIDES="$(mktemp)"
trap 'rm -f "$OVERRIDES"' EXIT
# ART 0.5.20 pins wandb==0.28; the project needs >=0.30 (W&B logging) --
# checked to resolve and to work together.
echo "wandb>=0.30.0" >"$OVERRIDES"
uv pip install --python "$PY" \
  --override "$OVERRIDES" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -e ".[r2]" "openpipe-art[backend]==$ART_VERSION"
"$PY" -c "import art, mas_deepr, torch, wandb; from art.local import LocalBackend; \
print('art', art.__version__ if hasattr(art, '__version__') else '?', \
'| torch', torch.__version__, '| wandb', wandb.__version__)"

# --- 5. Chromium for crawl4ai (the fetch_page tool) ----------------------------
say "Chromium for crawl4ai"
if ! "$PROJECT_DIR/.venv-gpu/bin/crawl4ai-setup"; then
  warn "crawl4ai-setup failed -- installing Chromium's system libraries and retrying"
  $SUDO "$PY" -m playwright install-deps chromium
  "$PROJECT_DIR/.venv-gpu/bin/crawl4ai-setup"
fi

# --- 6. GPU ------------------------------------------------------------------
say "GPU"
HAS_GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  "$PY" -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; \
print('torch CUDA OK:', torch.cuda.get_device_name(0))"
else
  warn "no working nvidia-smi -- fine on a CPU-only setup machine you'll \
snapshot, but the training VM needs an image with the NVIDIA driver"
fi

# --- 7. ART's vLLM runtime + model weights (~20 GB, cached on disk) ------------
say "ART vLLM runtime and $BASE_MODEL weights"
if ! "$PY" - <<PYEOF
from art.utils.cache_dirs import configure_model_cache_env
from art.vllm_runtime import ensure_vllm_runtime

configure_model_cache_env()  # HF_HOME etc. under \$ART_MEGATRON_CACHE_ROOT
print("vLLM runtime:", ensure_vllm_runtime(progress=print))

from huggingface_hub import snapshot_download

print("weights:", snapshot_download("$BASE_MODEL"))
PYEOF
then
  if [[ $HAS_GPU -eq 1 ]]; then
    echo "vLLM runtime / weights download failed" >&2
    exit 1
  fi
  warn "vLLM runtime / weights step failed without a GPU -- ART will retry \
on the first GPU run (slower start, nothing else lost)"
fi

# --- 8. datasets (Hugging Face -> assets/data) ---------------------------------
say "Training and evaluation datasets"
"$PY" - <<'PYEOF'
from mas_deepr.config import get_settings
from mas_deepr.data import (
    build_train_pool,
    load_dr_tulu_paper_rubrics,
    load_frames,
    load_health_bench,
    load_research_qa,
)

settings = get_settings()
failed = []
for name, load in [
    ("MuSiQue + HotpotQA", lambda: build_train_pool(settings)),
    ("DR Tulu rubrics", lambda: load_dr_tulu_paper_rubrics(settings)),
    ("FRAMES", lambda: load_frames(settings)),
    ("ResearchQA", lambda: load_research_qa(settings)),
    ("HealthBench", lambda: load_health_bench(settings)),
]:
    try:
        print(f"{name}: {len(load())} questions")
    except Exception as exc:  # noqa: BLE001 -- report all, then fail
        failed.append(name)
        print(f"{name}: FAILED -- {exc!r}")
if failed:
    raise SystemExit(f"dataset download failed: {', '.join(failed)}")
PYEOF

# --- 9. .env -----------------------------------------------------------------
say ".env (key names only -- values are never printed)"
if [[ ! -f .env ]]; then
  warn "no .env in $PROJECT_DIR yet -- copy yours here, then rerun this script"
else
  missing=0
  for key in WANDB_API_KEY CLOUDFLARE_ACCOUNT_ID R2_ACCESS_KEY_ID \
             R2_SECRET_ACCESS_KEY R2_BUCKET_NAME; do
    if grep -Eq "^${key}=.+" .env; then
      echo "  ok       $key"
    else
      echo "  MISSING  $key"; missing=1
    fi
  done
  if grep -Eq "^SEMANTIC_SCHOLAR_API_KEY=.+" .env; then
    echo "  ok       SEMANTIC_SCHOLAR_API_KEY (optional)"
  else
    echo "  -        SEMANTIC_SCHOLAR_API_KEY (optional, not set: slower, rate-limited)"
  fi
  if [[ $missing -eq 1 ]]; then warn "fill in the MISSING keys before running the queue"; fi
fi

say "Done. Next:"
cat <<EOM
  cd $PROJECT_DIR
  tmux new -s queue
  .venv-gpu/bin/python scripts/run_queue.py --smoke        # ~15-30 min check
  .venv-gpu/bin/python scripts/run_queue.py --shutdown     # the real run
  (detach from tmux with Ctrl-b d; reattach with: tmux attach -t queue)
EOM
