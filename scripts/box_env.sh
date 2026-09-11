# Environment for the A10 box. SOURCE this, do not execute it:
#
#   source ./scripts/box_env.sh
#
# Make it permanent once you are happy with it:
#
#   echo "source ~/LLM_fine_tuning_from_zero_to_hero/scripts/box_env.sh" >> ~/.bashrc
#
# WHY THIS FILE EXISTS
# --------------------
# This box has two disks, and the small one is the dangerous one:
#
#   /              /dev/sda1   98 G   95% full    <- home, .pyenv, .cache all live here
#   ~/slm-models   /dev/sdb   197 G   80% full    <- the data disk, 40 G free
#
# Every HuggingFace and pip default points at $HOME, i.e. the nearly-full disk. One
# `from_pretrained("meta-llama/Llama-3.1-8B")` would try to write 16 GB into 5 GB of
# space, fail partway, and leave a corrupt cache behind. These variables redirect the
# large writes onto the data disk BEFORE that can happen.

# --- refuse to run as a subprocess ------------------------------------------------------
# If this file is EXECUTED (./scripts/box_env.sh) rather than SOURCED, every `export`
# below applies to a child shell that exits one millisecond later. The mkdir calls take
# effect, so it looks like it worked -- but the calling shell gets nothing. That silent
# half-success is exactly why this guard exists.
#
# Environment variables travel DOWN the process tree, never UP: a child receives a *copy*
# of its parent's environment and cannot write back into it. `source` sidesteps this by
# running the commands in the current shell rather than forking one.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: this file must be SOURCED, not executed." >&2
    echo >&2
    echo "  You ran:   ./scripts/box_env.sh" >&2
    echo "  Run this:  source ./scripts/box_env.sh" >&2
    echo >&2
    echo "Executing it forks a child shell, exports into that child, and then the child" >&2
    echo "exits -- taking every variable with it. Your shell would be left unchanged," >&2
    echo "HF_HOME would stay unset, and the next model download would target the" >&2
    echo "nearly-full root disk." >&2
    echo >&2
    echo "Verify after sourcing:  echo \$HF_HOME" >&2
    exit 1
fi

# --- make the package importable --------------------------------------------------------
# src/ layout with no pyproject.toml, so nothing is pip-installed. Without this,
# `python -m llm_finetuning_from_zero_to_hero...` fails with ModuleNotFoundError.
# Derived from this file's own location rather than $PWD, so it is correct regardless of
# which directory you source it from.
FT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FT_REPO
export PYTHONPATH="$FT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# --- where the big things go ------------------------------------------------------------
FT_DATA_DISK="${FT_DATA_DISK:-$HOME/slm-models}"

# HF_HOME relocates the whole HuggingFace tree: model cache, datasets cache, and tokens.
# Setting this one variable is enough; HF_DATASETS_CACHE and TRANSFORMERS_CACHE are both
# derived from it. (TRANSFORMERS_CACHE is deprecated in transformers v5 and silently
# ignored -- see docs/notes/api-2026.md. Do not set it.)
export HF_HOME="$FT_DATA_DISK/hf_home"

# pip downloads and unpacks wheels here. torch + CUDA is ~6-8 GB per install attempt, and
# the cache keeps a copy of every wheel it has ever fetched, so this grows fast.
export PIP_CACHE_DIR="$FT_DATA_DISK/pip_cache"

# Build/extract scratch. pip unpacks wheels through TMPDIR, so leaving it on /tmp (which
# is on the root disk) can fill / even when PIP_CACHE_DIR is elsewhere.
export TMPDIR="$FT_DATA_DISK/tmp"

# Training outputs: checkpoints and adapters. A resumable 8B full-FT checkpoint is 48 GB
# (NUMBERS.md 12), so this must never be on the root disk.
export FT_ARTIFACTS="$FT_DATA_DISK/artifacts"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$FT_ARTIFACTS"

# --- models already on this box ----------------------------------------------------------
# Llama-3.1-8B-Instruct and Mistral-7B-v0.3 were already downloaded here before this
# curriculum started. Point at them directly rather than re-downloading 29 GB.
export FT_LLAMA_8B="$FT_DATA_DISK/base_models/llama_3_1_8b_instruct"
export FT_MISTRAL_7B="$FT_DATA_DISK/base_models/Mistral-7B-Instruct-v0.3"

# --- behaviour --------------------------------------------------------------------------
# Faster Hub downloads (Rust transfer backend). Harmless if hf_transfer is absent.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Silences the tokenizers fork warning that appears once dataloader_num_workers > 0.
export TOKENIZERS_PARALLELISM=false

# Lets the caching allocator grow segments instead of fragmenting into unusable blocks.
# This is the mitigation for "OOM at step 4000 having been fine at step 1" -- see
# question_answer.md Q00.3 consumer 3.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- report -----------------------------------------------------------------------------
echo "box_env: repo            $FT_REPO"
echo "box_env: PYTHONPATH      $PYTHONPATH"
echo "box_env: data disk       $FT_DATA_DISK"
echo "box_env: HF_HOME         $HF_HOME"
echo "box_env: PIP_CACHE_DIR   $PIP_CACHE_DIR"
echo "box_env: TMPDIR          $TMPDIR"
echo "box_env: free on data    $(df -h "$FT_DATA_DISK" 2>/dev/null | awk 'NR==2 {print $4}')"
echo "box_env: free on /       $(df -h / 2>/dev/null | awk 'NR==2 {print $4}')"

# Interpreter sanity check -- the L1 trap. These two MUST agree.
_ft_py="$(python -V 2>&1 | awk '{print $2}')"
_ft_pip="$(python -m pip -V 2>&1 | grep -oE 'python [0-9]+\.[0-9]+' | awk '{print $2}')"
if [ -n "$_ft_pip" ] && [ "${_ft_py%.*}" != "$_ft_pip" ]; then
    echo "box_env: WARNING python is $_ft_py but 'python -m pip' targets $_ft_pip"
    echo "box_env:         see docs/LESSONS.md L1 before installing anything"
else
    echo "box_env: python $_ft_py, pip targets the same interpreter -- OK"
fi
unset _ft_py _ft_pip
