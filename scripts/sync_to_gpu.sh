#!/usr/bin/env bash
# Push code, configs and docs from the Mac to the A10 box.
#
#   ./scripts/sync_to_gpu.sh --dry-run     # ALWAYS do this first
#   ./scripts/sync_to_gpu.sh
#
# The exclude list is the important part of this script. Syncing data/ or artifacts/
# would move tens of gigabytes of datasets and checkpoints over SSH — slowly, and for
# no reason, since both are regenerated on the box.
#
# Set the host once in ~/.ssh/config as described in docs/setup/remote-gpu.md, then
# this needs no arguments.

set -euo pipefail

HOST="${FT_GPU_HOST:-a10}"                                  # override: FT_GPU_HOST=...
REMOTE_DIR="${FT_GPU_DIR:-~/LLM_fine_tuning_from_zero_to_hero}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- what NOT to send ---
EXCLUDES=(
    # regenerated on the box, and huge
    '--exclude=data/'
    '--exclude=artifacts/'
    '--exclude=checkpoints/'
    '--exclude=books/'          # reference PDFs; read them locally
    # environments are per-machine — the box has its own FT_VENV
    '--exclude=.venv*'
    '--exclude=__pycache__/'
    '--exclude=*.pyc'
    '--exclude=.pytest_cache/'
    '--exclude=.ruff_cache/'
    # local noise
    '--exclude=.git/'
    '--exclude=.DS_Store'
    '--exclude=.vscode/'
    '--exclude=*.egg-info/'
    # notebook outputs bloat the diff and are never needed on the box
    '--exclude=.ipynb_checkpoints/'
)

DRY_RUN=()
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
    DRY_RUN=(--dry-run)
    echo ">>> DRY RUN — nothing will be transferred"
fi

echo ">>> $LOCAL_DIR  ->  $HOST:$REMOTE_DIR"
echo

# -a archive, -v verbose, -z compress, -h human-readable sizes
# --delete keeps the box a mirror: a file you deleted locally goes away there too.
#          Safe here ONLY because data/ and artifacts/ are excluded above.
rsync -avzh "${DRY_RUN[@]}" --delete \
    "${EXCLUDES[@]}" \
    "$LOCAL_DIR/" \
    "$HOST:$REMOTE_DIR/"

echo
if [[ ${#DRY_RUN[@]} -gt 0 ]]; then
    echo ">>> Dry run complete. Check that NOTHING from data/ or artifacts/ appeared above."
else
    echo ">>> Synced. Next:"
    echo "      ssh $HOST"
    echo "      tmux new -s ft"
    echo "      cd $REMOTE_DIR && pyenv activate FT_VENV"
fi

# --- note on rsync vs git ---
# Use this while iterating: it is a much faster loop.
# But from Phase 04 onward, a run's git SHA is part of its identity (see the run-id
# contract in docs/decisions/0001-*.md). So for any run whose number you will report,
# commit first and `git pull` on the box instead, so the SHA in run_card.md is real.
