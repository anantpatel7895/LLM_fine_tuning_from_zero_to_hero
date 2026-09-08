#!/usr/bin/env bash
# Launch a real training run with a proper run record.
#
#   tmux new -s ft
#   ./scripts/launch.sh configs/sft/llama31-8b-qlora.yaml
#
# Two jobs: (1) generate the run-id and write run_card.md BEFORE training starts, so the
# record exists even if the run dies, and (2) tee all output to the run directory.
#
# The run-id contract (ADR 0001):
#   {date}-{phase}-{method}-{model}-{confighash8}
#   20260907-p04-qlora-llama31-8b-a3f91c2e
#
# Why the record is written first: from Phase 04 onward every reported number must trace
# back to a config, a commit and a dataset version. Retrofitting that is impossible, and
# by Phase 15 you will have dozens of runs. See docs/decisions/0001-*.md.

set -euo pipefail

CONFIG="${1:-}"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    echo "usage: $0 <config.yaml>" >&2
    exit 1
fi

# --- refuse to run outside tmux ---
# An SSH drop kills a foreground process and takes the run with it. This has cost
# everyone hours at least once.
if [[ -z "${TMUX:-}" && "${FT_ALLOW_NO_TMUX:-0}" != "1" ]]; then
    echo "error: not inside tmux. A dropped SSH connection would kill this run." >&2
    echo "       run:  tmux new -s ft" >&2
    echo "       (to override anyway: FT_ALLOW_NO_TMUX=1 $0 $CONFIG)" >&2
    exit 1
fi

# The trainer this calls is built in Phase 04.
if ! python -c "import llm_finetuning_from_zero_to_hero.train.sft" 2>/dev/null; then
    echo "note: the SFT trainer does not exist yet — it is built in Phase 04." >&2
    echo "      See docs/phases/phase-04.md" >&2
    exit 2
fi

# --- build the run id ---
CONFIG_HASH="$(shasum -a 256 "$CONFIG" | cut -c1-8)"
DATE="$(date +%Y%m%d)"
PHASE="$(grep -oE '^phase:[[:space:]]*[0-9]+' "$CONFIG" 2>/dev/null | grep -oE '[0-9]+' || echo "xx")"
METHOD="$(grep -oE '^method:[[:space:]]*[a-z0-9_]+' "$CONFIG" 2>/dev/null | awk '{print $2}' || echo "run")"
MODEL_SLUG="$(grep -oE '^model_name_or_path:.*' "$CONFIG" 2>/dev/null \
    | awk '{print $2}' | tr -d '"' | awk -F/ '{print tolower($NF)}' | tr '.' '-' || echo "unknown")"

RUN_ID="${DATE}-p${PHASE}-${METHOD}-${MODEL_SLUG}-${CONFIG_HASH}"
RUN_DIR="artifacts/${RUN_ID}"

if [[ -d "$RUN_DIR" ]]; then
    echo "error: $RUN_DIR already exists — that config already ran today." >&2
    echo "       Change the config (which changes the hash) or move the old directory." >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

# --- freeze the config as it was at launch ---
cp "$CONFIG" "$RUN_DIR/config.yaml"

# --- write the run card BEFORE training ---
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
GIT_DIRTY="clean"
if ! git diff --quiet HEAD 2>/dev/null; then
    GIT_DIRTY="DIRTY"
fi

cat > "$RUN_DIR/run_card.md" <<EOF
# Run \`$RUN_ID\`

**Started:** $(date -u '+%Y-%m-%d %H:%M:%S UTC')
**Phase:** $PHASE · **Method:** $METHOD

## Provenance

| | |
|---|---|
| config | \`$CONFIG\` (frozen copy: \`config.yaml\`) |
| config sha256[:8] | \`$CONFIG_HASH\` |
| git SHA | \`$GIT_SHA\` |
| working tree | **$GIT_DIRTY** |
| host | $(hostname) |
| python | $(python -c 'import sys; print(sys.version.split()[0])') |
| torch | $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '?') |
| CUDA (torch) | $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '?') |
| transformers | $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo '?') |
| trl | $(python -c 'import trl; print(trl.__version__)' 2>/dev/null || echo '?') |
| peft | $(python -c 'import peft; print(peft.__version__)' 2>/dev/null || echo '?') |

$(if [[ "$GIT_DIRTY" == "DIRTY" ]]; then
    echo '> ⚠️ **The working tree was dirty at launch.** The git SHA above does not fully'
    echo '> describe this run. Do not report a number from it — commit and re-run.'
fi)

## GPU state at launch

\`\`\`
$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.sm,temperature.gpu --format=csv 2>/dev/null || echo 'nvidia-smi unavailable')
\`\`\`

## Predictions (rule 2 — fill these in BEFORE reading results)

| Quantity | Predicted | Measured | Gap, and why |
|---|---|---|---|
| Peak VRAM | | | |
| Tokens/sec | | | |
| Step time | | | |
| Wall-clock | | | |

## Result

_(fill in: final loss, eval metric, the number that moved)_

## What surprised me

_(raw material for docs/LESSONS.md)_
EOF

echo "=============================================================="
echo " RUN  $RUN_ID"
echo "   dir  : $RUN_DIR"
echo "   git  : ${GIT_SHA:0:8} ($GIT_DIRTY)"
echo "=============================================================="
if [[ "$GIT_DIRTY" == "DIRTY" ]]; then
    echo
    echo " ⚠️  Working tree is dirty. Fine for exploration; do not report a number."
fi
echo
echo " Detach with: ctrl-b then d       Reattach: tmux attach -t ft"
echo

python -m llm_finetuning_from_zero_to_hero.train.sft \
    --config "$CONFIG" \
    --override "output_dir=$RUN_DIR" \
    --run-id "$RUN_ID" \
    2>&1 | tee "$RUN_DIR/train.log"

echo
echo "=============================================================="
echo " DONE — now fill in the Predictions and Result tables in:"
echo "   $RUN_DIR/run_card.md"
echo "=============================================================="
