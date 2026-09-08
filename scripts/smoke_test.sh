#!/usr/bin/env bash
# The gate every recipe must pass before you spend real GPU time on it.
#
#   ./scripts/smoke_test.sh configs/sft/llama31-8b-qlora.yaml
#
# Runs the given recipe for ~50 steps on a sub-1B model with a tiny slice of data, so a
# broken recipe fails in about a minute instead of three hours in.
#
# Rule 4 of PLAN.md: a recipe that has not smoke-tested is not a recipe, it is a hope.
#
# What this catches, cheaply:
#   - template or masking errors        (Phase 03)
#   - target_modules matching nothing   (trainable params = 0.00%)
#   - dataset column-name mistakes
#   - OOM from an obviously wrong batch/sequence combination
#   - a config key the trainer silently ignores
#
# What it does NOT catch: anything that only appears at scale — a length outlier at step
# 4000, a loss spike, catastrophic forgetting. Those are Phase 12.

set -euo pipefail

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: $0 <config.yaml> [--steps N]" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "error: config not found: $CONFIG" >&2
    exit 1
fi

# The trainer this calls is built in Phase 04. Say so plainly rather than emitting a
# ModuleNotFoundError traceback.
if ! python -c "import llm_finetuning_from_zero_to_hero.train.sft" 2>/dev/null; then
    echo "note: the SFT trainer does not exist yet — it is built in Phase 04." >&2
    echo "      This script is the gate that Phase 04 onward runs through." >&2
    echo "      See docs/phases/phase-04.md" >&2
    exit 2
fi

STEPS="${SMOKE_STEPS:-50}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
SMOKE_ROWS="${SMOKE_ROWS:-256}"
OUT="artifacts/smoke/$(basename "${CONFIG%.*}")-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT"

echo "=============================================================="
echo " SMOKE TEST"
echo "   config : $CONFIG"
echo "   model  : $SMOKE_MODEL   (overriding whatever the config says)"
echo "   steps  : $STEPS"
echo "   rows   : $SMOKE_ROWS"
echo "   out    : $OUT"
echo "=============================================================="
echo

# Force the cheap settings regardless of what the config asks for. The point is to
# exercise the code path, not to train anything.
#
# NOTE: --report_to none matters here. Otherwise every smoke test litters your
# TensorBoard / W&B with 50-step junk runs that make the real ones hard to find.
python -m llm_finetuning_from_zero_to_hero.train.sft \
    --config "$CONFIG" \
    --override "model_name_or_path=$SMOKE_MODEL" \
    --override "max_steps=$STEPS" \
    --override "max_train_samples=$SMOKE_ROWS" \
    --override "per_device_train_batch_size=1" \
    --override "gradient_accumulation_steps=1" \
    --override "save_strategy=no" \
    --override "report_to=none" \
    --override "output_dir=$OUT" \
    --smoke \
    2>&1 | tee "$OUT/smoke.log"

# --smoke additionally makes sft.py print the loss-mask inspection for the first 3
# examples (Phase 03) and assert that trainable params > 0. Those two checks find most
# silent failures on their own.

echo
echo "=============================================================="
echo " SMOKE TEST PASSED"
echo "=============================================================="
echo
echo "Before launching the real run, confirm from the log above:"
echo "  [ ] trainable params is NOT 0.00%"
echo "  [ ] the loss-mask dump shows loss on assistant tokens ONLY,"
echo "      and the terminating EOS token IS in loss"
echo "  [ ] loss went down at all (it need not go down much in 50 steps)"
echo "  [ ] peak VRAM is roughly what you predicted in the phase doc"
echo
echo "Then:  tmux new -s ft  &&  ./scripts/launch.sh $CONFIG"
