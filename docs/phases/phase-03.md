# Phase 03 — Loss masking & the anatomy of a training example

**~8 hrs** · Prerequisite: [Phase 02](phase-02.md) · Next: [Phase 04](phase-04.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **prove token-by-token which positions carry loss, and catch a broken mask before spending a single GPU-hour on it**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `src/.../data/mask_inspector.py` — prints a colorized `token | id | label | in_loss?` table for any example; tests that catch an off-by-one on the assistant header; `docs/DATA.md` Part 3 completed

**Models:** Qwen3-0.6B (fast, CPU-runnable for the inspection work)

**Data:** the same hand-written conversations from Phase 02

---

## 1. Concept

- `labels = -100` is how PyTorch's cross-entropy ignores a position. That single mechanism is the whole of loss masking
- **Completion-only loss**: mask the prompt, train on the answer. Why training on the prompt sometimes helps (more signal on short data) and usually hurts
- Multi-turn masking: all assistant turns vs the last only, and the train/serve mismatch you accept either way
- The model shifts labels internally. If you shift too, every label is off by one and **the loss still descends smoothly**
- Packing: concatenating examples to fill a sequence rather than wasting compute on padding. Older tutorials warn that this lets sample B attend to sample A — **that warning is out of date.** `packing_strategy="bfd"` is now the default and auto-enables `padding_free`, so packed sequences do not attend across boundaries. Learn the mechanism anyway (`position_ids` + variable-length attention), because *"how does packing avoid cross-attention between examples?"* is a common interview question
- **`DataCollatorForCompletionOnlyLM` has been removed from TRL.** Every tutorial using `response_template="### Assistant:"` describes machinery that no longer exists. See [docs/notes/api-2026.md §3](../notes/api-2026.md)
- Right-padding for training, **left**-padding for generation. Get it backwards and generation starts inside the pad region

_(Write the explanation here as you work through it. Anything you cannot explain in your own words is not yet learned.)_

---

## 2. Numbers

**Pre-register your predictions here BEFORE launching anything** — rule 2 of [PLAN.md](../../PLAN.md). The gap between prediction and measurement is the lesson.

| Quantity | Predicted | Measured | Gap — and why |
|---|---|---|---|
| Peak VRAM | | | |
| Tokens/sec | | | |
| Step time | | | |
| Wall-clock to finish | | | |

Formulas used: see [NUMBERS.md](../../NUMBERS.md).

---

## 3. Landscape & choice

- `assistant_only_loss=True` (needs `{% generation %}` markers from Phase 02) vs hand-rolled masking in a custom collator. **Do both** — hand-roll it first so you know what the flag does, then switch to the flag and assert the masks are identical

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Build the inspector. It must show, for any example: the decoded token, its id, its label, and whether it contributes to loss. Colorize in-loss positions
- **Then deliberately reproduce all six silent failures** from `docs/DATA.md` Part 3 — double BOS, EOS not in loss, training on user turns, `pad_token == eos_token`, double-shifted labels, padding side mismatch. Each one gets a test that catches it
- The acceptance test: on a 3-turn conversation, assert exactly the assistant content plus its terminating EOS is in-loss, and nothing else
- Train a 0.6B for 100 steps with the mask broken, then fixed. Generate from both. **Look at what the broken one does** — this is the phase's real lesson and it takes 10 minutes of GPU time

**How you will know it worked:** _(state the acceptance criterion before you build)_

---

## 5. Build & Q&A

### What was built

_(files, commands, run-ids)_

### Results

_(the table, the plot, the number that moved)_

### What surprised you

_(this is the raw material for [docs/LESSONS.md](../LESSONS.md) — anything that surprised you is a candidate lesson)_

### Questions

_(append to [question_answer.md](../../question_answer.md) as `Q03.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
