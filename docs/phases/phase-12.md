# Phase 12 — Debugging & training pathologies

**~8 hrs** · Prerequisite: [Phase 11](phase-11.md) · Next: [Phase 13](phase-13.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **reproduce, diagnose and fix eleven named training failures on purpose, so that no future incident is a first-time incident**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `docs/LESSONS.md` with ≥11 entries, each linked to a runnable repro under `experiments/failures/`; a symptom → cause → where-to-look → fix playbook

**Models:** Qwen3-0.6B — every mechanism is identical at 0.6B and the loop is 60 seconds

**Data:** deliberately corrupted slices of your own Phase 09 mixture

---

## 1. Concept

- This phase is **consolidation, not reference**. It lands after full fine-tuning because by now you will have hit five of these for real, so the fixes mean something

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

- Induce each failure deliberately, on purpose, at 0.6B

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- **The eleven, each induced on purpose:** (1) loss spike at step N — LR 10× too high, or one pathological sample. (2) NaN loss — fp16 without loss scaling. (3) OOM at step 4000 but fine before — one max-length outlier plus `group_by_length`. (4) OOM only during eval — generation without `no_grad`, or eval batch > train batch. (5) **loss descends, model learns nothing** — all labels `-100`, or `target_modules` matched nothing (`print_trainable_parameters()` says 0.00%). (6) never stops generating — EOS not in loss. (7) repeats forever — over-trained on short responses. (8) retention benchmark drops 12 points — too many epochs, or LR too high on full FT. (9) **great eval, terrible in production — train template ≠ inference template. This is the #1 real-world cause.** (10) loss differs with gradient accumulation — loss normalization across accumulation steps. (11) resume produces a different curve — LR schedule / RNG / dataloader state not restored
- **Tooling to make each visible:** grad-norm and per-layer gradient histograms; `torch.cuda.memory._record_memory_history()` → snapshot → the memory-timeline viewer, which lets you *see* fragmentation instead of guessing; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `nvidia-smi dmon`; `py-spy dump` on a hung run; `NCCL_DEBUG=INFO` + `TORCH_DISTRIBUTED_DEBUG=DETAIL` for the PCIe hangs you will eventually hit; `CUDA_LAUNCH_BLOCKING=1` for async error attribution
- Determinism: seed everything, then write the honest list of where determinism is impossible anyway

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

_(append to [question_answer.md](../../question_answer.md) as `Q12.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
