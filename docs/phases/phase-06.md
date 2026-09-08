# Phase 06 — LoRA/QLoRA mechanics & systematic ablations

**~10 hrs** · Prerequisite: [Phase 05](phase-05.md) · Next: [Phase 07](phase-07.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **predict which LoRA configuration will win before you run it, and explain the cases where your prediction was wrong**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `experiments/lora-grid/results.md` with ≥8 runs; a merged model; a one-page *what I'd choose and when* decision table (the most interview-usable artifact in the repo)

**Models:** Qwen3-1.7B for the grid (≤45 min per run), Llama-3.1-8B to confirm the top two

**Data:** the Phase 04 dataset, unchanged — rule 6, one variable per run

---

## 1. Concept

- Why ΔW is *approximately* low-rank, and the intrinsic-dimension argument behind LoRA
- The alpha/rank scaling. `alpha = 2r` is a **convention, not a law**. **rsLoRA** uses α/√r and matters specifically at high rank
- **Target modules.** Attention-only vs all-linear. From `NUMBERS.md` §5, the MLP holds **67%** of the LoRA-targetable parameters — which is the arithmetic behind the empirical finding that all-linear usually wins. Attention adapters move *style*; MLP adapters move *knowledge*
- **DoRA** (magnitude/direction decomposition), **LoRA+** (higher LR on B than A), and the initialization schemes: PiSSA, OLoRA, LoftQ
- **QLoRA's honest cost**: dequantization overhead makes it meaningfully *slower* than bf16 LoRA on Ampere while saving ~12 GB on an 8B. So on this box QLoRA is a **capacity** tool (14B+), not a speed tool. Derive this yourself from the measurements
- **Merging**: `merge_and_unload()`. Why merging a QLoRA adapter into a bf16 base loses accuracy — the adapter was trained against *quantized* weights, so merging into dequantized weights leaves a residual error. And multi-adapter merging: linear, TIES, DARE

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

- A one-paragraph awareness note on VeRA / LoHa / LoKr, with an explicit **'not running these'**. That restraint is the anti-tool-tourism rule in action

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- The grid, launched as one overnight sweep on the 1.7B: `r ∈ {8, 16, 64, 256}` × `targets ∈ {q,v | all-attn | all-linear}` × `{LoRA, rsLoRA, DoRA}` × `{bf16, QLoRA}` — sampled, not exhaustive
- **Write your predicted ranking down before the sweep runs.** Then compare
- Confirm the top two on the 8B, scored with the Phase 05 harness including confidence intervals
- Compute the trainable-parameter count by hand for your winner and check it against `print_trainable_parameters()`
- Switch tracking to Weights & Biases here — eight runs is where a comparison UI starts to earn its keep

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

_(append to [question_answer.md](../../question_answer.md) as `Q06.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
