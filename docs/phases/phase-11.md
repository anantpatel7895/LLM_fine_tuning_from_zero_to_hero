# Phase 11 — Full fine-tuning & continued pre-training

**~10 hrs** · Prerequisite: [Phase 10](phase-10.md) · Next: [Phase 12](phase-12.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **state exactly when full fine-tuning or continued pre-training beats LoRA, and measure the forgetting each one causes**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** a full fine-tune of a 1–3B; a continued-pre-training run; a forgetting curve plotted across checkpoints; `docs/notes/cpt-vs-sft-vs-lora.md` with a decision tree

**Models:** Llama-3.2-1B or Qwen3-1.7B (full FT with ZeRO-2 across both cards)

**Data:** a raw-text domain corpus for CPT plus a replay mix from Phase 09

---

## 1. Concept

- Full fine-tuning hyperparameters: LR 1e-5–2e-5 and **why an order of magnitude below LoRA**, longer warmup, and the optimizer memory trade space (AdamW 12 B/param → 8-bit AdamW 2 → Adafactor's factored second moment)
- **Loss-spike recovery**: rollback to a checkpoint and skip the data window that caused it
- **Continued pre-training is a different job.** Raw text, no template, no masking — everything is in loss. Pack to full context. Whether to train embeddings. **Tokenizer extension** for a new domain (add tokens vs retrain merges; mean-init; fertility as the thing you are actually optimizing, with a direct effect on inference cost)
- **CPT destroys instruction-following.** The model becomes a text completer again. This shocks everyone the first time. The fix is the CPT → re-SFT sequence
- **Catastrophic forgetting, measured not asserted** — run the Phase 05 retention suite at several checkpoints and plot the curve. Then mitigate: replay at 1–10%, an LR ceiling, or LoRA instead of full FT as a forgetting mitigation in itself

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

- LoRA vs full fine-tuning **at matched wall-clock** on identical data. Expect them close on style tasks and full FT ahead on knowledge injection — then explain why in terms of *where each one writes*

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Run the full fine-tune with the Phase 10 ZeRO-2 config. Note that the 8B is out of reach on this box and say so honestly in the doc
- Run CPT on the domain corpus, then re-SFT, and measure instruction-following before and after each stage
- Plot the forgetting curve. This is the phase's real artifact
- Write the decision tree: *is the gap knowledge, format, or style?* The answer picks the method

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

_(append to [question_answer.md](../../question_answer.md) as `Q11.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
