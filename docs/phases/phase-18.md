# Phase 18 — Capstones & portfolio

**~12 hrs** · Prerequisite: [Phase 17](phase-17.md) · Next: —

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **ship three artifacts that a hiring manager can evaluate in five minutes**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** three portfolio artifacts, each with a URL

**Models:** the golden-thread model, fully assembled

**Data:** everything built in Phases 07–09

---

## 1. Concept

- This phase is assembly and communication, not new technique. Everything here was built earlier; the work is making it legible to someone who was not there

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

- Where to publish, and what a good model card actually contains

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- **Capstone A — the golden-thread model.** Full lineage documented: data pipeline → SFT → DPO → GRPO → eval → quantize → serve, with a lineage diagram and the number that moved at each step. Published to the Hub with a real model card and an honest limitations section. **Consider composing this with your existing RAG repo** — plug the fine-tuned generator into that system and report end-to-end answer quality with vs without the fine-tune, using the golden eval set you built there months earlier. *'My fine-tune improved a production RAG system by N points on a pre-existing eval set'* is worth more than the model itself, and almost nobody can show it
- **Capstone B — a released dataset.** The data-prep half of the goal, and the rarer half. Generation code, dedup and decontamination stats, a proper datasheet, and **an ablation proving it beats a naive baseline**
- **Capstone C — the cookbook README.** A five-minute read linking every phase artifact. Written for someone who will not read the phase docs

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

_(append to [question_answer.md](../../question_answer.md) as `Q18.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
