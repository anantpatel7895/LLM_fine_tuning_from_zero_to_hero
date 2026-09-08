# Phase 09 — Data prep III — mixing, ordering, and how much is enough

**~8 hrs** · Prerequisite: [Phase 08](phase-08.md) · Next: [Phase 10](phase-10.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **choose a data mixture by ablation rather than by vibe, and know how many examples you actually need**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** a 3-mixture ablation at matched token budget; a quality-vs-quantity result on your own data; the completed `docs/DATA.md` Part 7; an ADR on the chosen mixture

**Models:** Qwen3-1.7B for all mixtures

**Data:** the Phase 07 cleaned pool plus the Phase 08 synthetic set, blended at different ratios

---

## 1. Concept

- Per-domain weights and how each shifts behaviour. **Weight by tokens, not by examples** — tokens-per-source and epochs-per-source diverge sharply when average lengths differ
- **Replay / anchor data** — 1–10% general instruction data mixed in, which is your defence against the catastrophic forgetting measured in Phase 11
- Temperature-based upsampling of small high-value sources
- **Curriculum ordering** (easy→hard, short→long) tested against plain shuffling
- The LIMA question: how much data is enough, and does 1k excellent examples beat 10k mediocre ones from the same pool

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

- Curriculum ordering vs shuffling. **Report the honest result** — shuffling usually wins for SFT, and knowing that saves you weeks of pointless ordering work later

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Three mixtures at **matched token budget** — that matching is what makes the comparison mean anything
- The quality-vs-quantity run: top-1k by quality score vs 10k random **from the same source** at matched compute. Both directions happen in practice; knowing *when* each wins is the expertise
- Every run scored with the Phase 05 harness including confidence intervals, so you can tell a result from noise

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

_(append to [question_answer.md](../../question_answer.md) as `Q09.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
