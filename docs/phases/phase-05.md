# Phase 05 — Evaluation before optimization

**~10 hrs** · Prerequisite: [Phase 04](phase-04.md) · Next: [Phase 06](phase-06.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **build an evaluation that can actually detect a 3-point regression, and prove you did not break the base model in the process**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `src/.../eval/` — one command producing a JSON + markdown report; a frozen, versioned ~1000-item regression suite; a decontamination report; ADR on the eval contract

**Models:** the Phase 04 adapter vs the base Llama-3.1-8B-Instruct

**Data:** your own golden-thread eval set, plus public task benchmarks for retention (instruction-following, grade-school math, general knowledge)

---

## 1. Concept

- **Four layers, four different questions.** (1) Held-out loss — the only metric reading the same signal training reads, and it must use the *identical* template and mask or it means nothing. (2) Task evals — does it do the job. (3) LLM-as-judge — is it better, pairwise. (4) **Capability retention** — did you break anything that already worked
- Cross-model perplexity is **not comparable** — different tokenizers mean different token counts for the same text
- Judge protocol: position-swap to cancel position bias, explicit tie handling, length-bias control, and calibration against ~50 of your own human labels
- **Contamination**: 13-gram containment of your training set against every benchmark test split, plus embedding-nearest-neighbour for the paraphrase cases n-grams cannot see
- **Statistical power** — the derivation from `NUMBERS.md` §9. A 100-item eval cannot see a difference smaller than ~10 points. To claim a 3-point win you need ~1000 items

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

- `lm-evaluation-harness` vs `lighteval` vs hand-rolled. **Choice: a public harness for the retention benchmarks** (so your numbers are comparable to published ones) **plus a hand-rolled suite for the golden-thread task** (because nobody else's benchmark measures your task)
- Log-likelihood vs generative scoring, and why the same benchmark name gives different numbers under each

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Build the retention suite by hand: 60–100 prompts covering abilities the base model already has and must not lose — multi-turn coherence, refusal behaviour, code, non-English, format-following. **Almost nobody builds this, and it is what catches catastrophic forgetting in Phase 11**
- Freeze and version the eval sets. Commit them. An eval set that drifts makes every historical number meaningless
- Run the decontamination check against Phase 04's training data and report the rate per benchmark
- **Retro-apply the harness to Phase 04's adapter — and be ready for the answer to be 'no significant change'.** That outcome is the lesson, and it is exactly why data prep comes next

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

_(append to [question_answer.md](../../question_answer.md) as `Q05.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
