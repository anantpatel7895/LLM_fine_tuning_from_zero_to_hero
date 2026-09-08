# Phase 07 — Data prep I — sourcing, quality, dedup, decontamination

**~10 hrs** · Prerequisite: [Phase 06](phase-06.md) · Next: [Phase 08](phase-08.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **turn several messy public sources into one clean, deduplicated, decontaminated, versioned dataset, and say what each filter cost in eval points**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `src/.../data/{schema,clean,dedup,decontaminate,quality}.py`; `data/processed/<name>/v1/` with `manifest.json` + `datasheet.md`; the completed filter funnel table in `docs/DATA.md` Part 5; an ADR on which filters earn their place

**Models:** Qwen3-1.7B for the cleaning ablation; a reward model in 4-bit as one of the three quality scorers

**Data:** ≥3 heterogeneous sources — one large and known-messy, one small and human-curated, one tool-calling set for the golden thread

---

## 1. Concept

- **Schema is a one-way door.** A field you did not capture cannot be backfilled without regenerating. Over-declare
- Token-based length filters, never character-based — characters per token varies ~3× across languages and code
- **Refusal and assistant-tic slop** — *'As an AI language model'*, *'Certainly!'*, *'I hope this helps'*. This filter has an outsized effect on perceived quality relative to its simplicity
- **Four levels of duplicate, four different tools:** exact (content hash) → fuzzy (MinHash-LSH, and you **derive** the band/row split from your Jaccard threshold per `NUMBERS.md` §10 rather than copying `num_perm=128`) → semantic (embed + threshold; catches paraphrases MinHash structurally cannot) → **cross-split** (train-vs-eval leakage, the one that silently inflates your own numbers)
- **Three independent quality signals**, because they disagree and the disagreement is informative: reward-model score, LLM-rubric score, and a model-intrinsic score (perplexity or instruction-following-difficulty) that needs no external model at all
- Diversity and complexity tagging, so you can filter for quality **without collapsing coverage** — the failure mode of naive quality filtering

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

- Which of the three quality signals to use. **Choice: compute all three, correlate them, and pick by measured eval delta.** The correlation itself is a result worth writing down

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- The funnel: every stage records rows in, rows out, tokens, and **measured eval delta**. A stage with no measured delta does not stay
- The headline experiment: identical LoRA on raw vs cleaned data at the 1.7B, scored with the Phase 05 harness. **This is the first real 'data work is training work' number you will own**
- Write the datasheet properly — motivation, composition, collection, preprocessing, uses, distribution, license. Licenses must propagate or you cannot release anything in Phase 18
- Decontaminate against your own Phase 05 eval suite, not just public benchmarks

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

_(append to [question_answer.md](../../question_answer.md) as `Q07.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
