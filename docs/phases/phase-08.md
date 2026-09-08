# Phase 08 — Data prep II — synthetic generation & curation

**~10 hrs** · Prerequisite: [Phase 07](phase-07.md) · Next: [Phase 09](phase-09.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **generate synthetic training data whose quality you can prove, and measure the ways it degrades**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** a generation pipeline; 2–5k verified synthetic tool-calling examples pushed to the Hub as a dataset with a full datasheet; a rejected-examples analysis; diversity metrics plotted against temperature

**Models:** a strong open model as generator, served locally with vLLM on GPU 1 while GPU 0 trains — the first appearance of vLLM, foreshadowing Phase 16

**Data:** seed tasks and personas from Phase 07; hard negatives you construct yourself

---

## 1. Concept

- The generation methods and what each teaches: Self-Instruct (the basic bootstrap loop), Evol-Instruct (instruction complexity as a controllable axis), **Magpie** (prompt an aligned model with *only* the user-turn template prefix and let it hallucinate the instruction — which teaches viscerally that **the chat template is an attack surface**, closing the loop with Phase 02), persona-driven generation (coverage), rejection sampling with a verifier
- **Licensing is a first-class engineering constraint here, not a footnote.** Generator model, its license, and the derived-work implications go in the datasheet. Getting this right reads as senior
- GPU partitioning: `CUDA_VISIBLE_DEVICES` and `--gpu-memory-utilization` to run a generator and a trainer side by side
- Failure modes, all measurable: **mode collapse** (distinct-n, self-BLEU, embedding-space coverage — plot against temperature), teacher error propagation, format leakage from the generator's own template, and the trap of using the same model as generator *and* judge

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

- Verify synthetic tool calls with a **programmatic schema validator**, not a judge model. **Choice: programmatic**, because the golden thread was chosen precisely to make this possible — and it makes the same verifier reusable as the GRPO reward in Phase 15

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Generate, then **verify programmatically**: JSON-parseable, schema-valid, tool exists, arguments type-check. Everything that fails is kept in a rejects file, not discarded
- **Analyze the rejects.** What the generator gets wrong is more informative than what it gets right, and it tells you what your hard negatives should be
- Construct hard negatives deliberately: prompts where no tool is needed, ambiguous arguments, multi-tool sequences, unavailable tools
- Plot diversity against generation temperature and pick a point on that curve with a reason

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

_(append to [question_answer.md](../../question_answer.md) as `Q08.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
