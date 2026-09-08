# Phase 02 — Tokenizers, chat templates & special tokens

**~8 hrs** · Prerequisite: [Phase 01](phase-01.md) · Next: [Phase 03](phase-03.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **render, diff and hand-write a chat template, and prove token-for-token that your training string is the one you think it is**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `src/.../data/templates.py` — a registry mapping model → template → special-token contract; `tests/test_templates.py` with 8 invariants; the rendered comparison table written into `docs/DATA.md` Part 2

**Models:** Llama-3.1-8B-Instruct, Mistral-7B-v0.3, Qwen3-1.7B, gemma-3 (tokenizers only — no training in this phase, so it runs on the Mac)

**Data:** one hand-written 3-turn conversation, plus 20 rows of a tool-calling set for the `tools` field

---

## 1. Concept

- `apply_chat_template()` is a **Jinja program shipped inside `tokenizer_config.json`** — the template is part of the model artifact, not part of your code
- The three template families and what each costs in tokens: Llama-3 header blocks, Mistral `[INST]`, ChatML `<|im_start|>`
- Special tokens vs added tokens vs reserved tokens; `pad_token` vs `eos_token` and why conflating them causes a specific failure in Phase 03
- BPE vs SentencePiece, and **fertility** (tokens per word) as the metric that directly sets your inference cost
- The four-step protocol for adding a token: `add_special_tokens` → `resize_token_embeddings(pad_to_multiple_of=64)` → mean-init the new rows → `modules_to_save=['embed_tokens','lm_head']`

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

- Use the shipped template vs write your own. **Choice: write your own**, because `assistant_only_loss` in TRL needs `{% generation %}` markers that most community templates lack. A template you authored, in version control, with tests, is the deliverable
- Which model family to treat as canonical. **Choice: Llama-3.1** for the golden thread, with the other three as contrast drills

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Take one identical conversation. Render it through all four templates. **Diff the token IDs, not the strings** — string diffs hide the double-BOS and the token cost of each marker
- Build the registry: `templates.py` maps a model id to its template, its special-token contract, and its padding side
- Write the 8 invariants as pytest: exactly one BOS · every assistant turn ends with the EOS-equivalent · zero user/system tokens would be in-loss · round-trip decode equals the templated string · `add_generation_prompt=True` yields a strict prefix of the training sequence · fertility is within expected range · no unknown tokens · the `tools` field renders
- Adversarial close: hand-craft a conversation that breaks each invariant, and confirm the suite catches it

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

_(append to [question_answer.md](../../question_answer.md) as `Q02.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
