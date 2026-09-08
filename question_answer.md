# question_answer.md — running recall log

**Why this file is the real deliverable.** The adapters and the eval reports are artifacts; this file is the actual skill. A phase is not complete until you can answer its questions **without looking anything up.**

**Numbering:** `Q<phase>.<n>` — e.g. `Q03.2` is the second question from Phase 03.

**Legend**

| Tag | Meaning |
|---|---|
| ⭐ | Asked in interviews |
| 🔧 | Needed during a real incident |
| 📐 | **Must be derivable on a whiteboard, cold, no notes** — see the checklist in [NUMBERS.md](NUMBERS.md) |

The 📐 tag matters more here than in other repos: fine-tuning interviews are disproportionately derivation questions, and the derivations are all in `NUMBERS.md`.

**Workflow:** write the question when it comes up during a phase. Write the answer *from memory* at the end of the phase. If you cannot, the phase is not finished — go back.

---

## Phase 00 — Environment & hardware baseline

**Q00.1 📐⭐** — How much VRAM does a full fine-tune of an 8B model need with AdamW mixed precision, and where does every byte go?

> _(answer from memory)_

**Q00.2 ⭐** — Why does bf16 need no loss scaling while fp16 does?

> _(answer from memory)_

**Q00.3 🔧** — Why is 24 GB of VRAM never actually 24 GB? Name four consumers.

> _(answer from memory)_

**Q00.4 📐** — What is the A10's ridge point, and what does it tell you about optimizing training versus optimizing decode?

> _(answer from memory)_

**Q00.5 ⭐** — Your GPU reports compute capability `(8, 6)`. Name three things that rules out.

> _(answer from memory)_

---

## Phase 01 — Bridge: your GPT-2 loop → the HF stack

_(add during the phase)_

---

## Phase 02 — Tokenizers, chat templates & special tokens

_(add during the phase)_

---

## Phase 03 — Loss masking & the anatomy of an example

_(add during the phase)_

---

## Phase 04 — First real QLoRA SFT run

_(add during the phase)_

---

## Phase 05 — Evaluation before optimization

_(add during the phase)_

---

## Phase 06 — LoRA/QLoRA mechanics & ablations

_(add during the phase)_

---

## Phase 07 — Data prep I: sourcing, quality, dedup, decontamination

_(add during the phase)_

---

## Phase 08 — Data prep II: synthetic generation & curation

_(add during the phase)_

---

## Phase 09 — Data prep III: mixing, ordering, how much is enough

_(add during the phase)_

---

## Phase 10 — Multi-GPU on 2×A10 without NVLink

_(add during the phase)_

---

## Phase 11 — Full fine-tuning & continued pre-training

_(add during the phase)_

---

## Phase 12 — Debugging & training pathologies

_(add during the phase)_

---

## Phase 13 — Preference optimization: DPO & family

_(add during the phase)_

---

## Phase 14 — Reward modeling

_(add during the phase)_

---

## Phase 15 — GRPO & verifiable rewards

_(add during the phase)_

---

## Phase 16 — Merging, quantization & serving

_(add during the phase)_

---

## Phase 17 — Accelerator frameworks & reproducibility

_(add during the phase)_

---

## Phase 18 — Capstones & portfolio

_(add during the phase)_
