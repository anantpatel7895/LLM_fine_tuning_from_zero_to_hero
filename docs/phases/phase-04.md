# Phase 04 — Your first real QLoRA SFT run

**~8 hrs** · Prerequisite: [Phase 03](phase-03.md) · Next: [Phase 05](phase-05.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **run a supervised fine-tune end to end and justify every hyperparameter from a derivation rather than from a tutorial**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `src/.../train/sft.py` + `configs/sft/llama31-8b-qlora.yaml` + the first adapter + `artifacts/<run-id>/run_card.md` + 20 side-by-side base-vs-tuned generations

**Models:** Llama-3.1-8B-Instruct (QLoRA, one A10); Qwen3-0.6B for the smoke test

**Data:** a small human-written instruction set (~10k examples, ~8M tokens) — small enough for a ~3 hr epoch and free of the synthetic-slop and self-contamination that would confound your very first measurement

---

## 1. Concept

- NF4 quantization, double quantization, and `bnb_4bit_compute_dtype` — what is stored in 4 bits versus what the matmul actually runs in
- `LoraConfig`: rank, alpha, target modules, dropout. Just enough to run; Phase 06 does the ablation
- **Why LoRA's learning rate is ~1e-4 while full fine-tuning's is ~1e-5** — the adapter starts at exactly zero and has to travel, and the alpha/rank scaling interacts with the LR
- Warmup ratio, cosine vs linear vs WSD schedules, weight decay ≈ 0 on LoRA, `max_grad_norm=1.0`
- `gradient_checkpointing=True` + `use_reentrant=False` + `model.config.use_cache=False` — and what breaks if you forget each one
- Effective batch = micro × accum × world_size, but **track tokens/step, not examples/step**

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

- QLoRA vs bf16 LoRA for a first run. **Choice: QLoRA**, because ~10.5 GiB peak at sequence 2048 leaves real headroom to be careless while you are still learning the plumbing, whereas bf16 LoRA on the same model is a 20.1 GiB configuration at micro-batch 1 that one long example will OOM. Phase 06 measures what QLoRA costs you in speed
- Which model to run *first*. **Consider `Mistral-7B-v0.3` before Llama** — its trunk is identical in size to Llama-3.1-8B's, but its 32k vocabulary makes the loss head 4× cheaper, so the whole run is ~7.9 GiB. See the identical-trunk comparison at the top of [NUMBERS.md](../../NUMBERS.md)

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- **Pre-register the numbers (rule 2).** Before launching, write your predicted peak VRAM, tokens/sec and epoch time into this doc, derived from `NUMBERS.md` §1–4
- Smoke test on the 0.6B for 50 steps. Only then launch the real run into tmux
- Write `run_card.md` from this run onward — git SHA + dirty flag, config hash, dataset hash, resolved hyperparameters, `nvidia-smi` snapshot, predicted vs measured. **This cannot be retrofitted**
- Generate on 20 held-out prompts with the base model and the tuned model. Put them side by side in this doc
- Push the adapter to the Hub with a model card that says honestly: *not yet evaluated — see Phase 05*

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

_(append to [question_answer.md](../../question_answer.md) as `Q04.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
