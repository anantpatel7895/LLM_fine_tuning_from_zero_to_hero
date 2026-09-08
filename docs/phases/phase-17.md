# Phase 17 — Accelerator frameworks & reproducibility

**~8 hrs** · Prerequisite: [Phase 16](phase-16.md) · Next: [Phase 18](phase-18.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **say precisely what Unsloth, Axolotl and LLaMA-Factory hide, what that costs, and reproduce any run of yours from its run-id alone**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** a three-way bake-off against your hand-written Phase 04 baseline; `scripts/reproduce.sh <run-id>`; a published HF Hub model card; an ADR on framework adoption

**Models:** Llama-3.1-8B — re-running Phase 04 exactly

**Data:** the Phase 04 dataset, unchanged, so the comparison is clean

---

## 1. Concept

- **Reproduce Phase 04 exactly, three times.** Compare tokens/sec, peak VRAM, and — the part that matters — **eval score against your hand-written baseline**
- Then read *why* Unsloth is faster: fused Triton kernels, custom autograd, avoided recomputation, chunked cross-entropy. **Understanding the mechanism is the deliverable; the speedup is just evidence**
- Where each one stops. Unsloth: fastest single-GPU, weaker composable multi-GPU — a real limitation you can now *evaluate* because of Phase 10. Axolotl: mature FSDP/DeepSpeed, YAML-driven, the right pick for multi-GPU. LLaMA-Factory: widest method coverage plus a UI
- **Reproducibility discipline**: config-as-code (pydantic → YAML → hashed), full environment capture (requirements hash, torch/CUDA/driver, GPU model, git SHA + dirty flag), seed control, dataset version hash. Checkpoint hygiene: safetensors, adapter-only vs full per `NUMBERS.md` §12, `save_total_limit`, and what a *correct* resume needs (weights + optimizer + scheduler + RNG + dataloader position)
- Converting sharded DeepSpeed/FSDP checkpoints back to HF format
- Hub hygiene: a model card with `base_model`, license, dataset links, and eval results in the metadata block so the Hub renders them

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

- **The adoption rule, written as an ADR:** a framework is adopted only when it reproduces your baseline within noise *and* beats it on time-to-result. Anything else is tool tourism. This phase is last for exactly the same reason the repo forbade frameworks in Phase 04

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- **The actual test: re-run Phase 04 from its run-id and land within noise of the original loss curve.** If you cannot, that is the most valuable `LESSONS.md` entry in the repo
- Log what everyone omits: **sample generations every N steps.** It is the single most useful log in fine-tuning
- Publish the golden-thread model properly, with the eval table in the card

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

_(append to [question_answer.md](../../question_answer.md) as `Q17.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
