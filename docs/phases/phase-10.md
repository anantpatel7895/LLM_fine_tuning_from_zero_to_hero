# Phase 10 — Multi-GPU on 2×A10 without NVLink

**~10 hrs** · Prerequisite: [Phase 09](phase-09.md) · Next: [Phase 11](phase-11.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **derive each parallelism strategy's memory and communication cost, predict which wins on *this* box, then measure and explain your error**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** `configs/accelerate/{single,ddp,zero2,zero3,zero3-offload,fsdp}.yaml`; a predicted-vs-measured bake-off table; `NUMBERS.md` §6 confirmed against real all-reduce timings

**Models:** Llama-3.2-1B (full FT, to make sharding matter) and Llama-3.1-8B (LoRA)

**Data:** the Phase 09 winning mixture, unchanged

---

## 1. Concept

- Data vs tensor vs pipeline parallelism, and why only data parallelism is relevant on two cards
- **Per-GPU memory by strategy** — the table in `NUMBERS.md` §6. DDP 16 B/param, ZeRO-1 10, ZeRO-2 9, ZeRO-3 8 at G=2. Note the vocabulary mapping: FSDP `SHARD_GRAD_OP` ≡ ZeRO-2, `FULL_SHARD` ≡ ZeRO-3
- **The communication derivation, which is this phase's whole point.** DDP moves ≈2Ψ bytes per step. For LoRA r=16 on 8B that is 168 MB ≈ 8 ms over PCIe — negligible. For a full 8B it is 32 GB ≈ 1.6 s of *pure communication* per step. **Therefore ZeRO-3 on a PCIe pair is often slower than not sharding, and frequently slower than running two independent experiments.** Treat GPU 1 as a second experiment slot
- **Throughput engineering, where the tokens/sec actually come from:** sequence packing (naive concat vs first-fit-decreasing bin packing); the cross-contamination problem, and how `position_ids` + variable-length attention solve it — which `packing_strategy="bfd"` now does by default, though the mechanism is still the interview question; padding-free batching; `group_by_length` and how it hides OOMs until step 4000; **fused cross-entropy** (which removes the `b × s × V` logits tensor from `NUMBERS.md` §3 — often the single biggest memory win); `torch.compile` on Ampere; gradient checkpointing interacting with FSDP wrapping
- All three launchers, so nobody can hand you an unfamiliar repo: `torchrun`, `accelerate launch`, and a `deepspeed` JSON

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

- DeepSpeed vs PyTorch FSDP. **Both, once each**, because you will meet both in real repos — then an ADR saying which you reach for and when

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- **Write the predicted winner and the predicted tok/s for each strategy before running anything.** This is the phase where the prediction discipline pays off most
- Bake-off table: strategy × tokens/sec × peak VRAM per GPU × communication fraction of step time
- Confirm the PCIe all-reduce bandwidth from Phase 00 against what you observe in a real DDP run
- The honest conclusion in the ADR: when DDP, when ZeRO-2, when ZeRO-3 + offload, and when to just run two experiments

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

_(append to [question_answer.md](../../question_answer.md) as `Q10.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
