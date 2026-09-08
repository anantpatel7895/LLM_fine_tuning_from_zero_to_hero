# ADR 0001 — Stack, hardware constraints, and the run-id contract

**Date:** 2026-09-07
**Status:** Accepted

## Context

Starting a 19-phase curriculum to become expert at LLM fine-tuning and data preparation, on a dedicated 2× NVIDIA A10 box (24 GB each, Ampere SM 8.6, PCIe with no NVLink). The learner is an experienced engineer but a beginner at fine-tuning specifically, and has ~5–8 hrs/week.

Several decisions constrain everything downstream and are worth recording before any code exists, so that later "why is it like this?" has an answer.

## Decisions

### 1. Raw HuggingFace stack first; frameworks at Phase 17

Phases 00–16 use hand-written `transformers` + `peft` + `trl` + `bitsandbytes` + `accelerate`. Unsloth, Axolotl and LLaMA-Factory are deliberately withheld until Phase 17.

**Why:** starting with a config-driven framework teaches its YAML schema, not fine-tuning. You cannot see what an abstraction hides until you have built the thing it hides. The cost is real — Phases 04–11 will be slower and more error-prone than they need to be — and that friction is the curriculum.

**Adoption rule for Phase 17:** a framework is adopted only when it reproduces *our own* baseline within noise **and** beats it on time-to-result. Anything else is tool tourism.

**Rejected:** starting with Unsloth for fast wins. It would have produced a working model in week one and a shallow understanding by week twenty.

### 2. Llama-3.1-8B-Instruct as the headline model, a 1.5–1.7B as the ablation model

**Why:** from [NUMBERS.md §1](../NUMBERS.md), a bf16 LoRA on an 8B needs ~17 GB and fits one card; a full fine-tune of an 8B needs ~128 GB and does not fit at all. Meanwhile a 1.7B fully fine-tunes across both cards comfortably.

So: **ablate at 1.5B, confirm at 8B.** Grids run on the small model where a run costs 45 minutes; the top two configurations get confirmed once on the 8B. Without this rule the ablation phases do not fit in 5–8 hrs/week.

**Rejected:** doing everything at 8B (grids become unaffordable) and doing everything at 1.5B (results would not transfer, and the portfolio artifact would be uninteresting).

### 3. Tool-calling / structured output as the golden thread

One task carried through every phase, rather than a different toy task per technique.

**Why:** it is **programmatically verifiable** — is the JSON valid, is it the right tool, are the arguments correct. That single property means the same task supports SFT, automatic preference-pair construction, *and* GRPO with a computable reward, with no human labeling and no judge model in the critical path. It also fails visibly, which makes debugging instructive.

**Rejected:** a general chat-quality thread (needs a judge for every measurement, and "is this response nicer" teaches less when it breaks) and separate per-phase tasks (portfolio reads as many small demos instead of one deep story).

### 4. Hardware constraints treated as first-class curriculum content

The absence of NVLink is not an inconvenience to work around; it is the source of the most transferable lesson in the repo. From [NUMBERS.md §6](../NUMBERS.md): DDP for a LoRA adapter moves 168 MB per step (~8 ms, negligible), while a full fine-tune of an 8B moves 32 GB per step (~1.6 s of pure communication). Therefore ZeRO-3 on this box is frequently *slower* than not sharding — and often slower than running two independent experiments, one per card.

**Consequence written into the plan:** GPU 1 is treated as a second *experiment slot*, not a second *worker*, unless the arithmetic says otherwise. Phase 10 measures this rather than assuming it.

### 5. pyenv + split requirements files

`requirements.txt` for the Mac authoring environment; `requirements-gpu.txt`, fully pinned, for the box.

**Why:** `torch` with a CUDA build, `bitsandbytes`, `flash-attn`, `deepspeed` and `vllm` have no usable Mac wheels. A single requirements file would make the Mac environment uninstallable, which would block the "author locally, train remotely" workflow entirely.

**Rejected:** uv + `pyproject.toml` (used in the newer sibling repos, and better in most respects) — chosen against because most fine-tuning tooling, Docker images and community recipes assume plain pip, and fighting that on a remote GPU box is friction that teaches nothing. Revisit if pip resolution becomes painful.

### 6. TensorBoard first, Weights & Biases from Phase 06

**Why:** TensorBoard needs no account and no API key, so Phases 00–05 start with zero setup cost. W&B arrives exactly when it earns its place — Phase 06 runs an eight-configuration ablation grid, and comparing eight runs is what W&B is actually good at. Its shareable run links then double as portfolio evidence.

**Rejected:** W&B from the start (setup friction before there is anything to compare) and TensorBoard throughout (genuinely poor at ranking many runs, and produces nothing shareable).

### 7. Run-id contract, fixed now because it cannot be retrofitted

```
{date}-{phase}-{method}-{model}-{confighash8}
20260907-p04-qlora-llama31-8b-a3f91c2e
```

Everything about a run lives in `artifacts/<run-id>/`: the resolved config, `run_card.md`, `train.log`, the adapter, and the eval JSON.

`run_card.md` records, from the very first run in Phase 04: git SHA + dirty flag, config hash, dataset version hash, resolved hyperparameters, `nvidia-smi` snapshot, predicted-vs-measured VRAM and throughput, and the resulting metric.

**Why now:** by Phase 15 there will be dozens of runs, and "which config produced the model I pushed?" is unanswerable in retrospect if the record was not written at the time. This is the one piece of Phase 17's discipline that has to start at Phase 04.

## Consequences

- Phases 04–11 will be more painful than they would be with Axolotl. Accepted.
- Full fine-tuning of an 8B is out of scope for this hardware, so Phase 11 uses a 1–3B. Accepted and honest.
- FP8 and FlashAttention-3 are unavailable on SM 8.6, so any recipe assuming them is out. Recorded in [HARDWARE.md](../HARDWARE.md).
- Long-context extension and multimodal fine-tuning are excluded, as a decision rather than a gap. See the end of [PLAN.md](../../PLAN.md).
