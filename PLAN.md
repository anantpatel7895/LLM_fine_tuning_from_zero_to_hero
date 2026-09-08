# LLM Fine-Tuning: Zero → Expert

**Audience:** ~5 yrs AI/ML/GenAI engineering. No Python basics, no "what is an embedding", no transformer-math primer — you already built a GPT-2 124M training loop and LoRA from scratch. What you have *not* done is take a real open-weight model and make it measurably better on a task you care about.

**Premise:** the deliverable of each phase is *understanding*, verified by a working system and a number that moved.

**Two co-equal goals**, not one:

1. **Fine-tuning** — SFT, LoRA/QLoRA, full fine-tuning, preference optimization, RL with verifiable rewards, multi-GPU, quantization, serving.
2. **Data preparation** — sourcing, cleaning, dedup, decontamination, synthetic generation, curation, mixing. Three full phases, because this is where fine-tuning outcomes are actually decided and it is the half most people skip.

**Duration:** 19 phases, ~170 hrs. ~24 weeks at 5–8 hrs/week. Three phases are marked *compressible* if you want to land closer to 20.

**Hardware:** 2× NVIDIA A10 (24 GB each, 48 GB total), Ampere SM 8.6, PCIe without NVLink. Dedicated and always on. Every "what fits" claim in this repo is measured on that box, not copied from a blog post.

---

## Read this before the phase list

### Attention-hours ≠ wall-clock hours

Your box is dedicated and always on, so **GPU time is free and does not consume your 5–8 hrs/week.** Every phase here is designed as *"40 minutes to launch, sleep, 90 minutes to read the results."* A phase costing 8 attention-hours can contain 30 hours of GPU time.

Launch long runs into `tmux` on a Friday night and read them Saturday morning. This is the single biggest structural advantage you have over someone renting H100s by the hour — it is what makes the ablation-grid phases affordable at all, and the plan exploits it deliberately.

### The one table that governs every model choice

Derive this once in Phase 00 and it constrains the whole curriculum:

| Training regime | B/trainable param | 1.7B | 3B | 8B |
|---|---:|---:|---:|---:|
| Full fine-tune, AMP bf16 + fp32 master + AdamW | 18 | 36 GB | 55 GB | **135 GB** |
| Full fine-tune, pure-bf16 + fp32 AdamW (FSDP2 can do this) | 12 | 24 GB | 37 GB | 90 GB |
| Full fine-tune, pure-bf16 + 8-bit AdamW | 6 | 12 GB | 18 GB | 45 GB |
| LoRA in bf16 (frozen bf16 base) | — | 3.4 GB | 6.2 GB | **~20 GB** |
| QLoRA, NF4 + double quantization | — | ~1.4 GB | ~2.2 GB | **5.3 GB** |

Sharded across 2 GPUs, and remembering that **48 GB is not a pool — it is 24 + 24, and each rank needs its own share**, the ceilings are:

| | Ceiling on this box |
|---|---|
| Full fine-tune, plain FSDP2, no tricks | **3B class** |
| Full fine-tune of an 8B | only via DeepSpeed ZeRO-3 + CPU optimizer offload, ~4× slower |
| **bf16 LoRA** | **4B class** for real headroom |
| **QLoRA** | **8B comfortably** |

> ### The framing to remember
>
> **On 24 GB cards, bf16 LoRA is a 4B technique and QLoRA is the 8B technique.**
>
> bf16 LoRA on an 8B technically fits — ~20 GB with gradient checkpointing at sequence 2048 — but with ~2 GB headroom at micro-batch 1. One long example OOMs it. Full derivations in [NUMBERS.md](NUMBERS.md).

### The model ladder

| Tier | Model | Used for |
|---|---|---|
| **Smoke** | `Qwen3-0.6B` | Phases 00, 12 — template checks, induced failures, 60-second feedback loops |
| **Ablation** | `Qwen3-1.7B` | Every grid and sweep. Full fine-tunes on **one** card with an 8-bit optimizer (13.2 GB) |
| **Full-FT ceiling** | **`SmolLM3-3B`** | Phase 11. Apache-2.0, and its **entire training recipe is published** — so you can compare your choices against a fully open reference |
| **Headline** | **`Llama-3.1-8B-Instruct`** | The golden thread. **QLoRA**, ~10.5 GB at sequence 2048 |
| **Contrast** | `Mistral-7B-v0.3` | Phase 02 template drills — and see the identical-trunk comparison in [NUMBERS.md](NUMBERS.md), which makes it the *roomiest* 7B config here |
| **bf16 LoRA tier** | `Qwen3-4B` | Where bf16 LoRA actually has headroom |
| **Capacity stretch** | `Qwen3-14B` (QLoRA) | Proves the QLoRA-is-a-capacity-tool-not-a-speed-tool argument from Phase 06 |

⚠️ **Avoid `Qwen3.5-*` and `gemma-4-*`** for this curriculum — they are multimodal `ForConditionalGeneration` architectures and will not behave like plain text models. Details in [docs/notes/api-2026.md](docs/notes/api-2026.md).

**Anti-goal, stated plainly:** no Unsloth, no Axolotl, no LLaMA-Factory until Phase 17. Phases 00–16 are hand-written `transformers` + `peft` + `trl` + `bitsandbytes` + `accelerate`, on purpose. A framework gets adopted only once it reproduces *your own* baseline within noise.

---

## How to use this

`PLAN.md` is the map. Each phase has its own working document under [docs/phases/](docs/phases/).

| Document | Purpose |
|---|---|
| [PLAN.md](PLAN.md) | Rules, method, stack, phase index, the step template |
| [docs/phases/phase-NN.md](docs/phases/) | The working document for one phase |
| [NUMBERS.md](NUMBERS.md) | Every formula in the project, with worked examples — VRAM, throughput, LoRA parameter counts, token budgets |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Measured baseline of the 2×A10 box, and what actually fits on it |
| [docs/notes/api-2026.md](docs/notes/api-2026.md) | **Read this before copying code from anywhere.** What `transformers` v5 and `trl` v1 removed, and which widely-recommended tools are now abandoned |
| [docs/DATA.md](docs/DATA.md) | Cross-cutting data reference: schemas, chat templates, masking rules, every filter and what it cost, the glossary. **The co-equal twin of `NUMBERS.md` — if this file ends up shorter, the data-prep half of the goal was not met** |
| [docs/LESSONS.md](docs/LESSONS.md) | Transferable principles, each with the incident that produced it |
| [question_answer.md](question_answer.md) | Running Q&A log — recall practice, the real retention mechanism |
| [docs/decisions/](docs/decisions/) | Numbered ADRs: why this model, this framework, this eval suite |
| [docs/setup/remote-gpu.md](docs/setup/remote-gpu.md) | The Mac → A10 box workflow: SSH, tmux, sync, environment install |

---

## Rules of engagement

1. **Eval before optimization.** No phase tunes a hyperparameter before it has a metric to move. Phase 05 exists so that Phase 06's ablation grid is rankable rather than eight runs and a shrug.
2. **Predict, then measure, then explain the gap.** Before launching any run, write your predicted peak VRAM, tokens/sec and step time into the phase doc. **The delta between your prediction and the measurement is the actual lesson** — log it in `NUMBERS.md`. In fine-tuning, the prediction *is* the skill; "how much VRAM does an 8B full fine-tune need?" is a real interview question with a real answer.
3. **Ablate at 1.5B, confirm at 8B.** A hypothesis you cannot test in 45 minutes is a hypothesis you will not test. Grids run on the ablation-tier model; the top two configurations get confirmed once on the headline model.
4. **Smoke test first, always.** Every recipe runs ~50 steps on a sub-1B model before it runs for real (`scripts/smoke_test.sh`). A recipe that has not smoke-tested is not a recipe, it is a hope.
5. **Inspect the tensors, not the loss curve.** A broken loss mask produces a beautiful, believable loss curve and a model that never shuts up. When something is wrong, print token / id / label / in-loss side by side. Phase 03 builds the tool; every later phase uses it.
6. **One variable per run.** Data changes and training changes never ship in the same run, or you learn nothing from either.
7. **Every dataset is versioned and has a datasheet.** `data/processed/<name>/v<n>/` with row count, token histogram, content hash, license and provenance. An unversioned dataset makes every eval number that used it unattributable.
8. **Decontaminate before you evaluate, not after.** A contaminated benchmark score is worse than no score, because you will believe it.
9. **One golden thread.** Every phase improves the *same* model on the *same* task, so improvements compound and the lineage is legible. Ablations are side quests and are allowed to be thrown away.
10. **Every run is reproducible.** Committed config + fixed seed + a `run_card.md` in `artifacts/<run-id>/` recording the exact commit, dataset hash and resulting metric. A run you cannot re-create taught you nothing.
11. **Understand it before you accelerate it.** Raw HuggingFace, hand-written, first. Unsloth / Axolotl / LLaMA-Factory arrive at Phase 17, once you know what their YAML is hiding.
12. **Break it on purpose.** Every phase ends adversarially: what does this now fail on? The failure goes into the eval suite; the repro goes into `docs/LESSONS.md`.
13. **Write the phase doc as you go, not after.** The doc is where learning is consolidated; a phase with working code and an empty doc is not complete.
14. **Prose lives in `docs/`.** Code carries short comments, not essays.

---

## The golden thread

> **A tool-calling / structured-output assistant** — given a user request and a set of JSON tool schemas, emit a valid, correctly-argued tool call.

Why this task:

- **It is programmatically verifiable.** Is the JSON valid? Is it the right tool? Are the arguments right? That means the *same* task supports supervised fine-tuning, automatic preference-pair construction, *and* GRPO with a real computable reward — no human labeling, no judge model in the critical path.
- **It fails visibly.** Structured-output failures are obvious, unlike "is this chat response nicer", so debugging teaches you something.
- **It is what the market buys.** Tool calling and structured extraction are the two fine-tuning use cases most companies actually pay for.

| Role | Model | Why |
|---|---|---|
| Golden-thread main model | `Llama-3.1-8B-Instruct` | The reference point everyone knows; QLoRA fits on one A10 |
| Full fine-tuning + GRPO | `Qwen3-1.7B` | Small enough to fully fine-tune on 48 GB and to do RL rollouts at a tolerable speed |
| Template/tokenizer contrast drills | `Mistral-7B-v0.3`, `gemma-3` small | Different chat templates and tokenizers — the point is to feel the differences, not to train them |
| Smoke tests | `Llama-3.2-1B` / `Qwen3-0.6B` | Fast enough that a broken recipe fails in 60 seconds |

---

## The step format

Every phase document is built from this template. It is the same one used in the RAG repo, because it works.

| Part | Question it answers |
|---|---|
| **0. Learning goal** | What can you *do* after this that you couldn't before? |
| **1. Concept** | What is the idea, in plain language, and what problem does it solve? |
| **2. Numbers** | What is the arithmetic? Memory, parameter counts, throughput, token budgets — with a worked example |
| **3. Landscape & choice** | What are the options, and *why this one*? The rejected alternative gets named |
| **4. Design** | What are you about to build, and how will you know it worked? |
| **5. Build & Q&A** | The code, the run, the result, and the questions that go into `question_answer.md` |

---

## Phase index

Legend: ✅ Done · 🟡 In Progress · ⬜ Not Started

| # | Phase | Deliverable | Hrs | Status |
|---|---|---|---|:---:|
| [00](docs/phases/phase-00.md) | Environment & hardware baseline | `docs/HARDWARE.md` with measured numbers; sync + smoke-test scripts | 6 | ⬜ |
| [01](docs/phases/phase-01.md) | Bridge: your GPT-2 loop → the HF stack | Your loop mapped onto `Trainer`; your LoRA mapped onto `peft` source | 6 | ⬜ |
| [02](docs/phases/phase-02.md) | Tokenizers, chat templates & special tokens | Template renderer + round-trip tests across 4 model families | 8 | ⬜ |
| [03](docs/phases/phase-03.md) | **Loss masking & the anatomy of an example** | `mask_inspector.py`; tests catching off-by-one on assistant headers | 8 | ⬜ |
| [04](docs/phases/phase-04.md) | First real QLoRA SFT run | First adapter + `run_card.md` + before/after generations | 8 | ⬜ |
| [05](docs/phases/phase-05.md) | **Evaluation before optimization** | ~100-prompt regression suite, harness wiring, judge, baseline report | 10 | ⬜ |
| [06](docs/phases/phase-06.md) | LoRA/QLoRA mechanics & ablations | ≥8-run ablation grid + parameter-count math | 10 | ⬜ |
| [07](docs/phases/phase-07.md) | **Data prep I** — sourcing, quality, dedup, decontamination | 3 raw sources → one clean dataset + dataset card + stats | 10 | ⬜ |
| [08](docs/phases/phase-08.md) | **Data prep II** — synthetic generation & curation | 2–5k verified synthetic examples + rejected-example analysis | 10 | ⬜ |
| [09](docs/phases/phase-09.md) | **Data prep III** — mixing, ordering, how much is enough | 3-mixture ablation + quality-vs-quantity result + ADR | 8 | ⬜ |
| [10](docs/phases/phase-10.md) | Multi-GPU on 2×A10 without NVLink | Throughput table: DDP / ZeRO-2 / ZeRO-3 / offload / FSDP, with MFU | 10 | ⬜ |
| [11](docs/phases/phase-11.md) | Full fine-tuning & continued pre-training | Full FT of a 1–3B + catastrophic-forgetting eval + replay mix | 10 | ⬜ |
| [12](docs/phases/phase-12.md) | Debugging & training pathologies | ≥10 entries in `docs/LESSONS.md`, each from a bug you reproduced | 8 | ⬜ |
| [13](docs/phases/phase-13.md) | Preference optimization: DPO & family | DPO'd model + win-rate table; a preference dataset you built | 10 | ⬜ |
| [14](docs/phases/phase-14.md) | Reward modeling | Trained reward model + accuracy/calibration + length-bias analysis | 8 | ⬜ |
| [15](docs/phases/phase-15.md) | GRPO & verifiable rewards | GRPO run moving a verifiable metric + reward-hacking postmortem | 12 | ⬜ |
| [16](docs/phases/phase-16.md) | Merging, quantization & serving | Quantized model + vLLM endpoint + latency/quality table | 10 | ⬜ |
| [17](docs/phases/phase-17.md) | Accelerator frameworks & reproducibility | Phase 04 reproduced three ways; published HF Hub model card | 8 | ⬜ |
| [18](docs/phases/phase-18.md) | Capstones & portfolio | Three portfolio artifacts (below) | 12 | ⬜ |

---

## Why this order

The sequence is designed so that nothing is tuned before it can be measured, and so the two phases beginners most often skip arrive *before* the fun part.

- **Phase 03 (loss masking) comes before the first real training run.** This is the single biggest silent failure in beginner fine-tuning: get the mask wrong and you still see a smooth descending loss curve, and you still get a model that generates forever and never emits a stop token. You cannot debug it from the loss. You debug it by printing the mask — so you learn to print the mask first.
- **Phase 05 (eval) comes before Phase 06 (ablations).** Reversed, the ablation grid produces eight checkpoints you cannot rank.
- **Phases 07–09 (data prep) sit in the middle, not at the front.** Data work needs a feedback signal: one working training run (04) and one metric (05). Doing data prep first is how people end up with a beautiful dataset and no way to tell whether it helped.
- **Phase 12 (debugging) lands after full fine-tuning**, when you will have actually met loss spikes, NaNs and OOMs. It is a consolidation phase, not a reference chapter to read cold.
- **Phase 17 (frameworks) is deliberately last.** Starting with Axolotl or Unsloth teaches you their YAML schema, not fine-tuning. You earn the shortcuts by first writing what they generate.

---

## Capstones (Phase 18)

| | Artifact | What it proves |
|---|---|---|
| **A** | **The golden-thread model**, published to HF Hub with a real model card and a lineage diagram: data pipeline → SFT → DPO → GRPO → eval → quantize → vLLM | You can take a model end-to-end and show the number that moved at each step |
| **B** | **A released dataset** with generation code, dedup/decontamination stats, a dataset card, and an ablation proving it beats a naive baseline | The data-prep half of the goal — the rarer and more valuable half |
| **C** | **A cookbook README** a hiring manager can read in five minutes, linking every phase artifact | That you can communicate it |

---

## Stack

Raw HuggingFace first, by design (Rule 7).

Versions are pinned in `requirements-gpu.txt` and verified as of September 2026.

| Layer | Choice | Introduced |
|---|---|---|
| Core | `transformers` 5.16, `datasets` 5.0, `accelerate` 1.14 | Phase 01 |
| PEFT | `peft` 0.20 (LoRA, QLoRA, DoRA) | Phase 04 |
| Training loops | `trl` 1.12 (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`) | Phase 04 |
| Quantized training | `bitsandbytes` 0.50 (NF4, 8-bit optimizers) | Phase 04 |
| Attention | `sdpa` by default; FA2 via the `kernels` Hub path — **no compiling** | Phase 04 |
| Tracking | TensorBoard → Weights & Biases | 00 → 06 |
| Multi-GPU | **FSDP2** (in torch) for full FT; plain **DDP** for LoRA; DeepSpeed only for the 8B offload case | Phase 10 |
| Eval | `lm-eval` 0.4.13 (vLLM backend) + `inspect-ai` for judge/custom evals | Phase 05 |
| Data | `datatrove` (dedup/filter/decontaminate), `datasketch`, `semhash` | Phase 07 |
| Synthetic data | **roll your own on a local vLLM server** — ~100 lines | Phase 08 |
| Serving | vLLM 0.28 + `llmcompressor` for AWQ/GPTQ — **in a separate venv** | Phase 16 |
| Accelerators | Unsloth (single-GPU only), Axolotl, LlamaFactory | Phase 17 |

**Three stack facts worth knowing before you start**, all detailed in [docs/notes/api-2026.md](docs/notes/api-2026.md):

1. **`transformers` v5 and `trl` v1 broke the old API.** Nearly every tutorial online is written against removed API — `load_in_4bit=True`, `tokenizer=`, `DataCollatorForCompletionOnlyLM`. Read that file before copying any code.
2. **`distilabel` and Argilla are unmaintained** (no real code commits since early 2025; the authors publicly moved on). Almost all 2024–2025 synthetic-data articles recommend them. This repo does not.
3. **vLLM pins `torch==2.13.0` while training runs 2.14.0.** Installing vLLM into the training environment silently downgrades torch and breaks flash-attn. Serving gets its own virtualenv.

---

## Cadence, compression, and what is deliberately left out

**Recommended cadence:** one weekday evening for reading and derivation (template parts 0–4), one weekend morning for building and reading results (part 5), long runs launched into `tmux` on Friday night.

**If you need to land closer to 20 weeks**, cut in this order:

1. Fold Phase 17 (frameworks) into a 2-hr appendix of Phase 04. *Cost: low* — you keep the hand-written skill, lose the bake-off table.
2. Compress Phase 06 (LoRA ablations) to four runs instead of eight. *Cost: low.*
3. Compress Phase 09 (data mixing) to two mixtures instead of three. *Cost: medium* — mixing is where a lot of real gains hide.

**Never cut Phases 03, 05, 07, or 12.** Those four are the difference between this and a tutorial run-through, and they map exactly onto the skip-list above.

**Deliberately out of scope** (named so the boundary is a decision, not a gap):

- **Long-context extension** (RoPE scaling, YaRN, position interpolation) — does not fit 24 GB comfortably, and is additive rather than foundational.
- **Multimodal / VLM fine-tuning** — same reason.
- **PPO, RLOO, Online-DPO** — Phase 15 does GRPO because it drops the value network and therefore fits your hardware. The others are noted and not run, and the phase doc says why.

Both belong to a "Phase 19+, after the capstones" appendix.
