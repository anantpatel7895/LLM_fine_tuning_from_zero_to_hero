# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **19-phase self-study curriculum** (~170 hrs, ~24 weeks at 5–8 hrs/week) to make one learner — Anant — expert at LLM fine-tuning *and* data preparation, executed on a dedicated 2× NVIDIA A10 box. It is not a library, product, or service.

**The primary artifact is the phase documents, not the code.** A phase is complete when its doc is written, its `question_answer.md` entries can be answered from memory, and its deliverable exists — working code with an empty doc is an incomplete phase. Code exists to make a lesson concrete.

**Consequences for how to work here:**

- **Notebook first, then graduate to `src/`.** Every concept is met in `notebooks/NN_<topic>.ipynb` — run interactively, intermediate values visible, predictions written down before cells are run. Only once it works and is understood does it move into `src/llm_finetuning_from_zero_to_hero/` as tested code. A finished `.py` module hides the exploration that produced it, which is the part being learned. So: **do not open a phase by writing a module.**
- **Do not write code the learner should write.** For a phase not yet started, build scaffolding, tests and tooling that *enable* the exercise; do not hand over the finished exercise. In notebooks this means: give the plumbing (loading a config, fetching a tokenizer, formatting a comparison table) and leave the thinking, predictions and conclusions as cells for him to fill. When unsure, ask which part they want to write themselves.
- **Mentor mode: decide technical details, explain them, don't quiz.** Anant is an experienced engineer (~5 yrs AI/ML) but a genuine beginner at fine-tuning. **What he has actually built** (verified 2026-09-14, do not overstate this): the GPT-2 124M *architecture* from scratch and verified to <0.0001 against HF (`ChatGPT2_124M_Pararameter_training/src/xmer_arch/`, 276 lines — attention, MLP, layer, model, weight loading, tests); LoRA from scratch (`learning_torch/04_learning_lora_from_scratch.ipynb`); multi-head attention from scratch; a production RAG repo; FastAPI-for-ML. **He has never written a training loop** — no optimizer, no `backward()`, no step loop exists in any of his repos. `train_min.py` is his first. Do not describe Phase 01 Part 3 as a "diff against your GPT-2 loop." Choose library versions, hyperparameter starting points, tooling and layout yourself, and give the *why* plus the rejected alternative in the phase doc. Reserve questions for goals, pace, and whether an explanation landed — asking him to pick between tools he hasn't used yet stalls the work.
- **Phases 01–03 need no GPU** and are the current frontier. Phase 00 is 🟡 In Progress; everything from 04 on is ⬜ Not Started.

### Current state of the box (as of 2026-09-11)

Phase 00 is partly measured — see `docs/HARDWARE.md` §0 for live blockers. Established so far:

- ✅ Driver fixed (580.178.04), both A10s visible, **`PHB` topology confirmed — no NVLink**
- ✅ Host is stronger than planned: **Xeon 8358, 60 cores, 472 GiB RAM** (makes ZeRO-3 CPU optimizer offload viable in Phase 11)
- ✅ `Llama-3.1-8B-Instruct` and `Mistral-7B-v0.3` **already downloaded** to `~/slm-models/base_models/` on a second disk — no large download needed for Phases 02–04
- ⚠️ **Root disk `/` is at 95%, 5.0 GB free**, and `FT_VENV` lives there while the install needs 6–8 GB. Unresolved blocker
- ⚠️ `HF_HOME` unset — must be pointed at `/dev/sdb` (`~/slm-models`, 40 GB free) before any download
- ⚠️ The box is **not idle**; Anant's own vLLM server and a second job have held ~28 GiB of the 45 GiB VRAM. They are stoppable
- ❗ Torch is **not yet installed in `FT_VENV`** — the earlier install went to a Python 3.10 user directory (see L1)

## Read before writing any training code

**[docs/notes/api-2026.md](docs/notes/api-2026.md) is mandatory reading before touching `transformers`, `trl`, or `peft`.** `transformers` v5 and `trl` v1 removed API that essentially every existing blog post and most model knowledge still uses. Several changes are **silent** rather than errors:

- `torch_dtype=` → `dtype=`; default dtype is now `"auto"` (checkpoint precision), not fp32
- `load_in_4bit=True` **removed** → `quantization_config=BitsAndBytesConfig(...)`
- `Trainer(tokenizer=)` → `Trainer(processing_class=)`
- `report_to` now defaults to `"none"` — **runs silently stop logging**
- `DataCollatorForCompletionOnlyLM` **removed**. Completion-only loss is now `completion_only_loss=True` (default for prompt-completion data) or `assistant_only_loss=True` (conversational) — and the latter **requires `{% generation %}` markers in the chat template** or it silently masks the wrong thing
- `packing=True` is now safe (`bfd` strategy auto-enables `padding_free`); the old cross-contamination warnings describe `packing_strategy="wrapped"`

That file also carries the A10/sm_86 attention guidance, the FSDP2-vs-DeepSpeed decision, and which data tools are dead. It has a maintenance note at the bottom — update it when you discover a drift.

## Two machines, two environments

The Mac and the A10 box each get their own `FT_VENV`. Same name, deliberately different contents — a venv holds architecture-specific compiled binaries, so it can never be shared or synced. `scripts/sync_to_gpu.sh` excludes `.venv*`, `data/`, `artifacts/` and `checkpoints/` for this reason.

| | Mac (authoring) | A10 box (training) |
|---|---|---|
| Install | `requirements.txt` | `requirements-gpu.txt` (+ `requirements-serving.txt` in a **separate** venv) |
| Used for | docs, configs, tokenizers, chat templates, loss-mask inspection, `pytest`, git | all training, eval, benchmarks, vLLM |

`requirements-serving.txt` must stay in its own environment: vLLM pins an exact older torch and will silently downgrade the training env, breaking a compiled flash-attn.

## Commands

**Always `python -m pip`, never bare `pip`.** A stray `~/.local/bin/pip` for another Python
version shadowed the venv's pip on this box and silently installed 6 GB into the wrong
interpreter, twice. Same applies to every console script — `pytest`, `accelerate`,
`torchrun` are all shebang scripts and can be shadowed the same way; prefer
`python -m pytest`, `python -m torch.distributed.run`. Full mechanism: `docs/LESSONS.md` **L1**.

```bash
# --- environment (either machine; requirements file differs) ---
pyenv virtualenv 3.12.10 FT_VENV && pyenv activate FT_VENV
python -V && python -m pip -V            # MUST agree before installing anything
python -m pip install -r requirements.txt      # Mac
# box: torch from the cu128 index FIRST, then the rest.
# cu128 tops out at torch 2.11.0; 2.12+ is CUDA-13-only and bitsandbytes/flash-attn
# wheels are best-populated for cu12x, so 2.11.0 is the deliberate pin.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# -c is mandatory: without it pip silently swaps the cu128 torch for the CUDA 13 build
# from PyPI, breaking bitsandbytes and all of Phase 04.
python -m pip install -c constraints-gpu.txt -r requirements-gpu.txt

# --- tests (Mac, no GPU) ---
pytest tests/ -v
pytest tests/test_lora_equivalence.py::test_zero_init_is_identity -v   # single test

# --- Mac → box ---
./scripts/sync_to_gpu.sh --dry-run       # always first; confirm no data/ or artifacts/
./scripts/sync_to_gpu.sh

# --- on the box ---
./scripts/00_hardware_report.sh                                  # driver, CUDA, topology
python -m llm_finetuning_from_zero_to_hero.00_gpu_benchmark       # TFLOPS, bandwidth, all-reduce
./scripts/smoke_test.sh configs/sft/<recipe>.yaml                # ~50 steps, sub-1B — MANDATORY
tmux new -s ft && ./scripts/launch.sh configs/sft/<recipe>.yaml   # the real run

# --- Phase 01 minimal loop (runs on Mac CPU/MPS) ---
python -m llm_finetuning_from_zero_to_hero.train.train_min --steps 20 --inspect
```

`launch.sh` refuses to run outside `tmux` (an SSH drop kills the run) and `smoke_test.sh`/`launch.sh` exit with a pointer to `docs/phases/phase-04.md` because the trainer they call is built in Phase 04.

## Architecture

### Documents are the structure

| File | Role |
|---|---|
| [PLAN.md](PLAN.md) | **The index.** Rules of engagement, phase table, model ladder, golden thread. Root `README.md` is only a command cheat-sheet |
| [docs/phases/phase-NN.md](docs/phases/) | One working doc per phase. Fixed 6-part template: *Learning goal → Concept → Numbers → Landscape & choice → Design → Build & Q&A*, with a nav line and ✅/🟡/⬜ status |
| [NUMBERS.md](NUMBERS.md) | §1–§12, every formula with a worked example on 2× A10. Memory per param, activations, the logits trap, FLOPs, LoRA param count, comms volume, KV cache, eval statistical power, MinHash bands, roofline, checkpoint storage |
| [docs/DATA.md](docs/DATA.md) | The co-equal twin of NUMBERS.md. Format taxonomy, chat templates, **the six silent loss-masking failures**, record schema, filter funnel, dataset catalogue, glossary |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Measured baseline of the box. Predictions get replaced by real readings here |
| [docs/notes/api-2026.md](docs/notes/api-2026.md) | Current-API reference (see above) |
| [docs/BRIDGE.md](docs/BRIDGE.md) | Phase 01: what the learner hand-wrote ↔ what the library does ↔ what the difference costs |
| [docs/LESSONS.md](docs/LESSONS.md), [question_answer.md](question_answer.md) | Append-only. Principles with the incident that produced each; recall log as `Q<phase>.<n>` with ⭐/🔧/📐 markers |
| [docs/decisions/](docs/decisions/) | Numbered ADRs. `0001` fixes the stack, hardware constraints and run-id contract |

When adding cross-references, cite the specific section (`NUMBERS.md §6`, `DATA.md Part 3`) — the docs are heavily interlinked and section-level pointers are the convention.

### Hardware constraints are first-class curriculum content

2× A10, 24 GB each (48 GB total), Ampere sm_86, bf16 yes, **FP8 no**, FlashAttention-2 yes / FA3 no, **PCIe with no NVLink**. Dedicated and always on, so GPU time is free and does not consume study hours — phases are designed as "launch Friday night, read Saturday morning."

This yields: full fine-tune ceiling ≈ 1–3B, bf16 LoRA ceiling ≈ 8–14B, QLoRA ceiling ≈ 32B on one card. Hence **`Llama-3.1-8B-Instruct` as the headline model and `Qwen3-1.7B` as the ablation model** — ablate at 1.5B, confirm the top two configs at 8B.

The derivation that most shapes the code: DDP for LoRA all-reduces ~84 MB/step (≈8 ms, negligible), while full fine-tuning an 8B all-reduces 16 GB/step (≈1.6 s, dominant). **On this box the second GPU is often better used as a second experiment slot than a second worker.** See `NUMBERS.md §6`.

### Code layout

`src/llm_finetuning_from_zero_to_hero/` mirrors the doc tree: `data/` (loaders, templates, dedup, synthesis), `train/` (sft, dpo, grpo, full_ft), `eval/`, `serve/`. Most of it is created by the phase that needs it.

**Numeric filename prefixes only on run-only entry points.** `python -m pkg.00_thing` works, but `import pkg.00_thing` is a hard `SyntaxError` — a module name cannot begin with a digit. So `00_gpu_benchmark.py` keeps its prefix (only ever run), but anything importable or unit-tested (`templates.py`, `mask_inspector.py`) must not have one. Ordering for those lives in the phase docs.

`notebooks/` never imports from `src/` — code graduates into the package or dies in the notebook.

### Run identity

`launch.sh` writes `artifacts/<run-id>/run_card.md` **before** training starts, so the record survives a crashed run. Run-id is `{date}-p{phase}-{method}-{model}-{confighash8}`. The card captures git SHA, working-tree cleanliness, library versions and GPU state, plus a **Predictions vs Measured** table to fill in.

**rsync while iterating; `git pull` when the number matters.** `launch.sh` marks a run **DIRTY** if the tree had uncommitted changes — such runs are fine for exploration but their numbers must not be reported, because the recorded SHA doesn't describe what ran.

## Load-bearing rules (full list in [PLAN.md](PLAN.md))

1. **Eval before optimization** — nothing is tuned before there is a metric to move.
2. **Predict, then measure, then explain the gap** — write predicted VRAM/tok/s/step-time into the phase doc *before* launching. The delta is the lesson.
3. **Ablate at 1.5B, confirm at 8B.**
4. **Smoke test first, always** — a recipe that hasn't smoke-tested is a hope, not a recipe.
5. **Inspect the tensors, not the loss curve** — a broken loss mask yields a beautiful loss curve and a model that never stops generating.
6. **One variable per run** — data changes and training changes never ship together.
7. **Understand it before you accelerate it** — raw HuggingFace, hand-written, through Phase 16. Unsloth/Axolotl/LLaMA-Factory arrive only at Phase 17, and only once one reproduces the learner's own baseline within noise.
