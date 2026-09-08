# BRIDGE.md — what I wrote by hand ↔ what the library does ↔ what the difference costs

**Fill this in during [Phase 01](phases/phase-01.md), in your own words.** Do not copy the tables from the phase doc — the value is in the writing, not the table. Where you disagree with what the phase doc claims, write your version and note the disagreement; you may well be right.

**Why this file exists.** You have already built a GPT-2 124M training loop, LoRA from scratch, and multi-head attention from scratch. That is an unusual asset, and the fastest route to expert is to *diff* your code against the production library rather than start over from a tutorial. This file is that diff.

**Source locations** (read the source, not the docs — you already know what the code is trying to do):

```bash
python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"
python -c "import peft, os; print(os.path.dirname(peft.__file__))"
```

| Component | File to read |
|---|---|
| Llama architecture | `transformers/models/llama/modeling_llama.py` |
| LoRA layer | `peft/tuners/lora/layer.py` |
| Training step | `transformers/trainer.py` — `def training_step`, `def compute_loss` |
| SFT specifics | `trl/trainer/sft_trainer.py`, `trl/trainer/sft_config.py` |

---

## 1. Architecture — my GPT-2 vs Llama-3

| My GPT-2 | Llama-3 | What the difference costs / buys |
|---|---|---|
| | | |

Things to make sure you cover: attention head layout (MHA vs GQA), positional encoding (learned table vs RoPE), normalization (LayerNorm vs RMSNorm), the MLP (GELU 2-matrix vs SwiGLU 3-matrix), biases, and embedding tying.

**My notes:**

_(write here)_

---

## 2. LoRA — my `LinearWithLoRA` vs `peft`'s `Linear`

| `peft` feature | What failure made this configurable? | Do I need it? |
|---|---|---|
| | | |

Cover at minimum: `lora_B` zero-init, `use_rslora`, `init_lora_weights` variants, `modules_to_save`, `merge_and_unload`, `disable_adapter()`, `target_modules`.

**My notes:**

_(write here)_

---

## 3. The training loop — mine vs `Trainer`

| My loop has | `Trainer` adds | Why it matters / which failure it prevents |
|---|---|---|
| | | |

Cover: gradient accumulation and loss normalization, device placement, gradient clipping, LR scheduling, checkpoint/resume completeness, dataloader workers and length grouping, gradient checkpointing.

**My notes:**

_(write here)_

---

## 4. Verified numbers

Filled in by reading each model's `config.json` — see [phase-01.md §2](phases/phase-01.md).

| Check | Predicted | Actual | Gap |
|---|---|---|---|
| Llama-3.1-8B `embed + lm_head` params | 1.05 B | | |
| Llama-3.1-8B trunk params | 6.98 B | | |
| Mistral-7B-v0.3 trunk params | 6.98 B — **identical** | | |
| LoRA r=16 all-linear trainable params | 41.9 M (0.52%) | | |

---

## 5. The three things `Trainer` does that my loop did not

State them from memory, without looking at the source. If you cannot, Phase 01 is not finished.

1. _(…)_
2. _(…)_
3. _(…)_

---

## 6. What `train_min.py` deliberately omits

`src/llm_finetuning_from_zero_to_hero/train/train_min.py` prints this list when it finishes. Copy the list here once you have run it, because **that list is the argument for using the library** — not that TRL is conceptually different from your loop, but that it does about fifteen small things correctly that are each easy to get subtly wrong.

_(paste here after running `python -m llm_finetuning_from_zero_to_hero.train.train_min --steps 20`)_
