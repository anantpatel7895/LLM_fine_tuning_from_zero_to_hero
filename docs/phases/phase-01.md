# Phase 01 — Bridge: your GPT-2 loop → the HuggingFace stack

**~6 hrs** · Prerequisite: [Phase 00](phase-00.md) · Next: [Phase 02](phase-02.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **point at any component of the HuggingFace training stack and say what you already wrote by hand, what the library adds, and what that difference costs.**

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** [docs/BRIDGE.md](../BRIDGE.md) — a three-column table (*what I wrote* · *what the library does* · *what the difference costs*), plus `tests/test_lora_equivalence.py` proving your hand-written LoRA forward pass matches `peft`'s to float tolerance.

**Models:** none trained. This is a reading-and-diffing phase, and it runs on the Mac.

**Data:** none

---

## 1. Concept

### Why this phase exists, and why it is short

You have an unusual asset. Most people learning to fine-tune have never written a training loop, an attention head, or a LoRA layer. You have written all three:

| What you built | Where |
|---|---|
| A GPT-2 124M training loop | `ChatGPT2_124M_Pararameter_training/` |
| LoRA from scratch | `learning_torch/04_learning_lora_from_scratch.ipynb` |
| Multi-head attention from scratch | `learning_torch/03_understanding_multihead_attention.ipynb` |
| Embedding layers, layer norm, decoder blocks | `learning_torch/00_*.ipynb`, `01_decoder_transformer.ipynb` |

The fastest route from here to expert is **not** to start over with a tutorial. It is to **diff your code against the library** — because then every library feature arrives as an answer to a question you already know how to ask.

This phase is deliberately only 6 hours. Its whole job is to convert what you already know into leverage for Phases 02–18.

### The reading posture

There is one idea to carry through the whole phase, and it is the seed entry in [LESSONS.md](../LESSONS.md):

> **The library's extra parameter is usually somebody else's bug report.**

Every option in `LoraConfig` and `SFTConfig` exists because a default broke for someone. When you meet an unfamiliar parameter, don't skip it — ask *what failure made this configurable?* That question turns API surface into other people's accumulated field experience, and it is the single fastest way to absorb what they learned the hard way.

### Diff 1 — Architecture: your GPT-2 vs Llama

Read `LlamaAttention` and `LlamaForCausalLM` in `transformers` alongside your own `xmer_arch/`. The third column is the point.

| Your GPT-2 | Llama-3 | What the difference buys |
|---|---|---|
| Multi-head attention: `n_heads` Q, K and V | **Grouped-Query Attention**: 32 Q heads, **8 KV heads** | **4× smaller KV cache.** This is the single biggest serving-cost lever, and you already computed it in [NUMBERS.md §8](../../NUMBERS.md) — 128 KiB/token instead of 512 KiB/token |
| Learned positional embeddings (a lookup table) | **RoPE** — rotates Q and K by position-dependent angles | Position becomes *relative* and *extrapolable*, so context extension is possible at all. A learned table hard-caps you at its trained length |
| LayerNorm (subtract mean, divide by std) | **RMSNorm** (divide by RMS only — no mean subtraction, no bias) | Cheaper, one less reduction, and empirically no worse |
| GELU MLP: 2 matrices (`up`, `down`) | **SwiGLU**: 3 matrices (`gate`, `up`, `down`) | More expressive per parameter. Also why **67% of LoRA-targetable parameters live in the MLP** — see [NUMBERS.md §5](../../NUMBERS.md) — which is the arithmetic behind "target all linear layers, not just attention" |
| Biases on linear layers | **No biases** | Fewer parameters, no measured loss |
| Tied input/output embeddings (GPT-2 ties them) | **Untied** on the 8B | Costs memory — and it is *exactly* the `embed + lm_head` term that makes QLoRA on Llama-3.1-8B cost 5.31 GiB rather than 4.0 GiB |

Notice that four of those six rows are things you have *already computed the consequences of* in `NUMBERS.md`. That is what this phase is for.

### Diff 2 — LoRA: your `LinearWithLoRA` vs `peft`

Open `04_learning_lora_from_scratch.ipynb` next to `peft/tuners/lora/layer.py`.

**What you already have:** the `A` and `B` matrices, the `alpha/r` scaling, and merging `B @ A` back into the base weight. That is genuinely the core of it — `peft` does the same arithmetic.

**What `peft` adds, and the failure each one answers:**

| `peft` feature | The question it answers |
|---|---|
| `lora_B` initialized to **zero** | *"How do I guarantee the adapter is a no-op at step 0?"* Because `B = 0`, `B @ A = 0`, so the model at step 0 is **exactly** the base model. **This gives you a test you can actually write:** step-0 loss must equal base-model loss |
| `use_rslora` (scales by `α/√r` instead of `α/r`) | *"Why does raising the rank stop helping past ~64?"* The `α/r` scaling shrinks the update too aggressively at high rank |
| `init_lora_weights` ∈ `gaussian` / `pissa` / `olora` / `loftq` | *"Can I start `A` somewhere smarter than random?"* `loftq` in particular compensates for quantization error, which matters for QLoRA |
| `modules_to_save=["embed_tokens", "lm_head"]` | *"I added a special token, why can the model read it but never emit it?"* Because the output projection was frozen. Phase 02's hardest lesson |
| `merge_and_unload()` | *"How do I ship this as one model?"* — plus the subtlety that merging a **QLoRA** adapter into a **bf16** base leaves residual error, because it was trained against quantized weights |
| `disable_adapter()` context manager | *"How do I get the reference model's output without loading a second copy?"* This is precisely how `DPOTrainer` avoids materializing a second 8B in Phase 13 — the biggest memory win available there |
| `target_modules` | *"Which layers should I even adapt?"* Phase 06 ablates this |

That `disable_adapter()` row is worth pausing on. It is a small API detail that turns out to be the reason DPO fits on your hardware at all.

### Diff 3 — The training loop: yours vs `Trainer`

Read `Trainer.training_step` and `Trainer.compute_loss` next to your GPT-2 loop.

| Your loop has | `Trainer` adds | Why it matters |
|---|---|---|
| `loss.backward(); opt.step(); opt.zero_grad()` | Gradient accumulation, with correct **loss normalization** across accumulated steps (`num_items_in_batch`) | Getting this wrong is Phase 12 failure #10: your loss silently changes value when you change `gradient_accumulation_steps` |
| `.to(device)` by hand | `accelerate` device placement, and multi-GPU for free | Phase 10 |
| — | `max_grad_norm=1.0` gradient clipping | Your defence against the loss spikes in Phase 12 failure #1 |
| A manual LR schedule, if any | Warmup + cosine/linear/WSD, stepped correctly | Phase 04 derives why LoRA wants ~1e-4 and full FT wants ~1e-5 |
| — | Checkpointing that saves optimizer + scheduler + RNG + dataloader position | Phase 12 failure #11: an incomplete resume produces a *different* loss curve |
| A simple `DataLoader` | `dataloader_num_workers`, pinned memory, length grouping | Phase 10 throughput — and `group_by_length` is failure #3 |
| — | `gradient_checkpointing` | The 12.5× activation saving from [NUMBERS.md §2](../../NUMBERS.md). Non-negotiable on 24 GB |

**The honest summary:** `Trainer` is not doing anything conceptually beyond your loop. It is doing about fifteen small things correctly that are each easy to get subtly wrong. Which is exactly why Phase 12 exists — you will meet several of them personally.

---

## 2. Numbers

No new formulas here. Instead, this phase **checks that Phase 00's arithmetic actually describes real models.** Read the `config.json` for each and fill in the table at the top of [NUMBERS.md](../../NUMBERS.md):

```python
from transformers import AutoConfig
for name in ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-v0.3",
             "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM3-3B"]:
    c = AutoConfig.from_pretrained(name)
    print(name, c.num_hidden_layers, c.hidden_size, c.intermediate_size,
          c.vocab_size, c.num_attention_heads, c.num_key_value_heads)
```

| Check | Predicted | Actual | Gap |
|---|---|---|---|
| Llama-3.1-8B `embed + lm_head` params | 1.05 B | | |
| Llama-3.1-8B trunk params | 6.98 B | | |
| Mistral-7B-v0.3 trunk params | 6.98 B — **identical** | | |
| LoRA r=16 all-linear trainable params | 41.9 M (0.52%) | | |

That last row you verify directly with `model.print_trainable_parameters()`. If your hand calculation and the library disagree, one of you is wrong about `target_modules` — and finding out which is the exercise.

---

## 3. Landscape & choice

**Read the source, not the docs.** For these three diffs specifically, the source is clearer than the documentation, because you already know what the code is *trying* to do. Docs explain concepts you already have; source shows you the fifteen small correctnesses.

Where to look:

| Component | File |
|---|---|
| Llama architecture | `transformers/models/llama/modeling_llama.py` |
| LoRA layer | `peft/tuners/lora/layer.py` |
| Training step | `transformers/trainer.py` — search `def training_step` and `def compute_loss` |
| SFT specifics | `trl/trainer/sft_trainer.py`, `trl/trainer/sft_config.py` |

```bash
python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"
```

**Rejected alternative:** working through the HuggingFace course chapters. Good material, but written for someone who has not built a transformer — for you it would spend hours re-teaching attention. Reading source against your own notebooks is strictly faster.

---

## 4. Design

Three things get built.

1. **[docs/BRIDGE.md](../BRIDGE.md)** — the three-column table, filled in from your own reading rather than copied from §1 above. Write it in your own words; the value is in the writing, not the table.

2. **`src/.../train/train_min.py`** — a minimal training loop that uses HuggingFace primitives (`AutoModelForCausalLM`, `AutoTokenizer`, `AdamW`, a manual step loop) but **no `Trainer`**. This is the missing rung between your GPT-2 loop and Phase 04's `SFTTrainer`: same shape as the loop you already wrote, but on a real tokenizer and a real model. When you meet `Trainer` in Phase 04, you will know precisely which lines it replaced.

3. **`tests/test_lora_equivalence.py`** — the credential. Construct your hand-written `LinearWithLoRA` and a `peft` LoRA layer with **identical** `A` and `B` weights, feed both the same input, and assert the outputs match to float tolerance.

**How you will know it worked:**

- [ ] `pytest tests/test_lora_equivalence.py` passes — your LoRA and `peft`'s compute the same function
- [ ] A second test asserts that a freshly-wrapped `peft` model's output is **exactly** equal to the base model's (because `lora_B` is zero-initialized)
- [ ] `train_min.py` runs 20 steps on `Qwen3-0.6B` and the loss goes down
- [ ] `BRIDGE.md` has all three tables filled in, in your own words
- [ ] You can name, without notes, three things `Trainer` does that your GPT-2 loop did not

---

## 5. Build & Q&A

### Commands

```bash
# on the Mac — no GPU needed for any of this
pyenv activate FT_VENV

# find the installed source to read
python -c "import peft, os; print(os.path.dirname(peft.__file__))"

# the equivalence tests
pytest tests/test_lora_equivalence.py -v

# the minimal loop (CPU/MPS is fine for 20 steps on a 0.6B)
python -m llm_finetuning_from_zero_to_hero.train.train_min --steps 20
```

### What was built

_(fill in)_

### Results

_(the passing tests, and the loss curve from `train_min.py`)_

### What surprised you

_(candidates for [docs/LESSONS.md](../LESSONS.md) group **D — when to reach for which method**)_

### Questions

Append to [question_answer.md](../../question_answer.md) as `Q01.n`. Suggested seeds:

- **Q01.1 ⭐** Why is `lora_B` initialized to zero rather than randomly? What property does that guarantee, and what test does it enable?
- **Q01.2 ⭐** What does GQA change about the KV cache, and by how much on Llama-3.1-8B? *(You computed this in `NUMBERS.md` §8.)*
- **Q01.3 ⭐📐** Why does SwiGLU mean most LoRA capacity is in the MLP rather than in attention? Give the parameter counts.
- **Q01.4 🔧** Name three things `Trainer` does for you that a hand-written loop typically gets wrong.
- **Q01.5 ⭐** How does `DPOTrainer` avoid loading a second copy of the model as a reference? *(Hint: a `peft` context manager.)*

### Adversarial close

**What does this now fail on?** Your equivalence test proves the *forward* pass matches. It says nothing about gradients, about the `alpha/r` scaling under `use_rslora`, or about merged-vs-unmerged behaviour. Add whichever of those you can — and note the ones you cannot yet, because Phase 06 will.
