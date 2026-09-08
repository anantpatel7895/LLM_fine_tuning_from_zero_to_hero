# DATA.md — what a training example *is*

**Why this file exists.** Fine-tuning outcomes are decided by data far more than by hyperparameters, and data preparation is the half of the skill set most people never build. This is the deliberate counterweight to [NUMBERS.md](../NUMBERS.md).

> **Forcing function:** if this file is shorter than `NUMBERS.md` when the curriculum ends, the data-prep half of the goal was not met.

**How it grows.** Phase 02–03 fill in templates and masking. Phases 07–09 fill in schemas, filters, funnels and mixtures. Every filter that gets added must record **what it cost** — rows removed, tokens removed, and the measured eval delta. A filter with no measured delta is cargo cult.

---

## Part 1 — The format taxonomy

Every trainer in TRL expects a specific shape, and picking the wrong one produces either a crash or, worse, silent mistraining.

| Format | Shape | Used by | Loss applied to |
|---|---|---|---|
| **Prompt-completion** | `{"prompt": str, "completion": str}` | SFT | completion only |
| **Conversational** | `{"messages": [{"role", "content"}, …]}` | SFT | depends on `assistant_only_loss` — **see Part 3** |
| **Preference (explicit prompt)** | `{"prompt", "chosen", "rejected"}` | DPO, ORPO, CPO, reward modeling | chosen vs rejected margin |
| **Preference (implicit prompt)** | `{"chosen": [msgs], "rejected": [msgs]}` | DPO | prompt inferred as the common prefix |
| **Unpaired preference** | `{"prompt", "completion", "label": bool}` | KTO | per-example, signed |
| **Prompt-only** | `{"prompt": str}` | GRPO, online methods | generated completions, scored by a reward fn |
| **Tool-calling** | `messages` + `tools` (JSON schemas) | SFT | assistant turns incl. tool-call blocks |
| **Raw text** | `{"text": str}` | continued pre-training | everything — no template, no mask |

**Two things worth internalizing early:**

1. **Unpaired preference is the industrial default.** Real production feedback is thumbs-up/thumbs-down on individual responses, not ranked pairs. Most people mis-model this as pairs by arbitrarily pairing an up with a down; KTO exists precisely because that is wrong. Phase 13 does the conversion exercise deliberately.
2. **Continued pre-training is a genuinely different job**, not "SFT on raw text." No template, no masking, everything in loss, pack to full context. It destroys instruction-following, which is why Phase 11 has to re-SFT afterwards.

---

## Part 2 — Chat templates

_(Filled in during Phase 02.)_

### The core fact

`tokenizer.apply_chat_template()` runs a **Jinja program that ships inside `tokenizer_config.json`**. The template is part of the model artifact, not part of your code. Two consequences:

- If you fine-tune with template A and serve with template B, your model is broken in production while scoring fine in your eval. This is the single most common real-world fine-tuning failure.
- When you push a model to the Hub, the template must go with it. A merged model whose `tokenizer_config.json` lost the template you trained with is a broken artifact.

### The three template families

| Family | Turn markers | System role | Notes |
|---|---|---|---|
| **Llama 3.x** | `<\|start_header_id\|>role<\|end_header_id\|>` … `<\|eot_id\|>` | native, and the default template injects a date | Sequence opens with `<\|begin_of_text\|>` |
| **Mistral v0.3** | `[INST] … [/INST]` | **not native** — conventionally prepended into the first user turn | `</s>` terminates assistant turns |
| **ChatML** (Qwen, many others) | `<\|im_start\|>role\n … <\|im_end\|>` | native | The de facto community standard |

**Gemma** is a fourth shape worth rendering (`<start_of_turn>model` … `<end_of_turn>`, no system role) precisely because it breaks assumptions the other three share.

### The drill

Take one identical conversation. Render it through all four. Then **diff the token IDs, not the strings** — string diffs hide the thing that matters, which is how many tokens each special marker costs and whether a BOS appeared twice.

_(Phase 02 fills in the rendered comparison table here.)_

---

## Part 3 — Loss masking: the six silent failures

_(Filled in during Phase 03. This is the highest-value section in the file.)_

**Why "silent":** every one of these produces a smooth, descending, entirely believable loss curve. None of them raises an error. You cannot detect any of them from the loss — only by printing the token stream.

| # | Failure | Cause | Symptom in production |
|---|---|---|---|
| 1 | **Double BOS** | The template already adds `<\|begin_of_text\|>`, then you tokenize with `add_special_tokens=True` | Degraded output quality, no error, very hard to spot |
| 2 | **EOS not in loss** | Assistant turn's `<\|eot_id\|>` / `<\|im_end\|>` is masked out or absent from labels | **The model never learns to stop.** Infinite generation. The #1 "my fine-tune is broken" report |
| 3 | **Training on user turns** | Loss computed over the whole sequence instead of assistant turns only | The model learns to *imitate users* — asks questions instead of answering |
| 4 | **`pad_token = eos_token`** | Padding is masked by label ID, so real EOS tokens get masked too | Failure #2, arriving by a different route. Extremely common |
| 5 | **Double-shifted labels** | The model shifts labels internally; you shift them again | Every label off by one position. Loss still descends. Model learns garbage |
| 6 | **Padding side mismatch** | The same padding side used for training and for generation | Generation starts inside the pad region. **Train with right padding, generate with left padding** (and DPO requires left) |

### The rule

> **Print token / id / label / in-loss for three examples before every new data format.** Phase 03 builds `mask_inspector.py` for exactly this. Every later phase uses it. This is where most fine-tuning failures are born and where almost nobody looks.

### How completion-only loss is actually done in 2026

**The mechanism changed.** `DataCollatorForCompletionOnlyLM` — which every older tutorial uses, with `response_template="### Assistant:"` — **has been removed from TRL.** The replacements:

| Your data shape | What to do |
|---|---|
| **Prompt-completion** — `{"prompt": …, "completion": …}` | Nothing. `completion_only_loss` is **`True` by default** for this format |
| **Conversational** — `{"messages": [...]}` | Set **`assistant_only_loss=True`** |

⚠️ **`assistant_only_loss=True` requires the chat template to contain `{% generation %}` … `{% endgeneration %}` markers** around the assistant content. TRL auto-patches some known families (Qwen3 among them), but for other models it **silently fails to mask what you expect.**

That is exactly why Phase 02's deliverable is *a chat template you wrote yourself, in version control, with tests*, and why Phase 03 builds `mask_inspector.py`. **Verify per model. Do not trust the flag.**

Other sharp edges to verify on your own versions in Phase 03:

- silently ignored under some fused-kernel paths
- silent failure when a sequence exceeds `max_length` — and note `max_length` now defaults to only **1024**

### Packing — the cross-contamination warning is out of date

Older material warns that `packing=True` lets one packed example attend to another. **That describes the old behaviour.**

| Strategy | Behaviour |
|---|---|
| `packing_strategy="bfd"` | **The current default.** Best-fit-decreasing bin packing, and it **auto-enables `padding_free`** — so packed sequences do *not* attend across boundaries |
| `packing_strategy="wrapped"` | The old aggressive behaviour: concatenate and cut mid-sequence. This is what the warnings were about |
| `packing_strategy="bfd_split"` | Renamed from `"bfd-requeue"` |

So `packing=True` is safe now and is what you want. **Understand the mechanism anyway** — *"how does packing avoid cross-attention between examples?"* is a common interview question, and the answer is `position_ids` plus variable-length attention, not magic.

Full migration details in [notes/api-2026.md](notes/api-2026.md).

### Multi-turn decisions with real consequences

- **Loss on all assistant turns, or only the last?** All turns gives more signal per example, but early turns are conditioned on *ground-truth* context, whereas at deployment they are conditioned on *model-generated* context. There is a train/serve mismatch either way; know which one you chose.
- **How to truncate a conversation longer than `max_length`?** Drop the example, or drop whole early turns. **Never truncate mid-assistant-turn** — that literally trains the model to stop mid-sentence.

### Adding special tokens correctly

The four-step sequence, all of which are required:

1. `tokenizer.add_special_tokens(...)`
2. `model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)` — the alignment matters for tensor cores
3. **Mean-initialize the new embedding rows** from the existing embedding distribution. Fresh rows are random noise and will produce large early gradients
4. Make the output side trainable: `modules_to_save=["embed_tokens", "lm_head"]` — otherwise the model can *read* the new token but can never *emit* it

---

## Part 4 — The record schema

_(Filled in during Phase 07.)_

**Schema is a one-way door.** A field you did not capture cannot be backfilled without regenerating the dataset; a field you declared and never used cost nothing. So over-declare.

```python
class Example(BaseModel):
    id: str                       # stable, content-derived
    messages: list[Message]       # or text, for CPT
    source: str                   # dataset name
    source_url: str | None
    license: str                  # propagate it or you cannot release
    lang: str
    n_tokens: int                 # tokenized with THIS project's tokenizer
    n_turns: int
    quality_score: float | None   # from Phase 07 scorers
    dedup_cluster_id: str | None
    contamination_flags: list[str]
    split: Literal["train", "eval", "test"]
```

**Versioned on disk:**

```
data/processed/<name>/v1/
├── train.jsonl
├── eval.jsonl
├── manifest.json     # rows, token histogram, content hash, filter funnel
└── datasheet.md      # motivation, composition, collection, preprocessing, uses, license
```

---

## Part 5 — The filter funnel

_(Filled in during Phase 07. Every row must have a measured eval delta or it does not stay.)_

| Stage | Rows in | Rows out | Tokens | Eval delta | Verdict |
|---|---:|---:|---:|---:|---|
| raw | | | | — | |
| schema validation | | | | | |
| length filter (token-based) | | | | | |
| language ID | | | | | |
| markup/HTML residue | | | | | |
| refusal & assistant-tic slop | | | | | |
| degenerate repetition | | | | | |
| duplicate-prompt conflicts | | | | | |
| exact dedup | | | | | |
| fuzzy dedup (MinHash-LSH) | | | | | |
| semantic dedup | | | | | |
| decontamination vs eval sets | | | | | |
| quality threshold | | | | | |
| **final** | | | | | |

### Notes on individual stages

- **Length filters must be token-based, not character-based.** Characters per token varies ~3× across languages and code, so a character filter silently discriminates by language.
- **Refusal and assistant-tic slop** — "As an AI language model", "Certainly!", "I hope this helps!" — is what makes a fine-tune *sound* like a bad chatbot. This filter has an outsized effect on perceived quality relative to its simplicity.
- **Duplicate prompt, different answer** is the interesting conflict case. Picking one arbitrarily teaches the model inconsistency. Decide a policy (highest quality score wins / drop both) and record it.
- **Four levels of duplicate, four different tools:** exact (content hash) → fuzzy (MinHash-LSH, see [NUMBERS.md §10](../NUMBERS.md)) → semantic (embed + threshold; catches paraphrases MinHash structurally cannot) → **cross-split** (train-vs-eval leakage, the one that silently inflates your own numbers).
- **Decontamination** is 13-gram containment against every benchmark test split you intend to report, *plus* embedding-nearest-neighbour for the paraphrase cases n-grams cannot see.
- **Three independent quality signals**, because they disagree and the disagreement is informative: a reward-model score, an LLM-rubric score, and a model-intrinsic score (perplexity or instruction-following-difficulty, which needs no external model at all).

---

## Part 6 — Synthetic data

_(Filled in during Phase 08.)_

| Method | Mechanism | What it teaches |
|---|---|---|
| Self-Instruct | Bootstrap new tasks from seed tasks | The basic generation loop |
| Evol-Instruct | Mutate instructions along depth/breadth operators | Instruction complexity as a controllable axis |
| **Magpie** | Prompt an aligned model with *only* the user-turn template prefix and let it hallucinate the instruction | That the chat template **is an attack surface** — closes the loop with Part 2 |
| Persona-driven | Condition generation on a persona for coverage | Diversity as an explicit objective |
| Rejection sampling / best-of-n | Generate `n`, keep what a verifier or reward model approves | Same machinery as Phase 14 — arriving early |
| Teacher distillation | Copy a stronger model's outputs | Licensing constraints become real |

**Licensing is a first-class engineering constraint here, not a footnote.** Record the generator model, its license, and the derived-work implications in the datasheet. Llama-license outputs, Apache-2.0 outputs, and commercial-API outputs have materially different downstream rights. Getting this right reads as senior.

**Failure modes, all measurable:**

- **Mode collapse / diversity loss** — measure with distinct-n, self-BLEU, or embedding-space coverage. Plot it against generation temperature.
- **Teacher error propagation** — the student inherits the teacher's mistakes with higher confidence.
- **Format leakage** — the generator's own chat template artifacts end up inside your content.
- **Same model as generator and judge** — a closed loop that scores itself well and teaches nothing.

---

## Part 7 — Mixing and ordering

_(Filled in during Phase 09.)_

- **Per-domain weights** (chat / code / math / tool-calling / safety / multilingual) and how each shifts behaviour.
- **Tokens-per-source vs epochs-per-source are not the same thing** when average lengths differ. Always weight by tokens.
- **Replay / anchor data** — 1–10% general instruction data mixed into domain training, to prevent the catastrophic forgetting measured in Phase 11.
- **Curriculum ordering** (easy→hard, short→long) tested against plain shuffling. Report the honest result; shuffling usually wins for SFT, and knowing that saves weeks.
- **How much data is actually enough** — the LIMA question. Run it on your own pool: 1k top-scored examples versus 10k random from the *same* source at matched compute. Both directions happen; knowing *when* each wins is the expertise.

---

## Part 8 — The dataset catalogue

All verified present on the HuggingFace Hub with these licenses and sizes as of September 2026. **Re-check the license at point of use** — some are gated and some have changed.

⚠️ **License matters more than you think here.** Capstone B releases a public dataset, and a license you cannot propagate makes that impossible. Record the license in the datasheet from day one (Part 4).

### Instruction SFT

| Dataset | Rows | License | Why |
|---|---:|---|---|
| `HuggingFaceH4/no_robots` | 10,000 | ⚠️ **CC-BY-NC-4.0** | ⭐ The ideal *teaching* set: 10k **human-written** examples, one epoch in ~3 hrs, and free of the synthetic slop that would confound your first measurement. **Non-commercial** — fine for learning, call it out |
| `allenai/tulu-3-sft-mixture` | 939,343 | **ODC-BY** | ⭐ The best-documented modern mixture; the whole Tulu 3 recipe is published, so it is worth *studying* as a reference for Phase 09 |
| `HuggingFaceTB/smoltalk` | 1M+ | (unset on Hub) | ⭐ The SmolLM2/3 SFT mix — pairs with `SmolLM3-3B` for a fully-open end-to-end story |
| `teknium/OpenHermes-2.5` | 1M+ | ⚠️ **no license tag** | Widely used and deliberately messy — good Phase 07 cleaning material. Flag the license |
| `databricks/databricks-dolly-15k` | 15,000 | CC-BY-SA-3.0 | Historical baseline; quality is dated |

### Multi-turn chat

| Dataset | Rows | License | Why |
|---|---:|---|---|
| `trl-lib/Capybara` | 16,006 | (unset) | ⭐ **TRL's own docs use this as the canonical multi-turn example.** Already in `messages` format — zero preprocessing. Start here |
| `HuggingFaceH4/ultrachat_200k` | 515,311 | **MIT** | ⭐ Cleanest license, right size for a real run |
| `allenai/WildChat-1M` | 837,989 | ODC-BY | Real user↔assistant conversations — realistic distribution, contains toxicity |
| `lmsys/lmsys-chat-1m` | 1,000,000 | ⚠️ custom/gated | Real Chatbot-Arena traffic. Check the terms |

### Tool / function calling — the golden thread

| Dataset | Size | License | Why |
|---|---|---|---|
| `NousResearch/hermes-function-calling-v1` | 10k–100k | **Apache-2.0** | ⭐ **Top pick for the golden thread.** Best license, multi-turn tool use, actively updated |
| `Salesforce/xlam-function-calling-60k` | 10k–100k | **CC-BY-4.0** | ⭐ The reference single-turn function-calling set |
| `Team-ACE/ToolACE` | 10k–100k | Apache-2.0 | Complex and parallel calls |
| `interstellarninja/hermes_reasoning_tool_use` | 51,004 | Apache-2.0 | Tool use *with* reasoning traces — a nice bridge |

⭐ **TRL v1 has first-class tool-calling support** in both `SFTTrainer` and `DPOTrainer`: put `tool_calls` and `tool`-role messages in `messages`, and the JSON schemas in a **`tools` column**. Much cleaner than the hand-rolled prompt formats older tutorials use.

### Reasoning traces

| Dataset | Rows | License | Why |
|---|---:|---|---|
| `GAIR/LIMO` | **817** | Apache-2.0 | ⭐⭐ **817 examples.** Perfect pedagogy — demonstrates that reasoning can be elicited by a tiny high-quality set. The ideal first reasoning experiment on an A10 |
| `open-r1/OpenR1-Math-220k` | 100k–1M | **Apache-2.0** | ⭐ The most-used open reasoning set. Good default |
| `open-thoughts/OpenThoughts-114k` | 100k–1M | Apache-2.0 | ⭐ Manageable size |
| `nvidia/OpenMathReasoning` | 5,678,317 | CC-BY-4.0 | Huge — subsample |

### Preference pairs (DPO, Phase 13)

| Dataset | Rows | License | Why |
|---|---:|---|---|
| `trl-lib/ultrafeedback_binarized` | 63,135 | (unset) | ⭐ **Exactly what TRL's DPO docs use.** Correct `prompt`/`chosen`/`rejected` schema, zero preprocessing. Start here |
| `HuggingFaceH4/ultrafeedback_binarized` | 187,405 | **MIT** | The original; best license |
| `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 272,898 | ODC-BY | ⭐ Modern, documented recipe |
| `nvidia/HelpSteer3` | 100k–1M | CC-BY-4.0 | ⭐ Human-annotated, **multi-attribute graded** — which makes it the right source for the graded→pairs→KTO-labels conversion exercise in Phase 13 |
| `Skywork/Skywork-Reward-Preference-80K-v0.2` | 77,016 | ⚠️ unset | Strong for the Phase 14 reward-model work |

### Continued pre-training corpora (Phase 11)

| Dataset | Size | License | Domain shift |
|---|---|---|---|
| `HuggingFaceTB/finemath` | 10M–100M | **ODC-BY** | ⭐ **Math.** Well-sized for a 1–2 B token run on this box |
| `pile-of-law/pile-of-law` | 10M–100M | ⚠️ CC-BY-NC-SA-4.0 | ⭐ **Legal** — a very clear, measurable shift. Non-commercial |
| `allenai/peS2o` | 10B–100B | ODC-BY | ⭐ **Scientific papers.** Subsample |
| `HuggingFaceFW/fineweb-edu` | 1B–10B docs | **ODC-BY** | ⭐ **The replay mix.** Not a domain shift — this is the 10–30% general data you blend in to prevent catastrophic forgetting. An essential and usually-omitted step |
| `bigcode/the-stack-v2` | very large | ⚠️ per-file licenses | **Code.** Licensing needs care |
| `HuggingFaceFW/fineweb-2` | 1B–10B | ODC-BY | **Multilingual** shift |

**Best Phase 11 combination:** `finemath` or `pile-of-law` for a sharp measurable shift, **mixed with 10–30% `fineweb-edu` as replay**, full-fine-tuning `SmolLM3-3B` with FSDP2. That is ~1–2 B tokens ≈ 4–14 days of GPU time, which the box has for free.

---

## Glossary

Terms get defined here on first use, per the mentor-mode rule in [PLAN.md](../PLAN.md).

| Term | Meaning |
|---|---|
| **SFT** | Supervised fine-tuning — train on (input, desired output) pairs. The baseline technique |
| **LoRA** | Low-Rank Adaptation — freeze the base model, train two small matrices per layer whose product is the weight update |
| **QLoRA** | LoRA on top of a 4-bit-quantized frozen base. A capacity tool, not a speed tool |
| **Adapter** | The trained LoRA matrices, saved separately from the base model. ~84 MB for an 8B at r=16 |
| **Chat template** | The Jinja program in `tokenizer_config.json` that turns a message list into a token string |
| **Loss mask** | Which token positions contribute to the loss. Positions set to `-100` in `labels` are ignored |
| **Packing** | Concatenating multiple short examples into one full-length sequence to avoid wasting compute on padding |
| **Decontamination** | Removing training examples that overlap your evaluation sets |
| **DPO** | Direct Preference Optimization — train on (chosen, rejected) pairs without a separate reward model |
| **GRPO** | Group Relative Policy Optimization — RL that estimates advantage from a group of sampled completions, dropping the value network |
| **Catastrophic forgetting** | Losing pre-existing capabilities while learning the new task. Measured, not assumed |
| **MFU** | Model FLOPs Utilization — the fraction of the GPU's theoretical peak you actually achieve |
| **GQA** | Grouped-Query Attention — fewer KV heads than query heads, shrinking the KV cache 4× on Llama-3 |
