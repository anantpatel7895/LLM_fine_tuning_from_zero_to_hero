# NUMBERS.md — every formula, with a worked example on 2× A10

**Why this file exists.** Fine-tuning interviews are disproportionately *derivation* questions: "how much VRAM does a full fine-tune of an 8B need?", "why did enabling ZeRO-3 make it slower?", "how many concurrent requests can you serve?" These have real answers you can compute in your head. This file is where you learn to.

**How to use it.** Rule 2 of [PLAN.md](PLAN.md): before every run you **predict** the number, then you **measure** it, then you **explain the gap**. Each section below has a `Measured` row that starts empty and gets filled in from a real reading on the box. The gap is the lesson.

**Notation used throughout**

| Symbol | Meaning |
|---|---|
| `N` | number of model parameters |
| `Ψ` | number of *trainable* parameters (= `N` for full FT, much smaller for LoRA) |
| `L` | number of transformer layers |
| `h` | hidden size (`d_model`) |
| `V` | vocabulary size |
| `b` | micro-batch size |
| `s` | sequence length |
| `G` | number of GPUs |

**Reference model dimensions** (verified from each model's `config.json` on the Hub):

| Model | `N` | `L` | `h` | `i` (FFN) | `V` | kv_dim | embed+lm_head params |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 8,030,261,248 | 32 | 4096 | 14336 | 128,256 | 1024 | 1,050,673,152 |
| Mistral-7B-v0.3 | 7,248,023,552 | 32 | 4096 | 14336 | 32,768 | 1024 | 268,435,456 |
| Qwen3-1.7B | 2,031,739,904 | 28 | 2048 | — | 151,936 | — | — |
| Qwen3-0.6B | 751,632,384 | — | — | — | 151,936 | — | — |
| SmolLM3-3B | 3,075,098,624 | — | — | — | — | — | — |
| Llama-3.2-1B | 1,235,814,400 | 16 | 2048 | 8192 | 128,256 | 512 | — |

> ### ⭐ The comparison that teaches the most
>
> Subtract the embeddings from those first two rows:
>
> ```
> Llama-3.1-8B trunk:  8,030,261,248 − 1,050,673,152 = 6,979,588,096
> Mistral-7B-v0.3 trunk: 7,248,023,552 −   268,435,456 = 6,979,588,096
> ```
>
> **The trunks are identical to the parameter.** The entire 782 M parameter difference between these two models is *vocabulary size* — 128,256 versus 32,768.
>
> Consequences you can now predict without running anything: identical activation cost, 1.46 GiB less weight memory for Mistral, and a **4× cheaper loss head** (see §3). When a Llama run OOMs and the "same size" Mistral run does not, this is why.

⚠️ **A caution on model choice.** `Qwen3.5-*` and `gemma-4-*` are `ForConditionalGeneration` (multimodal) architectures and will not behave like plain text models — see [docs/notes/api-2026.md §11](docs/notes/api-2026.md). Stick to pure `ForCausalLM` models for this curriculum.

---

## §1 — Memory per parameter: the table that governs everything

This is the single most important derivation in the repo. Everything about which model you can train follows from it.

**Where the bytes go, for a mixed-precision AdamW full fine-tune:**

| Component | Precision | bytes/param |
|---|---|---:|
| Parameters (compute copy) | bf16 | 2 |
| Gradients | bf16 | 2 |
| Optimizer: fp32 master weights | fp32 | 4 |
| Optimizer: AdamW momentum `m` | fp32 | 4 |
| Optimizer: AdamW variance `v` | fp32 | 4 |
| **Total** | | **16** |

Then, per regime — **bytes per *trainable* parameter**, which is the distinction that makes LoRA cheap:

| Regime | B/trainable param | Why |
|---|---:|---|
| fp32 weights + fp32 grads + AdamW | 16 | the old-school setup |
| **AMP bf16 + fp32 master + AdamW** (the common default) | **18** | bf16 compute copy *and* fp32 master copy both exist |
| pure-bf16 weights & grads + fp32 AdamW | 12 | no fp32 master copy — FSDP2 can do this, DeepSpeed cannot (see below) |
| pure-bf16 + `bitsandbytes` AdamW8bit | 6 | `m` and `v` drop from 4+4 to 1+1 |
| pure-bf16 + Adafactor | ~4 | factored second moment, effectively no `v` |

### Weights, precisely

```
bf16:        W = 2 · N
NF4 + double quant:  W = 2 · N_embed  +  (4.127 / 8) · N_trunk
NF4, no double quant: W = 2 · N_embed  +  (4.5   / 8) · N_trunk
```

**The detail almost every tutorial omits: `bitsandbytes` leaves `nn.Embedding` unquantized and skips `lm_head` by default.** So on a large-vocabulary model those layers stay in bf16 and cost real memory.

Where does 4.127 bits come from? 4 bits per weight, plus an 8-bit absmax scale per block of 64 weights (0.125 b/param), plus an fp32 second-level absmax per 256 blocks (0.002 b/param). Without double quantization the scale is fp32 per block of 64 → 0.5 b/param → 4.5 bits total.

**Worked example — Llama-3.1-8B (`N` = 8.03 B, `N_embed` = 1.05 B, `N_trunk` = 6.98 B):**

```
Full FT, AMP + AdamW:  8.03e9 × 18 B = 134.6 GiB   → 2.8× more than your 48 GB. No.
Full FT, 8-bit AdamW:  8.03e9 ×  6 B =  44.9 GiB   → still no. (48 GB is not a pool —
                                                      it is 24 + 24, and each rank
                                                      needs the whole thing under DDP)
bf16 weights:          8.03e9 ×  2 B =  16.1 GiB
QLoRA NF4 weights:     2 × 1.05e9 + (4.127/8) × 6.98e9
                     = 2.10 GiB + 3.60 GiB = 5.31 GiB    ← not 4.0 GiB. The embeddings
                                                            are a third of it.
LoRA adapter (r=32, all-linear), grads + optimizer:  ≈ 1.25 GiB   (see §5)
```

**Conclusions for this box, derived not looked up:**

| | Ceiling on 2× A10 |
|---|---|
| Full fine-tune, plain FSDP2, no tricks | **3B class** |
| Full fine-tune, with an 8-bit optimizer | 4B class, tight |
| Full fine-tune of an 8B | only with **DeepSpeed ZeRO-3 + CPU optimizer offload**, and 3–5× slower |
| **bf16 LoRA** | **4B class** if you want headroom — an 8B needs ~20 GiB at micro-batch 1 |
| **QLoRA** | **8B comfortably**, 14B+ at a squeeze |

> ### The framing to remember
>
> **On 24 GB cards, bf16 LoRA is a 4B technique and QLoRA is the 8B technique.**
>
> bf16 LoRA on an 8B *technically* fits — 20.1 GiB with gradient checkpointing at sequence 2048 — but with roughly 2 GiB of headroom and micro-batch 1. One long example in the dataset OOMs it. That is not a configuration to learn on.

### And 24 GB is never 24 GB

Subtract before you plan:

| Overhead | Typical |
|---|---|
| CUDA context | 300–600 MiB |
| cuBLAS / NCCL workspaces (multi-GPU) | 100–500 MiB |
| Allocator fragmentation | 5–15% of allocated |
| Display / other processes | usually 0 on a headless box — verify |

Budget **~23.0 GiB usable** per card, not 24. Confirm the real number with the Phase 00 benchmark.

**KV cache during training is zero** — `use_cache=False`, which is now the default in `transformers` v5 configs during training. KV cache only matters for serving (§8).

| | Value |
|---|---|
| Predicted | as above |
| **Measured** | _(Phase 00 / Phase 04 — `torch.cuda.max_memory_allocated()`)_ |

---

## §2 — Activation memory

Activations are the term everyone forgets, and the term that actually causes the OOM at step 4000.

**Per token, per layer**, counting the tensors autograd has to retain (residual stream, post-RMSNorm, q/k/v, attention output, `o_proj` input, post-attention norm, gate/up/silu/silu·up, `down_proj` output):

```
A₁ = 2 × (6h + 2·kv_dim + 4i)     bytes per token per layer      (bf16)
```

Then:

```
no checkpointing:        A = L × s × b × A₁
gradient checkpointing:  A = L × s × b × h × 2   +   s × b × A₁
                             └ stored layer inputs ┘   └ one layer's recompute peak ┘
```

**Worked example — Llama-3.1-8B (`h`=4096, `kv_dim`=1024, `i`=14336, `L`=32, `b`=1):**

```
A₁ = 2 × (6×4096 + 2×1024 + 4×14336) = 2 × 83,968 = 167,936 B/token/layer
   → 5.12 MiB per token across the whole 32-layer stack

s=2048, checkpointing OFF:  32 × 2048 × 167,936        = 10.25 GiB
s=2048, checkpointing ON:   32 × 2048 × 4096 × 2  + …  =  0.82 GiB
s=4096, checkpointing OFF:                             = 20.50 GiB   → OOM
s=4096, checkpointing ON:                              =  1.64 GiB
```

> **Gradient checkpointing buys 12.5× on activations** — 10.25 GiB down to 0.82 GiB at sequence 2048 — for roughly a 30% throughput cost. At sequence 4096 it is not a choice: 20.5 GiB of activations plus 5.3 GiB of weights plus the loss head does not fit in 23 GiB.
>
> **This is the single most instructive experiment available on your hardware.** Phase 04 runs it both ways.

**Activations, not weights, are what kill you.** At sequence 4096 with checkpointing off, the activations (20.5 GiB) are nearly 4× the QLoRA weights (5.3 GiB). Beginners size their model to the card and then OOM anyway, because they budgeted the wrong term.

**No `s²` attention-matrix term** appears above, because both `sdpa` (the `transformers` default) and FlashAttention-2 avoid materializing the `b × heads × s × s` score matrix. Without either, at `s`=4096 and 32 heads that term alone would be `1 × 32 × 4096² × 2 B = 1.07 GiB **per layer**`. The formula above assumes you are using one of them — on A10, `sdpa` is fine.

**Batch headroom, which is the practical use of this formula.** With checkpointing on at `s`=2048, each additional sequence in the micro-batch costs ~0.82 GiB. So for QLoRA on Llama-3.1-8B (5.31 weights + 1.25 adapter + 1.47 logits + 1.6 overhead ≈ 9.6 GiB fixed): micro-batch 8 ≈ 16.3 GiB is safe, micro-batch 12 ≈ 19.5 GiB is tight. **Prefer a real micro-batch over `gradient_accumulation_steps` up to that point** — accumulation costs wall-clock, a bigger micro-batch does not.

> ⚠️ **This formula is an analytic upper bound.** Real kernels fuse some operations, so expect **70–100%** of the predicted value. Calibrate it in Phase 04 against `torch.cuda.max_memory_allocated()` and write your measured coefficient here.

| | Value |
|---|---|
| Predicted | 0.82 GiB at `b`=1, `s`=2048, checkpointed · 1.64 GiB at `s`=4096 |
| **Measured** | _(Phase 04 — and correct the `A₁` coefficient above)_ |

---

## §3 — The logits trap

The most common surprise OOM, and the one that has nothing to do with model size.

Naive cross-entropy holds the bf16 logits, an fp32 upcast, and the fp32 gradient all at once:

```
naive:            Lg ≈ b × s × V × (2 + 4 + 4) ≈ b × s × V × 8–10 bytes
chunked / fused:  Lg ≈ b × chunk × V × ~6      — bounded, does NOT scale with s
```

**Worked example — Llama-3.1-8B, `V` = 128,256, `b`=1:**

| `s` | naive | chunked |
|---:|---:|---:|
| 2048 | **2.45 GiB** | 1.47 GiB |
| 4096 | **4.89 GiB** | 1.47 GiB |

**For a single sequence.** At sequence 4096 the naive logits tensor (4.89 GiB) is *larger than the entire 4-bit model* (5.31 GiB — nearly as large). It is routinely the single biggest allocation in a run, and it has nothing to do with how many parameters the model has.

**Three consequences:**

1. **`trl` v1 defaults `SFTConfig.loss_type="chunked_nll"` precisely because of this.** Older tutorials predate that default and describe the naive behaviour. Do not turn it off casually — and note it is **incompatible with `use_liger_kernel=True`**, which forces `"nll"`. See [docs/notes/api-2026.md](docs/notes/api-2026.md).
2. It scales with **vocabulary size**, not parameter count. This is where the identical-trunk comparison at the top of this file pays off: Mistral-7B's 32,768-vocab loss head at `s`=4096 costs `1.25 GiB naive / 0.38 GiB chunked` — **4× less than Llama's** for exactly the same compute elsewhere.
3. Which makes **Mistral-7B-v0.3 the roomiest 7B-class configuration on this box** — a good reason to run your very first QLoRA there rather than on Llama.

---

## §4 — Training FLOPs and throughput

**FLOPs per token:**

```
full fine-tune:  ≈ 6N       (2N forward + 4N backward)
LoRA:            ≈ 4N + recompute overhead ≈ 5N to 6N
```

The `6N` breaks down as: forward is `2N` (one multiply + one add per parameter), backward is `4N` (gradients w.r.t. both inputs and weights). LoRA skips weight-gradient computation for frozen layers but adds recompute from gradient checkpointing, so in practice it lands close to full FT per token — **LoRA saves memory, not compute.** Internalize that; it surprises people.

**Then throughput:**

```
tokens/sec = (peak_TFLOPS × MFU) / FLOPs_per_token
```

`MFU` (Model FLOPs Utilization) is the fraction of theoretical peak you actually achieve. Realistic ranges: **25–40%** for a well-tuned run on an A10, lower with heavy checkpointing, much lower if you are memory-bound or comms-bound.

**Worked example — Llama-3.1-8B LoRA on one A10** (125 TFLOPS bf16 peak, assume 30% MFU):

```
FLOPs/token ≈ 5 × 8.03e9   = 4.0e10
tokens/sec  = (125e12 × 0.30) / 4.0e10
            = 3.75e13 / 4.0e10
            ≈ 937 tok/s      → call it 600–900 tok/s in practice
```

**Epoch time from a token budget:**

```
epoch_seconds = dataset_tokens / tokens_per_sec
```

A 10 000-example instruction set averaging ~800 tokens = **8 M tokens**:

```
8e6 / 750 ≈ 10 700 s ≈ 3.0 hours per epoch on one card
```

Which is exactly why you launch it Friday night.

### Modeled throughput and daily token budget

> ⚠️ **Every number in this subsection is a model, not a measurement.** It assumes an MFU
> (15–30%) and a communication efficiency, and could plausibly be off by ±50%. Treat it as
> an order of magnitude for planning, replace it with your own measurements from Phase 04
> and Phase 10, and note the gap.

Using `tok/s = G × 125e12 × MFU × comm_eff / (F × N)`, with `F ≈ 6` FLOPs/param/token for LoRA with checkpointing and `F ≈ 8` for full FT with checkpointing (2 forward + 4 backward + ~2 recompute):

| Configuration | MFU | comm_eff | tok/s | **M tokens/day** |
|---|---:|---:|---:|---:|
| QLoRA 8B, 1× A10 | 15% | 100% | 389 | **33.6** |
| QLoRA 8B, 2× A10 DDP | 15% | 98% | 763 | **65.9** |
| bf16 LoRA 8B, 1× A10 | 25% | 100% | 649 | **56.0** |
| bf16 LoRA 8B, 2× A10 DDP | 25% | 98% | 1 271 | **109.8** |
| Full FT 1.7B, 2× A10 ZeRO-2 | 30% | 75% | 3 464 | **299.3** |
| Full FT 3B, 2× A10 ZeRO-2/3 | 28% | 60% | 1 705 | **147.3** |
| Full FT 8B, ZeRO-3 + CPU offload | 25% | 25% | 243 | **21.0** |

Note the last row: CPU offload does not just cost memory bandwidth, it collapses effective throughput to a *quarter* of a comparable sharded run. That is the honest cost of pushing an 8B full fine-tune onto this box.

**What this means for continued pre-training (Phase 11).** A domain-shift run wants 0.5–5 B tokens:

- Full FT of a 3B at ~150 M tok/day → **3 days to 5 weeks**. Feasible.
- QLoRA on an 8B at ~66 M tok/day → a 1 B-token run is **~15 days**. Feasible if you are patient.
- Pre-training from scratch: Llama-3 used ~15 T tokens. On this box that is **roughly 400 years.** Not feasible, and worth stating plainly so the boundary is clear.

| | Value |
|---|---|
| Predicted | ~390 tok/s QLoRA 8B single card, ~3 hrs/epoch on 8 M tokens |
| **Measured** | _(Phase 04)_ |

---

## §5 — LoRA parameter count

For each replaced linear layer of shape `d_in × d_out`, LoRA adds two matrices `A (r × d_in)` and `B (d_out × r)`:

```
lora_params_per_module = r × (d_in + d_out)
```

**Worked example — Llama-3.1-8B, rank 16, all linear layers.** Per layer (`h`=4096, kv projections are 1024 wide because of GQA with 8 kv heads × 128 head_dim, MLP intermediate is 14336):

| Module | `d_in` | `d_out` | `r × (d_in + d_out)` |
|---|---:|---:|---:|
| `q_proj` | 4096 | 4096 | 131 072 |
| `k_proj` | 4096 | 1024 | 81 920 |
| `v_proj` | 4096 | 1024 | 81 920 |
| `o_proj` | 4096 | 4096 | 131 072 |
| `gate_proj` | 4096 | 14336 | 294 912 |
| `up_proj` | 4096 | 14336 | 294 912 |
| `down_proj` | 14336 | 4096 | 294 912 |
| **per layer** | | | **1 310 720** |

```
total = 1.311e6 × 32 layers = 41.9 M trainable parameters
      = 41.9e6 / 8.03e9      = 0.52% of the model
adapter size (bf16) = 41.9e6 × 2 B = 83.8 MB
```

**At rank 32** (the other common choice, and the one the §1 and §2 examples use): double it — `83.9 M` trainable parameters, `1.04%` of the model, `168 MB` of bf16 adapter, and `83.9e6 × 16 B ≈ 1.25 GiB` of gradients plus optimizer state.

**Remember these numbers — r=16: 42 M / 0.52% / 84 MB. r=32: 84 M / 1.04% / 168 MB.** They come back in §6 (communication volume) and in Phase 16 (multi-adapter serving economics).

### Two different "MLP share" figures — do not confuse them

Both are correct and they answer different questions. Mixing them up is easy, so pin them down:

| Question | Computation, per layer of Llama-3.1-8B | MLP share |
|---|---|---|
| What fraction of the **base model's** parameters are in the MLP? | MLP `3 × 4096 × 14336` = 176.2 M, attention = 41.9 M, total 218.1 M | **81%** |
| What fraction of the **LoRA adapter's** parameters land in the MLP (r=16)? | MLP `884 736`, attention `425 984`, total `1 310 720` | **67%** |

**Why they differ, and it is not a rounding artefact.** Base parameters scale with `d_in × d_out` — a *product*. LoRA parameters scale with `r × (d_in + d_out)` — a *sum*. A wide matrix like `gate_proj` (4096 × 14336) is enormous in the base model but its LoRA adapter is only `16 × 18432`. **LoRA deliberately flattens the size differences between matrices**, which is exactly why a rank-16 adapter can be 0.52% of the model while still reaching every part of it.

Either figure supports the same conclusion — **target all linear layers, not attention-only** — because most parameters *and* most adapter capacity sit in the MLP. Attention adapters tend to move style; MLP adapters move knowledge. Phase 06 ablates this.

**Sanity check you can always run:** `model.print_trainable_parameters()`. If it says `0.00%`, your `target_modules` matched nothing and your loss will still descend (see Phase 12 failure #5).

---

## §6 — Communication volume, and the derivation that decides Phase 10

This is the section where your specific hardware — two A10s on PCIe with **no NVLink** — teaches something most people never learn.

**DDP all-reduce volume per optimizer step:**

```
bytes_transferred ≈ 2 × Ψ_trainable × bytes_per_param
```

(The factor 2 is the ring all-reduce: one reduce-scatter pass plus one all-gather pass.)

**Your interconnect:** PCIe, measured in Phase 00. Expect **~15–22 GB/s effective** — versus ~300 GB/s for an NVLinked pair. That 15× difference is the whole story.

ZeRO-3 / FSDP `FULL_SHARD` is worse than DDP on volume, not better: it all-gathers parameters in the forward pass, all-gathers them *again* in the backward pass (when `reshard_after_forward=true`), and reduce-scatters gradients — roughly `3 × N × 2` bytes per step.

**The volume table, at 22 GB/s effective PCIe:**

| Configuration | Bytes moved per step | Pure PCIe time |
|---|---:|---:|
| **LoRA r=32, DDP** (gradient all-reduce only) | 0.17 GB | **7.6 ms** |
| Full FT 3B, ZeRO-3 / FSDP `FULL_SHARD` | 9.24 GB | 420 ms |
| Full FT 8B, DDP (hypothetical — does not fit anyway) | 16.06 GB | 730 ms |
| **Full FT 8B, ZeRO-3 / FSDP `FULL_SHARD`** | **24.09 GB** | **1 095 ms** |

For reference: PCIe Gen4 x16 is 64 GB/s on paper, ~22 GB/s realistically achievable for collectives. **NVLink on an A100 pair is 600 GB/s — about 25× more.**

**Two conclusions you derive rather than read:**

1. **DDP is free for LoRA.** 7.6 ms against a step time of roughly a second is under 1% overhead. Never reach for FSDP or ZeRO for LoRA work on this box.
2. **Sharding a big model over PCIe is a bad trade.** Over a second per step of pure communication, before any compute. Expect roughly a **3–5× slowdown** versus an NVLinked box.

> **Treat GPU 1 as a second *experiment slot*, not a second *worker*.**
>
> That inversion is genuine expert judgment, specific to your hardware, and it falls straight out of this arithmetic. It is also the single best thing in this repo to be able to explain in an interview.

### Per-rank memory by sharding strategy

Per-rank cost is `N × (B/param) / G  +  activations`, where `B/param` comes from §1 and only the sharded components divide:

| Strategy | Params | Grads | Optimizer |
|---|---|---|---|
| DDP / `NO_SHARD` / ZeRO-0 | full | full | full |
| ZeRO-1 | full | full | ÷`G` |
| ZeRO-2 / FSDP `SHARD_GRAD_OP` | full | ÷`G` | ÷`G` |
| ZeRO-3 / FSDP `FULL_SHARD` | ÷`G` | ÷`G` | ÷`G` |

**Worked out for a full fine-tune of Llama-3.1-8B across `G`=2, gradient checkpointing on, `s`=2048, against 23 GiB usable per rank:**

| Scenario | GiB/rank | CPU RAM | Fits? |
|---|---:|---:|:---:|
| ZeRO-3, bf16 params + fp32 grads & optimizer (16 B/p) | 62.2 | 0 | ❌ |
| ZeRO-3, pure-bf16 + fp32 AdamW (12 B/p) | 47.2 | 0 | ❌ |
| ZeRO-3 + 8-bit AdamW (6 B/p) | 24.8 | 0 | ❌ **1.8 GiB short** |
| **ZeRO-3 + CPU optimizer offload** | **17.3** | 59.8 | ✅ |
| **ZeRO-3 + CPU optimizer offload + 8-bit AdamW** | **17.3** | **15.0** | ✅ |
| ZeRO-3 + CPU parameter *and* optimizer offload | 9.8 | 89.7 | ✅ |

Note what the 8-bit optimizer actually buys in the winning row: not GPU memory, but **cutting the CPU RAM requirement from 60 GB to 15 GB.**

⚠️ **CPU-optimizer-only offload is a DeepSpeed capability.** FSDP2's `cpu_offload=true` is all-or-nothing — it drags parameters off too. That is the one and only reason DeepSpeed is in this repo's requirements; see [docs/notes/api-2026.md §7](docs/notes/api-2026.md).

**And what fits comfortably** — per-rank GiB across 2× A10, checkpointing on, `s`=2048:

| Model | `N` | ZeRO-2 (10 B/p) | ZeRO-3 (6 B/p) | +8-bit (3 B/p) |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 8.03 B | 54.7 ❌ | 47.2 ❌ | 24.8 ❌ |
| Qwen3-4B | 4.02 B | 28.3 ❌ | 24.5 ❌ | **13.3 ✅** |
| Llama-3.2-3B | 3.21 B | **23.0 ✅** | **20.0 ✅** | 11.0 ✅ |
| SmolLM3-3B | 3.08 B | **22.0 ✅** | **19.2 ✅** | 10.6 ✅ |
| Qwen3-1.7B | 2.03 B | **15.1 ✅** | **13.2 ✅** | 7.5 ✅ |

**So the 3B class is the largest model you can comfortably full-fine-tune** — meaning plain FSDP2/ZeRO-2, standard optimizer, no offload, room for a real batch. Qwen3-1.7B at 13.2 GiB even fits **one** card with an 8-bit optimizer, which makes it the right size for learning full-FT mechanics fast.

**Vocabulary mapping**, because two names for one thing causes real confusion: FSDP `NO_SHARD` ≡ DDP · `SHARD_GRAD_OP` ≡ ZeRO-2 · `FULL_SHARD` ≡ ZeRO-3 · `HYBRID_SHARD` ≡ ZeRO++ Stage 3.

| | Value |
|---|---|
| Predicted | PCIe ~22 GB/s; LoRA 7.6 ms/step; full-FT-8B ZeRO-3 ~1.1 s/step |
| **Measured** | _(Phase 00 all-reduce benchmark; Phase 10 bake-off)_ |

---

## §7 — Effective batch size

```
effective_batch = micro_batch × grad_accum_steps × world_size
tokens_per_step = effective_batch × avg_sequence_length
```

**The thing to actually track is `tokens_per_step`, not `effective_batch`.** When example lengths vary — and in instruction data they vary enormously — two runs with identical `effective_batch` can see 3× different amounts of text per step. Log tokens/step, and compare runs on tokens seen, not steps taken.

Gradient accumulation trades wall-clock for memory: it gives you a large effective batch at the activation cost of the small micro-batch (§2). It is the main lever you have on a 24 GB card.

---

## §8 — KV cache, and what GQA is worth

For inference, memory goes to the KV cache, not the optimizer:

```
kv_bytes_per_token = 2 × L × kv_heads × head_dim × bytes_per_element
```

(The 2 is one tensor for K and one for V.)

**Worked example — Llama-3.1-8B in bf16:**

```
2 × 32 × 8 × 128 × 2 B = 131 072 B = 128 KiB per token
```

**Concurrency from free VRAM:**

```
free after weights on one 24 GB card ≈ 22.0 − 16.1 ≈ 5.9 GB   (bf16 weights)
tokens of KV = 5.9e9 / 131072 ≈ 45 000 tokens
at 4096 context → ~11 concurrent requests
```

Serve the **4-bit** model instead and weights drop to ~4.5 GB, leaving ~17.5 GB → ~133 000 tokens of KV → **~32 concurrent requests at 4k context**. That is the real argument for serving quantized: not latency, *concurrency*.

**What GQA buys, in one line.** Llama-3.1-8B has 32 query heads but only 8 KV heads. Without GQA (32 KV heads) the cache would be `2 × 32 × 32 × 128 × 2 = 512 KiB/token` — **4× larger**, cutting your concurrency by 4×. That is GQA's entire value proposition, computed on your hardware.

---

## §9 — Statistical power of an evaluation

The derivation that prevents a whole class of self-deception in Phases 06–15.

For a proportion `p` measured on `n` items, the 95% confidence half-width is:

```
half_width ≈ 1.96 × √( p(1-p) / n )
```

At the worst case `p = 0.5`:

| `n` | 95% CI half-width |
|---:|---:|
| 50 | ±13.9 pts |
| 100 | ±9.8 pts |
| 400 | ±4.9 pts |
| 1000 | ±3.1 pts |
| 2000 | ±2.2 pts |

**Therefore: a 100-item eval cannot see anything smaller than about a 10-point difference.** To claim a 3-point win you need roughly 1000 items.

Write this on a sticky note. The most common way people fool themselves in fine-tuning is reporting a 4-point improvement on a 100-example eval set — which is indistinguishable from noise.

(For paired comparisons — same prompts, two models — you can do better than this, because the paired test removes prompt difficulty as a variance source. Phase 05 covers the paired case.)

---

## §10 — MinHash-LSH band/row selection

For near-duplicate detection, the probability that two documents with Jaccard similarity `s` become candidates under LSH with `b` bands of `r` rows each is:

```
P(candidate) = 1 − (1 − s^r)^b        where num_perm = b × r
```

This is an S-curve; you choose `(b, r)` to place its steep part at your similarity threshold. The approximate inflection point is:

```
s* ≈ (1/b)^(1/r)
```

**Worked example — `num_perm = 128`, target threshold `s* = 0.85`:**

Try `r = 16, b = 8`: `s* ≈ (1/8)^(1/16) = 0.8776`. Close.
Try `r = 13, b = 9` (117 perms): `s* ≈ (1/9)^(1/13) = 0.8455`.

Check the curve at `r=16, b=8`:

| `s` | `P(candidate)` |
|---:|---:|
| 0.70 | 0.024 |
| 0.80 | 0.222 |
| 0.85 | 0.549 |
| 0.90 | 0.891 |
| 0.95 | 0.998 |

So this configuration catches ~89% of 0.90-similar pairs and only 2% of 0.70-similar ones. **Derive this rather than copying `num_perm=128` from a tutorial** — the right threshold depends on whether you are removing verbatim reposts or paraphrases, and those want very different curves.

---

## §11 — Roofline: why decode and training want opposite optimizations

```
ridge_point = peak_FLOPS / memory_bandwidth
```

**A10:** 125 TFLOPS bf16 ÷ 600 GB/s ≈ **208 FLOP/byte.**

An operation with arithmetic intensity above 208 is compute-bound; below it, memory-bound.

- **Training** does large matmuls with high arithmetic intensity → sits to the *right* of the ridge → **compute-bound** → optimize FLOPs (better kernels, higher MFU, fused ops).
- **Autoregressive decode** loads the entire weight matrix to compute a single token's output → arithmetic intensity ≈ 1–2 → far to the *left* → **memory-bandwidth-bound** → optimize bytes moved (quantization, batching, speculative decoding).

This is why quantization barely speeds up training but substantially speeds up serving, and why your optimization priorities invert between Phase 10 and Phase 16.

---

## §12 — Checkpoint storage arithmetic

```
adapter only       = Ψ × 2 B
full weights       = N × 2 B
resumable full FT  = N × 2 (weights) + N × 12 (fp32 AdamW states) = N × 14
```

**Worked example — Llama-3.1-8B:**

| Artifact | Size |
|---|---:|
| LoRA adapter (r=16, bf16) | **84 MB** |
| Merged weights, bf16 | 16.1 GB |
| Merged weights, 4-bit | ~4.5 GB |
| Fully resumable full-FT checkpoint | ~112 GB |

At `save_total_limit=3`, adapter-only checkpointing costs 250 MB and full-FT checkpointing costs 336 GB. This is the disk-budget argument for `save_total_limit`, and a second argument for LoRA that has nothing to do with VRAM.

---

## Derivation checklist

Mark these ✅ when you can do them cold on a whiteboard, no notes. This is the `📐` tag in [question_answer.md](question_answer.md).

| # | Derivation | Status |
|---|---|:---:|
| 1 | VRAM for a full fine-tune of an `N`-param model, and where all 16 bytes go | ⬜ |
| 2 | Activation memory with and without gradient checkpointing | ⬜ |
| 3 | Why the fp32 logits tensor can exceed the model, and what fixes it | ⬜ |
| 4 | Tokens/sec from peak TFLOPS, MFU and `6N` — then epoch time | ⬜ |
| 5 | LoRA trainable-parameter count for a given rank and target set | ⬜ |
| 6 | DDP all-reduce volume, and why that makes ZeRO-3 a bad trade on PCIe | ⬜ |
| 7 | Per-GPU memory for DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 | ⬜ |
| 8 | KV cache per token, concurrency from free VRAM, and what GQA saves | ⬜ |
| 9 | Eval items needed to detect an `x`-point difference | ⬜ |
| 10 | MinHash-LSH `(b, r)` for a target Jaccard threshold | ⬜ |
| 11 | A10 ridge point, and why decode is memory-bound | ⬜ |
| 12 | Checkpoint size for adapter vs merged vs resumable | ⬜ |
