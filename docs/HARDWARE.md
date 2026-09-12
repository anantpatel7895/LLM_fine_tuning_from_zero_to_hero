# HARDWARE.md — what 2× A10 can and cannot do

**Status: awaiting Phase 00 measurements.** Every `MEASURED` cell below is empty on purpose. Fill each one from a real reading on the box — never from a spec sheet, never from a blog post. Rule 2 of [PLAN.md](../PLAN.md): predict, measure, explain the gap.

Run these two commands on the box and paste the output into the sections below:

```bash
./scripts/00_hardware_report.sh
python -m llm_finetuning_from_zero_to_hero.00_gpu_benchmark
```

---

## 1. The box

| Property | Expected | **MEASURED** |
|---|---|---|
| GPU model | NVIDIA A10 × 2 | ✅ **NVIDIA A10 × 2** |
| VRAM per card | 24 GB (GDDR6) | **23028 MiB = 22.49 GiB usable** — see note below |
| Compute capability | 8.6 (Ampere, GA102) | ✅ **8.6** |
| Driver version | — | **580.178.04** |
| CUDA runtime (torch) | — | ⚠️ torch not importable yet — see §0 |
| Interconnect | PCIe, **no NVLink** | ✅ **`PHB`** — traverses the PCIe host bridge (the CPU). Both cards on NUMA node 0, affinity 0-59 |
| TDP per card | 150 W | ✅ **150.00 W**, max SM clock 1695 MHz |

**On "24 GB":** `nvidia-smi` reports **23028 MiB**, not 24576. The card is marketed as 24 GB but ~1.5 GiB is reserved for ECC and firmware before you allocate anything. Then the CUDA context costs another ~300–600 MiB per process. So your real working budget is **≈21.9 GiB per card**, which is what `NUMBERS.md` §1 means by "24 GB is never 24 GB." Budget against 21.9, not 24.

**On `PHB` specifically:** this is the *second-best* of the non-NVLink options. `PIX` (single bridge) would be faster; `SYS` (crossing NUMA nodes) would be slower. Both cards sharing NUMA node 0 means all-reduce traffic does not cross the inter-socket link — one fewer penalty on a box that already has no NVLink.
| CPU / cores | — | **Intel Xeon Platinum 8358 @ 2.60 GHz, 60 cores** |
| System RAM | — | **472 GiB** (417 GiB free at idle) |
| Disk free (for `HF_HOME` + checkpoints) | ≥ 500 GB recommended | **Two disks** — see below |

### Storage: two volumes, and `HF_HOME` must point at the second

| Mount | Device | Size | Used | Free | Use for |
|---|---|---|---|---|---|
| `/` | `/dev/sda1` | 98 G | 93 G | **5.0 G (95%)** | ⚠️ Nearly full. Do **not** put models or checkpoints here |
| `~/slm-models` | `/dev/sdb` | 197 G | 149 G | **40 G (80%)** | ✅ This is the data disk. `HF_HOME` belongs here |

`HF_HOME` is currently **unset**, which means downloads default to `~/.cache/huggingface` on the 5 GB root partition. That must be changed before any model is fetched.

**What is on the data disk** (`~/slm-models/base_models/`):

| Model | Size | Relevant to this curriculum? |
|---|---|---|
| `gemma-4-26B-A4B-it-qat-q4_0-unquantized` | 49 G | ❌ Another project's serving artifact |
| `gemma-4-26B-A4B-it-W8A16-compressed` | 29 G | ❌ ditto |
| `gemma_4_26b_a4b_awq` | 17 G | ❌ ditto |
| `gemma-4-26B-A4B-it-W4A16-compressed` | 17 G | ❌ ditto |
| `llama_3_1_8b_instruct` | 15 G | ✅ **The headline model — already here** |
| `Mistral-7B-Instruct-v0.3` | 14 G | ✅ **The contrast model — already here** |

The four gemma-4-26B quantization variants total **112 GB**. Reclaiming them would triple the free space. They are inference-serving artifacts (W8A16/W4A16/AWQ/QAT are quantization formats, not training formats) belonging to separate work — confirm ownership before touching.

**The win:** Llama-3.1-8B-Instruct and Mistral-7B-v0.3 are already downloaded, so Phases 02–04 need no large download at all.

### GPU contention — the box is NOT idle

Measured 2026-09-11, `nvidia-smi --query-compute-apps`:

| PID | Process | VRAM |
|---|---|---|
| 52415 | `VLLM::EngineCore` | 18660 MiB |
| 7139 | `.pyenv/versions/aml_sml_env/bin/python3` | 10082 MiB |

**~28.7 GiB of the 45 GiB total is already allocated to other work.** A vLLM server is serving something (almost certainly one of the gemma variants above) and a second Python job is running.

This contradicts the planning assumption that the box is dedicated and free. Until resolved it changes what is possible: roughly 4.4 GiB free on one card and 12.9 GiB on the other. QLoRA on an 8B needs ~6–8 GiB with headroom, so it fits on **one** card at best — and Phase 10's two-GPU work is impossible while both are occupied.

*Measured 2026-09-11 on host `genai-experiment`. GPU rows still blank — the driver is broken, see §0.*

### What the CPU and RAM figures buy us

Both are better than this curriculum assumes, and two later phases get easier as a result:

- **472 GiB RAM** makes DeepSpeed ZeRO-3 **CPU optimizer offload** genuinely viable in Phase 11. An 8B full fine-tune needs ~96 GB of fp32 AdamW optimizer state; that fits in host RAM with enormous room to spare. The constraint becomes PCIe transfer time, not capacity.
- **60 cores** means `dataloader_num_workers` will never bottleneck training, and Phase 07's MinHash-LSH deduplication over a large corpus — normally the slowest data step — parallelizes well.

---

## 0. Blockers found on first run (2026-09-11)

Recorded here because Phase 00 is not complete until all three clear. Diagnosis: `./scripts/00b_diagnose_blockers.sh`.

| # | Blocker | Evidence | Impact |
|---|---|---|---|
| 1 | **NVIDIA driver/library version mismatch** | `Failed to initialize NVML: Driver/library version mismatch. NVML library version: 580.178` | **Total.** No GPU is usable. `nvidia-smi`, torch CUDA, everything |
| 2 | **Root filesystem 92% full** | `/dev/sda1  98G  90G  7.9G  92% /`, and `HF_HOME` unset so models default to `~/.cache/huggingface` on `/` | **Severe.** Llama-3.1-8B in bf16 is ~16 GB — it cannot even download. The curriculum needs roughly 300–500 GB for models, datasets and checkpoints |
| 3 | `pyenv FT_VENV` not created | `pyenv: version 'FT_VENV' is not installed` | Expected — not yet done on the box |

**Do not treat #2 as secondary to #1.** Fixing the driver first only gets you to the point where the first model download fails.

**Why TDP matters here.** The A10 is a 150 W single-slot datacenter card, not a 350 W A100. It throttles under sustained load, so a long training run's steady-state clock is meaningfully below its boost clock. Record clocks *during* a run, not at idle.

---

## 2. The capability matrix — SM 8.6

This table is the reason most fine-tuning tutorials are subtly wrong for your box. Reference it every time you reject a config.

| Feature | Available on SM 8.6? | Note |
|---|:---:|---|
| **bf16** | ✅ | Use it. No loss scaling needed, unlike fp16 |
| **fp16** | ✅ | Works, but needs loss scaling — prefer bf16 |
| **TF32** | ✅ hardware, ❌ **off by default** | **MEASURED 2026-09-11: `TF32 matmul enabled → no`.** My earlier claim that recent torch enables it was wrong — torch has defaulted `torch.backends.cuda.matmul.allow_tf32 = False` since 1.12, for numerical-reproducibility reasons. Enable explicitly if you want it. **But it barely matters here:** TF32 only accelerates **fp32** matmuls, and this curriculum trains in bf16 throughout. It would affect only the stray fp32 ops (optimizer math, norms). Not a missed speedup worth chasing — just do not believe a tutorial that says it is automatic |
| **FP8** | ❌ | No FP8 tensor cores. Needs SM 8.9 (Ada) / 9.0 (Hopper). Rules out FP8 training, and `--kv-cache-dtype fp8` in vLLM is **not a usable Ampere path** — it runs attention in FP8. Keep the bf16 KV cache; save KV memory via GQA and `--max-model-len` instead |
| **`sdpa`** | ✅ **default, and a good one** | PyTorch's own flash kernel. Does not materialize the `s × s` matrix, so the activation formula in `NUMBERS.md` §2 holds |
| **FlashAttention-2** | ✅ | Supported on Ampere. Buys roughly **10–20% over `sdpa`** here — not the 3× the marketing implies. Get it via `attn_implementation="kernels-community/flash-attn2"`, which downloads a prebuilt kernel and skips the 30-minute compile |
| **FlashAttention-3** | ❌ | **Actively disabled on compute capability 8.6** — insufficient shared memory for the implementation. Stronger than "Hopper-optimized": it will not run |
| **FlashAttention-4** | ❌ | Blackwell (SM100+) |
| **NVLink / NVSwitch** | ❌ | PCIe only — this is the constraint that drives the whole Phase 10 analysis |
| **`torch.compile`** | ✅ | Works; watch for recompilation on variable sequence lengths |
| **INT8 / INT4 inference** | ✅ | AWQ / GPTQ / bnb all fine |
| **Multi-Instance GPU (MIG)** | ❌ | A100/H100 feature |

**Three things this rules out immediately:** FP8 anything, FlashAttention-3, and any tutorial whose speed claims assume NVLink.

---

## 3. Measured microbenchmarks

These become your "is the box healthy?" regression baseline. Re-run them if a training run is inexplicably slow.

**Measured 2026-09-11**, GPUs idle, `torch 2.11.0+cu128`, driver 580.178.04.

### 3a. Achieved matmul throughput

| Shape (bf16) | Theoretical peak | **MEASURED TFLOPS** | % of peak | Time |
|---|---|---|---|---|
| 2048 × 2048 × 2048 | 125 TFLOPS | **85.54** | 68.4% | 0.201 ms |
| 4096 × 4096 × 4096 | 125 TFLOPS | **85.98** | 68.8% | 1.599 ms |
| 8192 × 8192 × 8192 | 125 TFLOPS | **79.70** | 63.8% | 13.795 ms |

**~86 TFLOPS sustained, 69% of spec.** That is a good result for a 150 W card, and it is the number to use in throughput predictions — not 125.

Note the shape trend is the *opposite* of the usual expectation: 8192³ is **slower** than 4096³ (63.8% vs 68.8%), not faster. At 8192³ the working set no longer fits in L2, so the kernel becomes bandwidth-starved; the largest shape is not the most efficient one. That is a real effect on this card and it matters when picking micro-batch and sequence length in Phase 04.

⚠️ **Idle-clock caveat:** these were measured on a cold, idle card at 28 °C. A 150 W A10 throttles under a sustained 3-hour run. Verify with `nvidia-smi dmon` **during** training, not before it, and expect the real figure to be lower.

### 3b. Memory bandwidth

| Test | Theoretical | **MEASURED** | % of peak |
|---|---|---|---|
| 2 GiB device-to-device copy | 600 GB/s | **482.5 GB/s** | 80.4% |

80% of spec is normal and healthy — achievable bandwidth is typically 75–85% of the marketing figure.

**Measured ridge point = 86 TFLOPS ÷ 482.5 GB/s ≈ 178 FLOP/byte** (spec-vs-spec gives 208).

The benchmark script printed **259**, which is wrong — it divided *spec* FLOPS by *measured* bandwidth, mixing the two. Use **178**: both terms must come from the same source or the ratio is meaningless. Either way the conclusion in `NUMBERS.md` §11 is untouched — training sits far to the right of it, decode far to the left.

### 3b-2. Usable memory per card — measured, not assumed

| Quantity | **MEASURED** |
|---|---|
| `nvidia-smi` total | 23028 MiB (22.49 GiB) |
| torch nominal total | 22.06 GiB |
| Free before any allocation | 21.79 GiB |
| **Largest sum successfully allocated** | **21.75 GiB** (98.6% of nominal) |
| SMs per card | 72 |

**Budget 21.75 GiB per card.** Not 24, not 23. The gap from 22.49 → 21.79 GiB is **~277 MiB of CUDA context**, which is the real answer to `question_answer.md` Q00.3 consumer 2.

⚠️ The script reports "CUDA context cost 2.0 MiB", which is **mismeasured** — it captures only the incremental allocation after the context already existed. The true cost is the nominal-minus-free gap above. Fix noted in the script.

Expect a few hundred MiB less again once NCCL initialises for multi-GPU.

### 3c. All-reduce bandwidth over PCIe — **the number that decides Phase 10**

**Measured 2026-09-11, `world_size=2` under `torchrun`, NCCL over `PHB`:**

| Payload | **MEASURED effective GB/s** | Time |
|---|---|---|
| 4 MiB | **12.24 GB/s** | 0.343 ms |
| 84 MiB (= one r=16 LoRA adapter on 8B) | **14.86 GB/s** | 5.929 ms |
| 1 GiB | **15.16 GB/s** | 70.848 ms |

**Verdict: ~15.2 GB/s asymptotic.** Prediction was 15–22 GB/s, so reality landed at the **bottom** of the predicted band — the prediction held, but only just. For reference an NVLinked pair does ~300 GB/s, so **this interconnect is ~20× slower** than what most multi-GPU tutorials silently assume.

Note bandwidth *rising* with payload (12.24 → 14.86 → 15.16). Small transfers are latency-dominated, so they never reach asymptotic bandwidth. This matters directly: LoRA gradients are an 84 MiB payload, i.e. still slightly below the plateau.

### What this decides — re-derived from the measured 15.2 GB/s

| Regime | Gradient payload | Comms time/step | Verdict |
|---|---|---|---|
| **DDP + LoRA r=16 on 8B** | 84 MiB (42 M params × 2 B) | **~5.9 ms** (measured directly) | **Negligible** against a ~1 s step. DDP is unambiguously correct |
| **DDP + full FT of 8B** | 16.1 GiB (8.03 B params × 2 B) | **~1.06 s** | **Dominates.** Plus ZeRO-3's parameter all-gathers on top |

⚠️ **The script's derived lines say 168 MB → 11.1 ms and 32.1 GB → 2.12 s — exactly 2× the figures above.** It applied the textbook `2Ψ` all-reduce heuristic. For a **ring all-reduce the bus volume per rank is `2(N-1)/N × S`, which at `N=2` is exactly `S`, not `2S`** — so the `2×` is wrong for a two-GPU box. The measured 84 MiB row (5.929 ms) is ground truth and confirms it: no doubling.

Trust the measured row. The `2Ψ` form is right for large `N` (it approaches `2S`), which is why the heuristic is everywhere — it just does not apply to a pair.

**The conclusion is unchanged either way**, which is itself the useful part: 5.9 ms and 11.1 ms are both negligible against a 1-second step, and 1.06 s and 2.12 s both dominate it. The decision was robust to a 2× error in the derivation.

> **→ Treat GPU 1 as a second *experiment slot*, not a second *worker*,** unless the arithmetic for a specific job says otherwise. Confirm in Phase 10.

### 3d. Baseline generation throughput

| Model | Setting | **MEASURED tok/s** |
|---|---|---|
| `Qwen3-0.6B` | bf16, batch 1, greedy | |
| `Llama-3.2-1B` | bf16, batch 1, greedy | |

---

## 4. What fits — predicted, then confirmed

Predictions come from [NUMBERS.md §1](../NUMBERS.md). Confirm each with `torch.cuda.max_memory_allocated()` as you reach the relevant phase.

All figures are **per GPU** against ~23 GiB usable, at micro-batch 1 with gradient checkpointing on unless stated. `GC` = gradient checkpointing.

| Configuration | Predicted peak | **MEASURED** | Phase | Verdict |
|---|---|---|---|---|
| QLoRA, Mistral-7B-v0.3, `s`=2048, GC on | **7.9 GiB** | | 04 | ✅ roomiest 7B config — small vocab |
| QLoRA, Llama-3.1-8B, `s`=2048, GC on | **10.5 GiB** | | 04 | ✅ the golden-thread default |
| QLoRA, Llama-3.1-8B, `s`=2048, GC **off** | 19.9 GiB | | 04 | ✅ but no headroom — run it once to feel the difference |
| QLoRA, Llama-3.1-8B, `s`=4096, GC on | **11.3 GiB** | | 04 | ✅ |
| QLoRA, Llama-3.1-8B, `s`=4096, GC **off** | 30.1 GiB | | 04 | ❌ **OOM.** The single most instructive experiment here |
| QLoRA, Llama-3.1-8B, `s`=2048, GC on, micro-batch 8 | ~16.3 GiB | | 06 | ✅ prefer this over grad accumulation |
| LoRA bf16, Llama-3.1-8B, `s`=2048, GC on | **20.1 GiB** | | 06 | ⚠️ fits at mbs=1 only. One long example OOMs it |
| LoRA bf16, Qwen3-4B, `s`=2048, GC on | ~12 GiB | | 06 | ✅ where bf16 LoRA actually belongs |
| Full FT, Qwen3-1.7B, 8-bit optimizer, **1 GPU** | **13.2 GiB** | | 11 | ✅ full FT on one card |
| Full FT, SmolLM3-3B, FSDP2 pure-bf16, 2 GPUs | **19.2 GiB** | | 11 | ✅ the comfort ceiling |
| Full FT, Qwen3-4B, ZeRO-3 + 8-bit optimizer | 13.3 GiB | | 11 | ✅ tight |
| Full FT, Llama-3.1-8B, ZeRO-3, no offload | **47.2 GiB/rank** | | 11 | ❌ **2× over budget** |
| Full FT, Llama-3.1-8B, ZeRO-3 + CPU optim offload + 8-bit | **17.3 GiB/rank** + 15 GiB CPU RAM | | 11 | ✅ but ~4× slower — the "here is the wall" demo |
| GRPO, Qwen3-1.7B, trainer GPU 0 + vLLM GPU 1 | ~20 / ~20 GiB | | 15 | |
| vLLM serving, Llama-3.1-8B bf16 | 16.1 GiB weights + KV | | 16 | ~11 concurrent @ 4k ctx |
| vLLM serving, Llama-3.1-8B AWQ W4A16 | ~4.5 GiB weights + KV | | 16 | ~32 concurrent @ 4k ctx |

> **The two rows to internalize:** gradient checkpointing turns an OOM into 11.3 GiB at sequence 4096, and bf16 LoRA on an 8B is a 20 GiB configuration with no room to breathe. Derivations in [NUMBERS.md §1–§3](../NUMBERS.md).

---

## 5. Environment on the box

| Item | Value | **MEASURED / confirmed** |
|---|---|---|
| Python | 3.12.x via pyenv, virtualenv `FT_VENV` | |
| torch | pinned in `requirements-gpu.txt` | |
| CUDA (torch build) | | |
| `HF_HOME` | on the large disk, **not** `~` | |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | |
| flash-attn | built and importable | |
| bitsandbytes | GPU support confirmed | |

---

## 6. Observed anomalies

Anything surprising goes here with the date, so you can tell "the box changed" from "my code changed."

| Date | Observation | Resolution |
|---|---|---|
| | | |
