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
| GPU model | NVIDIA A10 × 2 | |
| VRAM per card | 24 GB (GDDR6) | |
| Compute capability | 8.6 (Ampere, GA102) | |
| Driver version | — | |
| CUDA runtime (torch) | — | |
| Interconnect | PCIe, **no NVLink** | |
| TDP per card | 150 W | |
| CPU / cores | — | |
| System RAM | — | |
| Disk free (for `HF_HOME` + checkpoints) | ≥ 500 GB recommended | |

**Why TDP matters here.** The A10 is a 150 W single-slot datacenter card, not a 350 W A100. It throttles under sustained load, so a long training run's steady-state clock is meaningfully below its boost clock. Record clocks *during* a run, not at idle.

---

## 2. The capability matrix — SM 8.6

This table is the reason most fine-tuning tutorials are subtly wrong for your box. Reference it every time you reject a config.

| Feature | Available on SM 8.6? | Note |
|---|:---:|---|
| **bf16** | ✅ | Use it. No loss scaling needed, unlike fp16 |
| **fp16** | ✅ | Works, but needs loss scaling — prefer bf16 |
| **TF32** | ✅ | On by default for matmul in recent torch; ~free speedup |
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

### 3a. Achieved matmul throughput

| Shape (bf16) | Theoretical peak | **MEASURED TFLOPS** | % of peak |
|---|---|---|---|
| 4096 × 4096 × 4096 | 125 TFLOPS | | |
| 8192 × 8192 × 8192 | 125 TFLOPS | | |
| 2048 × 2048 × 2048 | 125 TFLOPS | | |

Expect small shapes to fall well short of peak — that gap *is* the reason micro-batch size affects MFU.

### 3b. Memory bandwidth

| Test | Theoretical | **MEASURED** |
|---|---|---|
| Large device-to-device copy | 600 GB/s | |

### 3c. All-reduce bandwidth over PCIe — **the number that decides Phase 10**

| Payload | **MEASURED effective GB/s** | Time |
|---|---|---|
| 4 MB | | |
| 84 MB (= one r=16 LoRA adapter on 8B) | | |
| 1 GB | | |

Expect roughly **15–22 GB/s** effective, versus ~300 GB/s on an NVLinked pair. Small payloads will be latency-dominated and show much lower effective bandwidth — that is expected and it matters, because LoRA gradients *are* a small payload.

Once you have this number, plug it into [NUMBERS.md §6](../NUMBERS.md) and re-derive whether DDP or ZeRO-3 wins for each regime. **Do not skip that step** — it is the most transferable thing you will learn from this hardware.

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
