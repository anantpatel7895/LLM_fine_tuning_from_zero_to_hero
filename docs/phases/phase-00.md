# Phase 00 — Environment & hardware baseline

**~6 hrs** · Prerequisite: — · Next: [Phase 01](phase-01.md)

> **Status:** 🟡 In Progress — hardware measured and recorded (2026-09-11); `docs/HARDWARE.md` §1–3 complete. Remaining: write your own predictions into §2 below and close the Q&A.

## 0. Learning goal

After this phase you can **derive what will fit on a 24 GB card before touching a trainer, and explain the gap between your derivation and what the box actually does.**

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** [docs/HARDWARE.md](../HARDWARE.md) with every `MEASURED` cell filled in from a real reading; a working Mac→box sync; `NUMBERS.md` §1–§6 confirmed or corrected against measurement.

**Models:** `Qwen3-0.6B` (smoke only — nothing is trained in this phase)

**Data:** none

---

## 1. Concept

### Why this phase exists at all

Most fine-tuning tutorials were written on an A100 or an H100. Yours is an A10 — one third the memory, one third the bandwidth, no NVLink, and a different compute capability that removes several features those tutorials assume. If you skip this phase, you will spend Phases 04–11 confused about why numbers do not match, and you will accept configurations that are wrong for your hardware.

The deeper reason: **prediction is the core fine-tuning skill.** "How much VRAM does an 8B full fine-tune need?" is a real interview question with a real answer, and being able to answer it means you can plan work instead of discovering by OOM. This phase is where you learn to predict, and where you calibrate your predictions against the truth.

### What "compute capability 8.6" means

Every NVIDIA GPU reports a compute capability — a `(major, minor)` version identifying its architecture's feature set. Yours reports `(8, 6)`: Ampere, GA102 silicon.

```python
import torch
torch.cuda.get_device_capability()   # (8, 6)
```

That single number determines what you can and cannot run. The full matrix lives in [HARDWARE.md](../HARDWARE.md), but the three that will bite you:

- **FP8 does not exist on 8.6.** It needs 8.9 (Ada) or 9.0 (Hopper). So no FP8 training, and no FP8 KV cache in Phase 16.
- **FlashAttention-2 works; FlashAttention-3 does not.** FA3 is Hopper-only.
- **bf16 works, and you should use it over fp16.** Both are 16-bit, but bf16 keeps fp32's exponent range at the cost of mantissa precision. That range is why bf16 needs no loss scaling while fp16 does — fp16's narrow exponent range means small gradients underflow to zero, so fp16 training has to multiply the loss by a large constant before the backward pass and divide it out after. bf16 simply does not have that problem. One less moving part, and one fewer source of NaNs.

### Why there is no NVLink, and why that is the most interesting thing about this box

NVLink is a direct high-bandwidth interconnect between GPUs — around 300 GB/s on an NVLinked A100 pair. Your two A10s talk over **PCIe**, at roughly 15–22 GB/s effective. That is a 15× difference, and it inverts a piece of conventional wisdom.

The conventional wisdom is "more GPUs, use ZeRO-3." The arithmetic on your box says otherwise, and you will derive it in §2 and measure it in Phase 10. The short version: sharding a model across GPUs means shipping parameters back and forth on every forward and backward pass. When the wire is 15× slower, that traffic stops being free.

**This is not a limitation to apologize for. It is the source of the most transferable thing you will learn here** — that parallelism strategy is a function of interconnect bandwidth, and that people who only ever used NVLinked machines do not know this.

### Roofline: why training and serving want opposite optimizations

Every GPU has a **ridge point** — the arithmetic intensity (FLOPs per byte of memory traffic) above which a kernel is limited by compute and below which it is limited by memory bandwidth:

```
ridge_point = peak_FLOPS / memory_bandwidth
A10:          125 TFLOPS / 600 GB/s ≈ 208 FLOP/byte
```

- **Training** does big matmuls with high arithmetic intensity — well above 208 — so it is **compute-bound**. Optimize FLOPs: better kernels, higher MFU, fused operations.
- **Autoregressive decoding** loads an entire weight matrix to produce one token. Arithmetic intensity ≈ 1–2, far below 208, so it is **memory-bandwidth-bound**. Optimize bytes moved: quantization, batching, speculative decoding.

Keep this in your pocket. It is why quantization barely speeds up training but substantially speeds up serving, and it is why your optimization priorities will completely invert between Phase 10 and Phase 16.

### The thermal reality of a 150 W card

The A10 is a single-slot, 150 W datacenter card — not a 350 W A100. Under sustained load it throttles, so a long run's steady-state clock is meaningfully below its boost clock. This matters because you will measure throughput in a 60-second benchmark and then get less of it in a 3-hour run.

Record clocks **during** a run, not at idle: `nvidia-smi dmon`.

---

## 2. Numbers

This phase's job is to fill in [NUMBERS.md](../../NUMBERS.md) §1–§6 and check each formula against reality.

### Predict first — do this before running the benchmark

Write your answers here from the formulas, *then* run `00_gpu_benchmark.py`.

| Quantity | Formula | **Your prediction** | Measured | Gap — and why |
|---|---|---|---|---|
| VRAM, full FT of Llama-3.1-8B, AdamW | `N × 16 B` | | — | n/a — cannot run |
| VRAM, bf16 LoRA of Llama-3.1-8B | `N × 2 B` + adapter | | Phase 06 | |
| VRAM, QLoRA of Llama-3.1-8B | `N × 0.5 B` + adapter | | Phase 04 | |
| Usable VRAM per card | 24 GB − context − fragmentation | | | |
| Achieved matmul TFLOPS at 8192³ | — | | | |
| Memory bandwidth | — | | | |
| **PCIe all-reduce, 84 MB payload** | — | | | |
| A10 ridge point | `peak_FLOPS / bandwidth` | 208 FLOP/byte | | |
| Tokens/sec, 8B LoRA, 30% MFU | `(peak × MFU) / 5N` | | Phase 04 | |

### The derivation you must be able to do cold

Where do 16 bytes per parameter come from in a mixed-precision AdamW full fine-tune?

| Component | Precision | bytes/param |
|---|---|---:|
| Parameters (compute copy) | bf16 | 2 |
| Gradients | bf16 | 2 |
| fp32 master weights | fp32 | 4 |
| AdamW momentum `m` | fp32 | 4 |
| AdamW variance `v` | fp32 | 4 |
| **Total** | | **16** |

So Llama-3.1-8B full fine-tune = `8.03e9 × 16 = 128.5 GB`. Your box has 48 GB. **The answer is no, and now you know why, and by how much.**

This is `📐` derivation #1 in [NUMBERS.md](../../NUMBERS.md). Do not move to Phase 01 until you can produce that table from memory.

### The communication derivation that decides Phase 10

Once the benchmark gives you a real PCIe number, plug it in:

```
DDP all-reduce volume per step ≈ 2 × Ψ_trainable × bytes_per_param

LoRA r=16 on 8B:  Ψ = 42e6  →  2 × 42e6 × 2 B = 168 MB  →  168e6 / <your GB/s>  = ___ ms
Full FT of 8B:    Ψ = 8.03e9 →  2 × 8.03e9 × 2 B = 32 GB →  32.1e9 / <your GB/s> = ___ s
```

If those come out at roughly 8 ms and 1.6 s respectively, you have just derived — on your own hardware, from your own measurement — that **DDP is free for LoRA and catastrophic for full fine-tuning on this box.** Write that in [HARDWARE.md](../HARDWARE.md) §3c. It is the phase's most valuable output.

---

## 3. Landscape & choice

**pyenv + `requirements.txt`, split into Mac and GPU files.**

The alternative was `uv` + `pyproject.toml`, which is what your newest sibling repo uses and is better on most axes — faster resolution, a real lockfile. Rejected here because `torch` with a CUDA build, `bitsandbytes`, `flash-attn`, `deepspeed` and `vllm` have no usable Mac wheels, and because virtually every fine-tuning recipe, Docker image and community script assumes plain pip. Fighting that on a remote box is friction that teaches nothing about fine-tuning. Revisit if pip resolution gets painful.

The **split** is the load-bearing part: `requirements.txt` keeps the Mac environment installable so you can inspect data and edit configs locally, and `requirements-gpu.txt` holds everything CUDA.

**TensorBoard now, Weights & Biases from Phase 06.** No account, no API key, nothing to configure while you are still learning the plumbing. W&B arrives exactly when Phase 06's eight-run ablation grid makes a comparison UI worth the setup — and its shareable run links then become portfolio evidence. Rejected: W&B from day one (setup friction before there is anything to compare) and TensorBoard forever (genuinely bad at ranking many runs).

**rsync while iterating, git when reporting.** rsync is a faster loop; git guarantees the box is at a known commit, which matters because from Phase 04 onward a run's git SHA is part of its identity. Use both, for different things.

---

## 4. Design

Four things get built:

1. **`scripts/00_hardware_report.sh`** — one command that dumps driver, CUDA, topology, clocks, power and the capability matrix. Run it on the box; paste the output into [HARDWARE.md](../HARDWARE.md) §1–2.
2. **`src/.../00_gpu_benchmark.py`** — matmul TFLOPS at three shapes, memory bandwidth, **PCIe all-reduce at three payload sizes**, and a baseline generation throughput. Fills [HARDWARE.md](../HARDWARE.md) §3.
3. **`scripts/sync_to_gpu.sh`** — rsync Mac → box with an explicit exclude list. The exclusions are the important part.
4. **`scripts/smoke_test.sh`** — the 50-step, sub-1B gate that every later recipe must pass before a real run.

**How you will know it worked:**

- [ ] `00_hardware_report.sh` runs clean on the box and confirms `(8, 6)`, two GPUs, and PCIe (not NVLink) topology
- [ ] The all-reduce benchmark produces a number, and that number plugged into [NUMBERS.md §6](../../NUMBERS.md) reproduces the ~8 ms / ~1.6 s split
- [ ] Every `MEASURED` cell in [HARDWARE.md](../HARDWARE.md) §1–3 is filled
- [ ] `sync_to_gpu.sh --dry-run` shows code and configs moving, and **nothing** from `data/` or `artifacts/`
- [ ] You can produce the 16-bytes-per-parameter table from memory

---

## 5. Build & Q&A

### Run these on the box

```bash
# 1. environment
pyenv activate FT_VENV
pip install -r requirements-gpu.txt

# 2. hardware report
./scripts/00_hardware_report.sh | tee artifacts/hardware_report.txt

# 3. benchmarks  (the all-reduce test needs both GPUs, so it uses torchrun)
python -m llm_finetuning_from_zero_to_hero.00_gpu_benchmark
```

Then paste both outputs into [HARDWARE.md](../HARDWARE.md).

### What was built

_(fill in)_

### Results

_(paste the reports, then fill in the MEASURED columns in §2 above)_

### What surprised you

_(the gap between prediction and measurement — this is the phase's actual lesson)_

### Measured 2026-09-11 — predictions vs reality

| Quantity | Predicted | **Measured** | Gap, and what it taught |
|---|---|---|---|
| All-reduce bandwidth | 15–22 GB/s | **15.2 GB/s** | Landed at the **bottom** of the band. The prediction held, but a band that wide is barely a prediction — tighten it next time |
| Usable VRAM/card | ~21.9 GiB | **21.75 GiB** | Good. "24 GB" is off by 2.25 GiB |
| CUDA context | 300–600 MiB | **~277 MiB** | Slightly better than predicted |
| Matmul throughput | — (spec 125 TFLOPS) | **86 TFLOPS, 69% of spec** | Strong for 150 W. **Use 86, not 125, in every throughput estimate** |
| Memory bandwidth | — (spec 600 GB/s) | **482.5 GB/s, 80%** | Normal; achievable is always 75–85% of spec |
| Ridge point | 208 FLOP/byte (spec) | **178 FLOP/byte** (measured/measured) | Same conclusion — training right of it, decode far left |

**Three things the measurements caught that the derivations got wrong** — the real value of rule 2:

1. **The `2Ψ` all-reduce heuristic overstates a 2-GPU box by exactly 2×.** A ring all-reduce moves `2(N-1)/N × S` per rank, which is `S` at `N=2`, not `2S`. The measured 84 MiB row (5.929 ms) disproved the derived 11.1 ms directly. `2Ψ` is the large-`N` limit — it just does not apply to a pair.
2. **The ridge point was computed from spec FLOPS ÷ measured bandwidth**, giving a meaningless 259. Both terms must come from one source.
3. **The "CUDA context cost" probe measured allocator rounding, not the context** — CUDA was already initialised by the time it ran. The real figure is nominal-minus-free.

All three are fixed in `00_gpu_benchmark.py`. Worth noticing that **the Phase 10 conclusion survived a 2× error** — 5.9 ms and 11.1 ms are both negligible against a 1 s step; 1.06 s and 2.12 s both dominate it. A decision robust to a 2× error in its input is a decision you can trust.

**Unexpected finding:** matmul efficiency *falls* at the largest shape — 8192³ hits 63.8% where 4096³ hits 68.8%. The working set stops fitting in L2 and the kernel goes bandwidth-starved. Bigger is not automatically better, which is worth remembering when choosing micro-batch and sequence length in Phase 04.

### Questions

Seeded in [question_answer.md](../../question_answer.md) as `Q00.1`–`Q00.5`. Answer them from memory at the end of the phase.

### Adversarial close

**What does this now fail on?** Some candidates worth checking: does the benchmark handle a single-GPU box gracefully? What happens if one GPU is already busy? Does the report distinguish "no NVLink" from "NVLink present but not used"?
