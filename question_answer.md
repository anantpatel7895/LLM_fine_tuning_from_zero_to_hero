# question_answer.md — running recall log

**Why this file is the real deliverable.** The adapters and the eval reports are artifacts; this file is the actual skill. A phase is not complete until you can answer its questions **without looking anything up.**

**Numbering:** `Q<phase>.<n>` — e.g. `Q03.2` is the second question from Phase 03.

**Legend**

| Tag | Meaning |
|---|---|
| ⭐ | Asked in interviews |
| 🔧 | Needed during a real incident |
| 📐 | **Must be derivable on a whiteboard, cold, no notes** — see the checklist in [NUMBERS.md](NUMBERS.md) |

The 📐 tag matters more here than in other repos: fine-tuning interviews are disproportionately derivation questions, and the derivations are all in `NUMBERS.md`.

**Workflow:** write the question when it comes up during a phase. Write the answer *from memory* at the end of the phase. If you cannot, the phase is not finished — go back.

---

## Phase 00 — Environment & hardware baseline

> **How to use the answers below.** Cover the answer, say it out loud, *then* read. The
> 📐 ones you should be able to reproduce on a whiteboard with no notes — that is the bar
> fine-tuning interviews actually test.

**Q00.1 📐⭐** — How much VRAM does a full fine-tune of an 8B model need with AdamW mixed precision, and where does every byte go?

> **16 bytes per parameter.** For an 8B model that is **128 GB** — before a single activation.
>
> | Component | Precision | Bytes/param |
> |---|---|---:|
> | Weights (the copy used for forward/backward) | bf16 | 2 |
> | Gradients | bf16 | 2 |
> | Master weights (the authoritative copy the optimizer updates) | fp32 | 4 |
> | Adam first moment `m` (`exp_avg`) | fp32 | 4 |
> | Adam second moment `v` (`exp_avg_sq`) | fp32 | 4 |
> | **Total** | | **16** |
>
> The key insight: **the optimizer, not the model, is the problem.** Weights+gradients are only 4 of the 16 bytes; the fp32 master copy plus Adam's two moments are 12. That is why 8-bit optimizers (→10 bytes/param) and Adafactor (which factorizes `v` instead of storing it per-parameter) exist, and why ZeRO shards optimizer state *first* (ZeRO-1) — it is the biggest single win available.
>
> Why a fp32 master copy at all: bf16 has only 7 mantissa bits, so a tiny update added to a large weight rounds away entirely (`w + lr*g == w`). Training would silently stall. The fp32 copy preserves those small updates.
>
> **On this box:** 128 GB against 45 GB total ⇒ a full fine-tune of an 8B is **impossible**, even with both cards and ZeRO-3 (which gets you to 16/N = 8 bytes/param = 64 GB, still over). It needs ZeRO-3 *plus* CPU optimizer offload — viable here only because the host has 472 GB of RAM. This is exactly why the curriculum full-fine-tunes a 1.7B and uses LoRA for the 8B. See `NUMBERS.md` §1.

**Q00.2 ⭐** — Why does bf16 need no loss scaling while fp16 does?

> Both are 16 bits. They spend those bits differently:
>
> | | Sign | Exponent | Mantissa | Smallest normal value |
> |---|---|---|---|---|
> | fp32 | 1 | 8 | 23 | ~1.2e-38 |
> | fp16 | 1 | **5** | 10 | ~6.1e-5 |
> | bf16 | 1 | **8** | 7 | ~1.2e-38 |
>
> **bf16 keeps fp32's exponent — its full dynamic range — and pays with mantissa bits (precision).** fp16 does the opposite.
>
> Gradients during training are routinely around 1e-7 or smaller. In fp16 those fall below ~6e-5 and **underflow to exactly zero** — the gradient is silently destroyed. Loss scaling works around it: multiply the loss by a large factor (say 1024) before `backward()` so gradients land inside fp16's representable range, then divide them out before the optimizer step. It needs dynamic adjustment and overflow detection, which is extra machinery that can itself fail.
>
> bf16 spans the same exponent range as fp32, so small gradients stay representable and **no scaling is needed at all**. The trade is precision, and empirically range matters far more than precision for training stability.
>
> Your A10 supports bf16 natively (`torch.cuda.is_bf16_supported()` → True on sm_80+). **Always use bf16 here.**

**Q00.3 🔧** — Why is 24 GB of VRAM never actually 24 GB? Name four consumers.

> Measured on this box: `nvidia-smi` reports **23028 MiB**, not 24576 — **1.5 GiB is gone before you allocate anything.**
>
> 1. **Firmware and ECC reserve** — ~1.5 GiB, taken by the card itself. This is the 24576 → 23028 gap.
> 2. **CUDA context** — ~300–600 MiB *per process*, created on first CUDA call. Kernels, the driver's bookkeeping, cuBLAS/cuDNN handles. Multi-process training pays this per rank.
> 3. **Allocator fragmentation** — PyTorch's caching allocator requests large blocks and reuses them, so `memory_reserved()` exceeds `memory_allocated()`. With variable sequence lengths this gap grows, which is why a run can OOM at step 4000 having been fine at step 1. Mitigate with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
> 4. **Library workspaces and comms buffers** — cuBLAS/cuDNN scratch space for matmul and attention kernels, plus **NCCL buffers** for multi-GPU all-reduce.
>
> A fifth that bit you directly: **other people's processes.** Your vLLM server held 18660 MiB while you were trying to plan a training run.
>
> **Practical rule:** budget against **~21.9 GiB usable per card**, not 24. When a prediction says "it just fits in 23 GB", it does not.

**Q00.4 📐** — What is the A10's ridge point, and what does it tell you about optimizing training versus optimizing decode?

> **Ridge point = peak compute ÷ peak bandwidth = 125 TFLOPS ÷ 600 GB/s ≈ 208 FLOP/byte.**
>
> That is the arithmetic intensity at which the card is perfectly balanced. An operation doing *more* than 208 floating-point operations per byte it moves is **compute-bound**; fewer, and it is **memory-bound**. Everything below is just reading off which side you're on.
>
> **Training sits to the right (compute-bound).** Large batched matmuls reuse each loaded weight across many rows of the batch, giving intensity in the thousands. So you optimize **FLOPs delivered**: keep the tensor cores fed with large matmuls, use bf16, pack sequences so you aren't multiplying padding, raise MFU.
>
> **Decode sits far to the left (memory-bound), and it isn't close.** Generating one token at batch size 1 loads *every weight in the model* to perform a single vector-matrix multiply — intensity ≈ 1–2 FLOP/byte, roughly **100× below the ridge point.** The tensor cores idle while the memory bus saturates. So you optimize **bytes moved**: quantize the weights (4-bit means 4× fewer bytes and therefore ~4× faster decode), shrink the KV cache, and batch requests — batching is what raises intensity and drags decode back toward the ridge.
>
> **The payoff:** this is why quantization transforms inference latency but barely helps training throughput, and why the same card needs opposite tuning in Phase 04 and Phase 16. See `NUMBERS.md` §11.

**Q00.5 ⭐** — Your GPU reports compute capability `(8, 6)`. Name three things that rules out.

> `(8, 6)` = Ampere GA102. Ruled out:
>
> 1. **FP8** — no FP8 tensor cores. Needs sm_89 (Ada) or sm_90 (Hopper). So no FP8 training, and vLLM's `--kv-cache-dtype fp8` is not a usable path here.
> 2. **FlashAttention-3** — not merely "Hopper-optimized", it is **actively disabled** on 8.6 for insufficient shared memory per SM. It will not run. FlashAttention-**2** is supported.
> 3. **MIG (Multi-Instance GPU)** — partitioning one physical GPU into isolated instances is A100/H100 only.
>
> Also unavailable, if you want more: Transformer Engine's FP8 paths, and FlashAttention-4 (Blackwell, sm_100+).
>
> **Available and worth using:** bf16, TF32, `sdpa` (already a flash kernel — the transformers default), FlashAttention-2, `torch.compile`, INT8/INT4 inference. See `docs/HARDWARE.md` §2 for the full matrix.

### Questions from what actually went wrong on 2026-09-11

These five came out of real incidents during setup, not from reading. That makes them the
most valuable ones in this section.

**Q00.6 ⭐** — torch 2.11.0 is the newest build on the cu128 index; torch 2.12+ ships CUDA 13 wheels only. Your driver is 580.178.04, which supports CUDA 13. So why install the *older* CUDA 12.8 build anyway? Name the dependency that decides it, and the phase that would break without it.

> **`bitsandbytes` decides it, and Phase 04 breaks without it.**
>
> The driver is genuinely not the constraint — 580.178.04 exceeds the ~580.65 that CUDA 13 requires, so the newer path would load fine. The constraint is the **ecosystem below torch**:
>
> - **`bitsandbytes`** provides NF4 4-bit quantization. That *is* QLoRA. Without it, Phase 04's "first real QLoRA SFT run" cannot happen, and the whole 8B-on-one-card strategy collapses — a bf16 8B LoRA needs ~17 GB where QLoRA needs ~5. Its prebuilt wheels are best-populated for CUDA 12.x.
> - **flash-attn** and the `kernels-community` Hub variants are likewise mostly cu12x. On CUDA 13 you risk a 30-minute source compile, or discovering no sm_86 variant exists.
>
> And the cost of staying on 12.8 is **nothing we care about**: CUDA 12.8 fully supports sm_86, and no phase here needs a CUDA 13 feature. The headline CUDA 13 wins are FP8-related, and **this hardware has no FP8** (see Q00.5).
>
> **The generalizable lesson:** you do not pick the newest CUDA, you pick the newest CUDA that your *hardest* dependency has working binaries for. Chase the version your critical library supports, not the version at the top of the download page.

**Q00.7 🔧⭐** — `pyenv activate FT_VENV` succeeded, `VIRTUAL_ENV` was correct and `python -V` printed 3.12.10. `pip install torch` then reported success, and `import torch` failed with `ModuleNotFoundError`. Explain the exact mechanism, in four steps, and say why `python -m pip` is structurally immune to it.

> **Four facts that compose into the trap:**
>
> 1. **The shell knows nothing about Python.** Typing `pip` makes the shell walk `$PATH` left to right and run the **first** file named `pip` it finds, then stop. That is the entire algorithm.
> 2. **A virtualenv is mostly a `PATH` edit** — it prepends `$VIRTUAL_ENV/bin` and sets `VIRTUAL_ENV`. Prepending cannot beat an entry that was *already* earlier. Here `~/.local/bin` came first and won.
> 3. **`pip` is a small script with a hardcoded interpreter.** Its first line is a shebang naming an absolute Python path. **That shebang — not the shell, not `VIRTUAL_ENV` — decides which `site-packages` is written.**
> 4. **So the halves disagreed.** `pip` (shebang → python3.10) installed to `~/.local/lib/python3.10/site-packages`; `python` (3.12.10) reads `FT_VENV/lib/python3.12/site-packages`, which was empty.
>
> Neither command malfunctioned. They were correctly told *different things*.
>
> **Why `python -m pip` cannot fail this way:** you run `python` — resolved once, correctly — and instruct *that interpreter* to load its own bundled `pip` module from its own `sys.path` and execute it. There is no second executable, therefore **no shebang and no second `PATH` lookup**. The interpreter installs for itself. It is structurally incapable of targeting a different Python.
>
> **Why it cost two attempts:** the failure is silent and *impersonates the opposite problem*. pip said "Successfully installed", the prompt said `(FT_VENV)`, `VIRTUAL_ENV` and `python -V` were both right. The only symptom, `ModuleNotFoundError`, reads as "I forgot to install it" — so the rational next move is to install again, which succeeds again, into the wrong place again.
>
> **Generalizes to every console script**, because all are shebang scripts: `pytest`, `jupyter`, `huggingface-cli`, and critically **`accelerate`** and **`torchrun`**. A shadowed `accelerate launch` in Phase 10 would run your multi-GPU job under a different interpreter — and present as a distributed-training bug, not an environment one. See `docs/LESSONS.md` **L1**.

**Q00.8 🔧** — What single line in pip's output is proof that you are *not* in the virtualenv you think you are, and why does it prove that?

> ```
> Defaulting to user installation because normal site-packages is not writeable
> ```
>
> **Why it is proof:** pip falls back to a `--user` install only when it cannot write to its target `site-packages`. An active virtualenv's `site-packages` lives in your own directory and is **always writable by you**. So the message is self-contradictory with being in a venv — if pip were the venv's pip, the directory would be writable and this line could not appear.
>
> It means pip's target is some *system* Python's `site-packages` (root-owned, hence unwritable), i.e. **pip is not the venv's pip**.
>
> This line appeared in the very first install on this box and was the earliest available evidence — before any `ModuleNotFoundError`. Reading it would have saved the second attempt. **Treat it as an error, not a notice.**

**Q00.9 📐⭐** — `nvidia-smi topo -m` reports `PHB` between your two cards. What does `PHB` mean physically? Rank it against `PIX`, `NODE`, `SYS` and `NV#`. Which phase's central decision does this one word determine, and what does it decide?

> **`PHB` = "PCIe Host Bridge":** GPU-to-GPU traffic travels up the PCIe tree to the **CPU's root complex** and back down to the other card. There is no direct card-to-card link.
>
> **Ranked best to worst:**
>
> | Code | Path | Rough bandwidth |
> |---|---|---|
> | `NV#` | Bonded NVLinks, card to card directly | 300+ GB/s |
> | `PIX` | At most a single PCIe bridge | ~20–25 GB/s |
> | `PXB` | Multiple PCIe bridges, not reaching the host bridge | ~15–20 GB/s |
> | **`PHB`** | **Up through the CPU's PCIe host bridge** ← **this box** | **~15–22 GB/s** |
> | `NODE` | Between PCIe host bridges inside one NUMA node | slower |
> | `SYS` | Across NUMA nodes, over the socket link (QPI/UPI) | worst |
>
> So this is the middle of the non-NVLink options — and better than it could be, because both cards sit on **NUMA node 0**, so traffic never crosses the inter-socket link. `SYS` would have been worse.
>
> **It determines Phase 10 (multi-GPU strategy),** and the derivation is short. DDP all-reduces roughly `2 × trainable_bytes` per step:
>
> - **LoRA r=16 on 8B:** 42 M trainable params × 2 bytes = **84 MB** → ~**8 ms/step** over ~20 GB/s. Against a ~1 s step, negligible. **DDP is unambiguously correct.**
> - **Full fine-tune of 8B:** 16 GB of gradients → ~**1.6 s/step of pure communication**, plus ZeRO-3's parameter all-gathers on top. Communication *dominates* compute.
>
> **The conclusion an expert draws:** on a PCIe pair with no NVLink, ZeRO-3 for a large model is often slower than simply **running two independent LoRA experiments in parallel** — treating the second GPU as a second *experiment slot* rather than a second *worker*. That judgment falls directly out of one word in `nvidia-smi topo -m`. See `NUMBERS.md` §6.

**Q00.10 🔧** — `Failed to initialize NVML: Driver/library version mismatch`. What are the two halves of an NVIDIA driver install, why can they diverge without anyone touching the machine, and what are the *two different* fixes depending on which is stale?

> **The two halves:**
>
> 1. **The kernel module** (`nvidia.ko`) — loaded into the running kernel, talks to the hardware.
> 2. **The userspace libraries** (`libnvidia-ml.so` = NVML, `libcuda.so`) — what `nvidia-smi` and torch link against.
>
> They must be the **same version**. NVML refuses to initialize on mismatch rather than risk undefined behaviour.
>
> **How they diverge with nobody touching anything:** a package update — often `unattended-upgrades` — replaces the userspace libraries **on disk immediately**, but the kernel module already loaded **keeps running until it is unloaded**. You now have new libraries talking to an old module. Nothing visibly happened; the next `nvidia-smi` just fails. Your error said `NVML library version: 580.178`, i.e. userspace was at 580.178 while the loaded module was something else.
>
> **Two fixes, and picking wrong wastes an hour:**
>
> | Situation | Fix |
> |---|---|
> | New module **is on disk**, old one still loaded | **Reload it.** Reboot, or `rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && modprobe nvidia` — which requires **every GPU process stopped first** |
> | New module was **never built** (DKMS failed against the running kernel) | **A reboot will not help.** Rebuild/reinstall the driver, then reboot |
>
> **How to tell them apart** — the one comparison that settles it:
>
> ```bash
> cat /proc/driver/nvidia/version      # what is LOADED
> modinfo -F version nvidia            # what is ON DISK
> dkms status                          # did it build for this kernel?
> ```
>
> This is what `scripts/00b_diagnose_blockers.sh` exists to check.

**Q00.11 🔧** — A tool reports a package as "NOT INSTALLED". Why is that answer sometimes actively harmful, and what should a diagnostic check *before* attempting the import?

> **Because "absent" and "present but unimportable" need opposite fixes**, and collapsing them sends you down the wrong path confidently.
>
> - **Absent** → install it.
> - **Present but broken** → installing again is useless. It will "succeed" and stay broken. The real causes are a truncated download (classically a disk that filled mid-install), mismatched CUDA wheels (cu13 libraries under a cu12 torch), a conflicting version pin dragged in by another package, or — as here — a perfectly fine install belonging to a *different interpreter*.
>
> That is precisely the loop you hit: the report said "NOT INSTALLED", so reinstalling was the obvious move, and it succeeded twice while changing nothing.
>
> **What to check first: distribution metadata, not the import.**
>
> ```python
> import importlib.metadata as md
> try:
>     version = md.version("torch")     # reads the .dist-info directory
> except md.PackageNotFoundError:
>     version = None                    # genuinely absent -> pip install
> ```
>
> If metadata exists, the package **is** installed for this interpreter, so report `INSTALLED (2.11.0) but IMPORT FAILED` **and print the exception text** — that message names the actual fault.
>
> **One subtlety worth keeping:** catch `Exception`, not `ImportError`. A torch whose CUDA libraries are mismatched raises `OSError` from the dynamic loader, and an `except ImportError` would let it escape as a raw traceback. `scripts/00_hardware_report.sh` was rewritten after this incident to do exactly this.

---

## Phase 01 — Bridge: your GPT-2 loop → the HF stack

_(add during the phase)_

---

## Phase 02 — Tokenizers, chat templates & special tokens

_(add during the phase)_

---

## Phase 03 — Loss masking & the anatomy of an example

_(add during the phase)_

---

## Phase 04 — First real QLoRA SFT run

_(add during the phase)_

---

## Phase 05 — Evaluation before optimization

_(add during the phase)_

---

## Phase 06 — LoRA/QLoRA mechanics & ablations

_(add during the phase)_

---

## Phase 07 — Data prep I: sourcing, quality, dedup, decontamination

_(add during the phase)_

---

## Phase 08 — Data prep II: synthetic generation & curation

_(add during the phase)_

---

## Phase 09 — Data prep III: mixing, ordering, how much is enough

_(add during the phase)_

---

## Phase 10 — Multi-GPU on 2×A10 without NVLink

_(add during the phase)_

---

## Phase 11 — Full fine-tuning & continued pre-training

_(add during the phase)_

---

## Phase 12 — Debugging & training pathologies

_(add during the phase)_

---

## Phase 13 — Preference optimization: DPO & family

_(add during the phase)_

---

## Phase 14 — Reward modeling

_(add during the phase)_

---

## Phase 15 — GRPO & verifiable rewards

_(add during the phase)_

---

## Phase 16 — Merging, quantization & serving

_(add during the phase)_

---

## Phase 17 — Accelerator frameworks & reproducibility

_(add during the phase)_

---

## Phase 18 — Capstones & portfolio

_(add during the phase)_
