# Phase 16 — Merging, quantization & serving

**~10 hrs** · Prerequisite: [Phase 15](phase-15.md) · Next: [Phase 17](phase-17.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **take an adapter to a measured endpoint, and price every optimization in quality rather than only in speed**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** a merged model, a quantized model, a live vLLM endpoint, a throughput-vs-latency frontier plot, and **an eval delta per quantization format**

**Models:** the golden-thread model from Phase 13/15

**Data:** the Phase 07 domain data as the quantization calibration set — not a generic web corpus

---

## 1. Concept

- Export correctly: merge, `save_pretrained` with the right dtype, and the point everyone misses — **the chat template is part of the model artifact.** If `tokenizer_config.json` does not carry the template you trained with, you have shipped Phase 12 failure #9
- **Quantization for inference is a different problem from QLoRA for training**, and conflating them is a common beginner error. Inference quantization is post-training and optimizes for bandwidth; QLoRA is a training-memory trick
- W4A16 (GPTQ / AWQ-style), GGUF for CPU and your Mac, and bnb-4bit plus the reason it is a poor *serving* choice
- **Calibration-set selection** — use *your* domain data, and measure the difference against a generic corpus. This is a real, measurable effect that almost nobody checks
- Per-channel vs per-group quantization (g=128); KV-cache quantization, where **FP8 KV is unavailable on SM 8.6** so int8 is your option — another A10-specific derivation
- **vLLM**: PagedAttention and the fragmentation problem it solved; continuous batching; `--max-model-len` traded against the KV budget per `NUMBERS.md` §8; **tensor-parallel 2 without NVLink** — measure the overhead and derive from your Phase 00 PCIe number why TP across PCIe is a bad trade for an 8B that already fits on one card (**run two replicas instead**); **multi-LoRA serving** (`--enable-lora`, several adapters on one base) which is the killer economic argument for LoRA and the reason you might never merge; prefix caching

_(Write the explanation here as you work through it. Anything you cannot explain in your own words is not yet learned.)_

---

## 2. Numbers

**Pre-register your predictions here BEFORE launching anything** — rule 2 of [PLAN.md](../../PLAN.md). The gap between prediction and measurement is the lesson.

| Quantity | Predicted | Measured | Gap — and why |
|---|---|---|---|
| Peak VRAM | | | |
| Tokens/sec | | | |
| Step time | | | |
| Wall-clock to finish | | | |

Formulas used: see [NUMBERS.md](../../NUMBERS.md).

---

## 3. Landscape & choice

- Merge vs serve-adapters-live. **Measure both.** Multi-adapter serving on one base is often the better architecture and it is under-appreciated

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Measure TTFT and inter-token latency at p50/p95, and tokens/sec at concurrency 1 / 8 / 32 / 64. Plot the throughput-vs-latency frontier
- **Then the step nobody takes: re-run the full Phase 05 eval harness on bf16 vs W4A16 vs GGUF-Q4.** Report the quality delta alongside the speed delta
- A quantization that costs 4 points for 2× throughput is a **decision**, not a free win. 'Quantization is free' is the most common serving myth and you will have the numbers to kill it
- Run the GGUF build on your Mac. Your own fine-tuned model, running locally, is a satisfying and portfolio-legible artifact

**How you will know it worked:** _(state the acceptance criterion before you build)_

---

## 5. Build & Q&A

### What was built

_(files, commands, run-ids)_

### Results

_(the table, the plot, the number that moved)_

### What surprised you

_(this is the raw material for [docs/LESSONS.md](../LESSONS.md) — anything that surprised you is a candidate lesson)_

### Questions

_(append to [question_answer.md](../../question_answer.md) as `Q16.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
