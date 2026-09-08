# Phase 15 — GRPO & verifiable rewards

**~12 hrs** · Prerequisite: [Phase 14](phase-14.md) · Next: [Phase 16](phase-16.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **close one reinforcement-learning loop end to end on your own hardware, design a reward function, and watch it get hacked**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** a GRPO run moving a verifiable metric; reward curves; a **reward-hacking postmortem**; `docs/notes/rl-for-llms.md`

**Models:** Qwen3-1.7B — trainer on GPU 0, vLLM generation on GPU 1

**Data:** prompt-only data for the golden-thread task, plus a public verifiable-reward set as a control

---

## 1. Concept

- Policy-gradient recap, then PPO, then **why GRPO fits small GPUs**: it drops the value network entirely and estimates advantage *group-relatively* from G sampled completions per prompt. No critic means roughly half the memory
- The KL-to-reference penalty and what happens without it (the policy walks away from fluent language entirely)
- Group size G and its variance/compute trade
- **The compute reality: generation dominates.** Which is why the architecture is vLLM on GPU 1 serving rollouts while the trainer occupies GPU 0 — and why the model has to be small
- **Verifiable rewards only at this scale.** Exact match, schema validity, unit tests passing. No learned reward model in the loop, because a learned RM plus RL plus 24 GB is three problems at once

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

- Where PPO / RLOO / Online-DPO / Nash-MD fit, and **an explicit written statement of why you are not running them on this hardware.** Naming the boundary is part of the expertise

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Design the reward: schema-valid JSON + correct tool + correct arguments, as a weighted sum, reusing the Phase 08 verifier
- **Then watch it get hacked.** The model will find the format reward without the correctness reward, or exploit a lenient parser. **That incident is the single most valuable `LESSONS.md` entry in the repo** — write down what it found and how you closed the hole
- Plot reward, KL from reference, and the true task metric on the same axis. When reward rises and the true metric does not, you are looking at Goodhart's law in your own run
- Compare against the Phase 13 DPO checkpoint on the same eval. Be honest if DPO wins — on a small model with a good verifier, it sometimes does

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

_(append to [question_answer.md](../../question_answer.md) as `Q15.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
