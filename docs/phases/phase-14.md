# Phase 14 — Reward modeling

**~8 hrs** · Prerequisite: [Phase 13](phase-13.md) · Next: [Phase 15](phase-15.md)

> **Status:** ⬜ Not Started

## 0. Learning goal

After this phase you can **train a reward model, measure it honestly, and know the three different jobs one reward model can do**.

| Step | Status |
|---|:---:|
| 1. Concept | ⬜ |
| 2. Numbers | ⬜ |
| 3. Landscape & choice | ⬜ |
| 4. Design | ⬜ |
| 5. Build & Q&A | ⬜ |

**Deliverable:** a trained reward model with a scorecard (pairwise accuracy on held-out pairs, per-category breakdown, length-bias analysis); a best-of-n vs greedy quality/latency table

**Models:** Qwen3-1.7B with a Bradley-Terry head

**Data:** the preference dataset you built in Phase 13

---

## 1. Concept

- `RewardTrainer` and the Bradley-Terry head — a scalar output trained so that `r(chosen) > r(rejected)`
- **Pairwise accuracy of 65–75% is normal, not broken** — human agreement on preference is itself only ~70%. Knowing this stops you chasing a number that cannot exist
- **Reward hacking as a Goodhart phenomenon**, with its characteristic curve: true quality rises then *falls* while measured reward keeps rising. Detecting the turn is the skill
- **Length bias** — reward models learn 'longer is better' from the data and then teach it back. Measure reward against length explicitly
- Reward-model ensembles; generative / LLM-judge reward models; **PRM vs ORM** (process vs outcome supervision) and why process supervision matters for reasoning

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

- **Three uses for one reward model, which is the insight that ties this curriculum together:** an RL signal (Phase 15), **best-of-n / rejection sampling** at inference *and* as a data generator (Phase 08), and **data filtering** (Phase 07). Same artifact, three jobs — and you have now done all three

**Decision:** _(what you chose, and the alternative you rejected — one paragraph each)_

---

## 4. Design

- Train the RM, then score it: held-out pairwise accuracy with a confidence interval, a per-category breakdown, and reward plotted against response length
- Best-of-n against greedy on the golden-thread task: measure the quality gain **and the latency cost**, because best-of-n is n× the compute
- Verify the RM against your programmatic verifier on the subset where the verifier applies. Where they disagree, the RM is wrong — that is a rare and valuable calibration opportunity

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

_(append to [question_answer.md](../../question_answer.md) as `Q14.n`, tagged ⭐ interview / 🔧 incident / 📐 whiteboard-derivable)_

### Adversarial close

**What does this now fail on?** _(rule 12 — the failure goes into the eval suite, the repro goes into LESSONS.md)_
