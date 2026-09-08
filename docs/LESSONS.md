# LESSONS.md — transferable principles, each with the incident that produced it

**The rule for this file:** a lesson only gets written down after something actually went wrong (or surprised you) in a real run. No lesson may be added from reading. That constraint is what makes the file worth re-reading later — every entry has a scar attached.

**Format:** `L<n>` — the principle in one line, then *Origin:* the specific incident, then *How to apply:* what you now do differently.

Grouped by **triggering situation**, not by phase, because that is how you will search it at 11 pm during an incident. The origin index at the end gives the chronology.

---

## A. When you are trusting your data

_(empty — fill from Phases 02, 03, 07, 08, 09)_

---

## B. When you are trusting your metrics

_(empty — fill from Phases 05, 09, 16)_

---

## C. Memory and throughput intuition

_(empty — fill from Phases 00, 04, 10, 11)_

---

## D. When to reach for which method

_(empty — fill from Phases 06, 11, 13, 14, 15)_

---

## E. Reproducibility and process

_(empty — fill from Phases 04, 12, 17)_

---

## Origin index

| Lesson | Phase | Incident |
|---|---|---|
| | | |

---

## Seed entry — the one lesson you are allowed to write before an incident

**L0 — The library's extra parameter is usually somebody else's bug report.**

*Origin:* Written at the start, before Phase 00, as a reading posture rather than an incident.

*How to apply:* Every option in `LoraConfig`, `SFTConfig` or `TrainingArguments` exists because a default broke for someone. When you meet an unfamiliar parameter, do not skip it — ask *what failure made this configurable?* That question turns API surface into accumulated field experience, and it is the fastest way to absorb what other people learned the hard way.
