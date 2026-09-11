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

**L1 — `pip` and `python` are two unrelated lookups. Always run `python -m pip`.**

*Origin:* Phase 00, 2026-09-11, on the A10 box. `pyenv activate FT_VENV` succeeded — the prompt showed `(FT_VENV)`, `VIRTUAL_ENV` pointed at the right directory, and `python -V` printed the correct 3.12.10. Then `pip install -r requirements.txt` reported **"Successfully installed torch-2.14.0"** and `python -c "import torch"` immediately failed with `ModuleNotFoundError: No module named 'torch'`. Installing it a second time also "succeeded", and it still could not be imported.

The cause, from `pip -V`:

```
python  →  ~/.pyenv/shims/python                       Python 3.12.10   ← the venv
pip     →  ~/.local/bin/pip        ...python3.10/site-packages/pip (python 3.10)
```

`~/.local/bin` sat **earlier in `PATH`** than the venv's `bin/`, so a stray user-installed pip for Python 3.10 shadowed the venv's pip. Every package landed in `~/.local/lib/python3.10/site-packages`, which a Python 3.12 interpreter never reads. About 6 GB of torch and CUDA wheels went to an interpreter that would never be used again — onto the *nearly-full root disk*, while the intended target sat empty.

### Why this happens — the mechanism worth understanding

Four facts that compose into the trap:

1. **The shell knows nothing about Python.** When you type `pip`, the shell walks `$PATH` left to right and runs the **first** file called `pip` it finds. That is the entire algorithm.

2. **A virtualenv is not magic — it is mostly a `PATH` edit.** "Activating" prepends `$VIRTUAL_ENV/bin` to `PATH` and sets `VIRTUAL_ENV`. It cannot stop an *earlier* `PATH` entry from winning.

3. **`pip` is a tiny script with a hardcoded interpreter.** Look inside `~/.local/bin/pip` and the first line is a shebang naming an absolute Python path. **That shebang — not your shell, not `VIRTUAL_ENV` — decides which `site-packages` gets written.** So `pip` installs for whatever Python is baked into the copy of `pip` that PATH happened to find.

4. **`python -m pip` has no such lookup.** You run `python` (resolved correctly), and tell *that interpreter* to load its own `pip` module and run it. There is no second executable and no shebang involved, so the interpreter necessarily installs for itself. It is structurally incapable of targeting the wrong Python.

### Why it is dangerous rather than merely annoying

It is **silent, and it mimics the opposite problem.** pip prints success. The prompt, `VIRTUAL_ENV` and `python -V` all look correct. The only symptom is `ModuleNotFoundError`, whose obvious reading is *"I forgot to install it"* — so you install again, it succeeds again, into the wrong place again. That loop can eat an evening, and on a tight disk each iteration burns gigabytes.

*How to apply:*

- **Never type bare `pip`. Type `python -m pip`.** No exceptions, even when the venv "looks" active.
- **Two-second check before any install:** run `pip -V` and `python -V` and confirm the versions match. They print their interpreter for exactly this reason.
- **Learn the tell:** `Defaulting to user installation because normal site-packages is not writeable` inside a supposedly active venv means **you are not in the venv you think you are**. A real venv's `site-packages` is always writable by you, so that line is proof of the shadowing, not a harmless notice.
- **This generalizes to every console script**, because they are all shebang scripts: `pytest`, `jupyter`, `huggingface-cli`, and — critically for this curriculum — **`accelerate` and `torchrun`**. In Phase 10 a shadowed `accelerate launch` would start your multi-GPU job under a *different* interpreter than the one you tested with, and the failure would look like a distributed-training bug rather than an environment bug. Prefer `python -m pytest`, `python -m torch.distributed.run`, `python -m accelerate.commands.launch`.
- **Corollary for diagnostics:** "not installed" and "installed but unimportable" need opposite fixes, so never let a tool collapse them. `scripts/00_hardware_report.sh` was changed after this incident to check distribution metadata first and report `INSTALLED but IMPORT FAILED` separately, with the exception text.

---

## Origin index

| Lesson | Phase | Incident |
|---|---|---|
| L1 | 00 (2026-09-11) | `pip` resolved to a Python 3.10 user install while `python` was the 3.12.10 venv; torch "installed successfully" twice and was unimportable both times |

---

## Seed entry — the one lesson you are allowed to write before an incident

**L0 — The library's extra parameter is usually somebody else's bug report.**

*Origin:* Written at the start, before Phase 00, as a reading posture rather than an incident.

*How to apply:* Every option in `LoraConfig`, `SFTConfig` or `TrainingArguments` exists because a default broke for someone. When you meet an unfamiliar parameter, do not skip it — ask *what failure made this configurable?* That question turns API surface into accumulated field experience, and it is the fastest way to absorb what other people learned the hard way.
