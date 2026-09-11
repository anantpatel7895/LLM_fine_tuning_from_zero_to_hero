# LLM_fine_tuning_from_zero_to_hero

Becoming expert at LLM fine-tuning **and** data preparation, on 2× NVIDIA A10.

**→ The map is [PLAN.md](PLAN.md). Start there.**

| | |
|---|---|
| The curriculum | [PLAN.md](PLAN.md) — 19 phases, ~170 hrs |
| Current phase | [docs/phases/phase-00.md](docs/phases/phase-00.md) |
| Every formula | [NUMBERS.md](NUMBERS.md) |
| Everything about data | [docs/DATA.md](docs/DATA.md) |
| What the box can do | [docs/HARDWARE.md](docs/HARDWARE.md) |
| Recall practice | [question_answer.md](question_answer.md) |
| Hard-won principles | [docs/LESSONS.md](docs/LESSONS.md) |
| Remote GPU workflow | [docs/setup/remote-gpu.md](docs/setup/remote-gpu.md) |

---

## Command cheat-sheet

### Environment

> ⚠️ **Three requirements files. Using the wrong one is the most likely setup mistake.**
> They are not interchangeable, and installing the Mac file on the box leaves you with
> **no training libraries at all** (`peft`, `trl`, `accelerate`, `bitsandbytes` are
> GPU-only). Pick by machine:

> 🔴 **Always `python -m pip`, never bare `pip`.** On this box a stray
> `~/.local/bin/pip` for Python 3.10 shadowed the venv's pip and silently installed 6 GB
> into the wrong interpreter, twice. `python -m pip` cannot do that. Full explanation:
> [docs/LESSONS.md](docs/LESSONS.md) **L1**. Sanity check before installing — these two
> must agree:
>
> ```bash
> python -V && python -m pip -V
> ```

**On the Mac** — authoring only: notes, configs, tokenizers, templates, `pytest`.

```bash
pyenv virtualenv 3.12.10 FT_VENV && pyenv activate FT_VENV
python -m pip install -r requirements.txt        # <-- Mac ONLY. No CUDA, no trainers.
```

**On the A10 box** — training. Note torch comes from the CUDA index *first*, separately:

```bash
pyenv virtualenv 3.12.10 FT_VENV && pyenv activate FT_VENV
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt    # <-- box ONLY
```

**Serving (Phase 16)** — a *separate* venv on the box. vLLM pins an exact older torch and
will silently downgrade your training environment if installed alongside it:

```bash
pyenv virtualenv 3.12.10 FT_SERVE && pyenv activate FT_SERVE
python -m pip install -r requirements-serving.txt
```

| File | Machine | Contains |
|---|---|---|
| `requirements.txt` | Mac | docs, configs, tokenizers, pytest. **No CUDA, no trainers** |
| `requirements-gpu.txt` | box (`FT_VENV`) | `peft`, `trl`, `accelerate`, `bitsandbytes`, flash-attn |
| `requirements-serving.txt` | box (`FT_SERVE`) | vLLM + quantization. **Never in `FT_VENV`** |

### Sync Mac → box

```bash
./scripts/sync_to_gpu.sh                  # rsync code + configs, never data/ or artifacts/
```

### Hardware report (run on the box)

```bash
./scripts/00_hardware_report.sh           # driver, CUDA, topology, capability
python -m llm_finetuning_from_zero_to_hero.00_gpu_benchmark    # TFLOPS, bandwidth, all-reduce
```

### Smoke test before any real run

```bash
./scripts/smoke_test.sh configs/sft/<recipe>.yaml    # ~50 steps, sub-1B model
```

### Launch a real run (in tmux, so it survives the SSH drop)

```bash
tmux new -s ft
./scripts/launch.sh configs/sft/<recipe>.yaml
# ctrl-b d to detach; tmux attach -t ft to come back
```

### Watch a run

```bash
tail -f artifacts/<run-id>/train.log
tensorboard --logdir artifacts/ --port 6006     # then: ssh -L 6006:localhost:6006 <box>
watch -n 2 nvidia-smi
```

### Useful digging

```bash
nvidia-smi topo -m                          # interconnect topology — PCIe vs NVLink
nvidia-smi dmon                             # live power, clocks, utilization
python -c "import torch; print(torch.cuda.get_device_capability())"   # expect (8, 6)
find docs/phases -name '*.md' | sort        # the phase list
```

---

## Run-id convention

```
{date}-{phase}-{method}-{model}-{confighash8}
20260907-p04-qlora-llama31-8b-a3f91c2e
```

Everything about a run lives in `artifacts/<run-id>/`: the config, `run_card.md`, logs, adapter, eval JSON.

---

## Repo rules

- `notebooks/` never imports from `src/` — code either graduates into the package or dies in the notebook.
- Prose lives in `docs/`. Code carries short comments, not essays.
- `data/` and `artifacts/` are gitignored; their `manifest.json` and `datasheet.md` are not.
- **Numeric filename prefixes only on run-only entry points.** `python -m pkg.00_thing` works, but `import pkg.00_thing` is a `SyntaxError` — a module name cannot start with a digit. So `00_gpu_benchmark.py` (only ever run) keeps its prefix; anything you will import or unit-test (`templates.py`, `mask_inspector.py`) must not have one. Ordering for those lives in the phase docs instead.
