# Remote GPU workflow — Mac → A10 box

The split: **you author on the Mac, you train on the box.** Notes, configs, data-inspection code and this whole repo live on the Mac and are version-controlled there. The box runs training and holds datasets, checkpoints and logs.

Nothing in this document is fine-tuning-specific. It is the plumbing that has to be boring so the interesting parts can be interesting.

---

## 1. SSH, once

Set up key-based auth and a host alias so every later command is short.

```bash
# on the Mac — generate a key if you don't have one for this
ssh-keygen -t ed25519 -C "mac-to-a10" -f ~/.ssh/id_a10

# copy the public key to the box
ssh-copy-id -i ~/.ssh/id_a10.pub <user>@<box-host>
```

Then add an alias to `~/.ssh/config`:

```
Host a10
    HostName <box-host-or-ip>
    User <user>
    IdentityFile ~/.ssh/id_a10
    ServerAliveInterval 60
    ServerAliveCountMax 10
    # forward TensorBoard and a vLLM port back to the Mac
    LocalForward 6006 localhost:6006
    LocalForward 8000 localhost:8000
```

`ServerAliveInterval` matters: without it, a laptop that sleeps drops the connection and takes your foreground training process with it. Which is why everything real runs inside tmux.

Test: `ssh a10 nvidia-smi`

---

## 2. tmux, because SSH will drop

**Rule: never run a training job in a bare SSH session.** The connection will drop — laptop sleeps, wifi changes, VPN reconnects — and an un-detached process dies with it, losing hours of GPU time.

```bash
ssh a10
tmux new -s ft              # create a named session
# ... launch the run ...
# ctrl-b then d             → detach; the run keeps going
exit                        # safe to close the SSH session entirely

# later, from anywhere:
ssh a10
tmux attach -t ft           # right back where you were
tmux ls                     # what sessions exist
```

Naming convention: one session per concurrent run — `ft`, `ft2`, `vllm`, `eval`. On this box you will often have two training sessions at once, one per GPU (see [NUMBERS.md §6](../NUMBERS.md) on treating GPU 1 as a second experiment slot).

---

## 3. Sync

Code goes Mac → box. Data and checkpoints never come back.

`scripts/sync_to_gpu.sh` uses rsync with an **explicit exclude list**. The exclusions are the important part — syncing `data/` or `artifacts/` will either take hours or fill your Mac's disk with checkpoints.

```bash
./scripts/sync_to_gpu.sh            # push code + configs + docs
./scripts/sync_to_gpu.sh --dry-run  # see what would move, first
```

**Alternative worth knowing:** commit and `git pull` on the box. Slower loop, but it guarantees the box is at a known commit — which matters for the run records in Phase 04 onward, where a run's git SHA is part of its identity. Use rsync while iterating, git for anything you will report a number from.

Pull results *back* selectively — eval JSON and reports, never checkpoints:

```bash
rsync -avz a10:~/LLM_fine_tuning_from_zero_to_hero/artifacts/<run-id>/{run_card.md,eval.json,train.log} \
    artifacts/<run-id>/
```

---

## 4. Environment on the box

pyenv, matching the convention in your other repos:

```bash
# once
curl https://pyenv.run | bash
pyenv install 3.12.10
pyenv virtualenv 3.12.10 FT_VENV

# every session
pyenv activate FT_VENV
pip install -r requirements-gpu.txt
```

**`requirements-gpu.txt` is separate from `requirements.txt` for a reason:** `torch` with a CUDA build, `bitsandbytes`, `flash-attn`, `deepspeed` and `vllm` have no working Mac wheels. Keeping them in a separate file means the Mac environment stays installable and you can inspect data and edit configs locally without fighting CUDA wheels.

Environment variables to set in `~/.bashrc` on the box:

```bash
export HF_HOME=/mnt/<big-disk>/hf                 # NOT ~/, models are tens of GB
export HF_HUB_ENABLE_HF_TRANSFER=1                # much faster model downloads
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduces fragmentation OOMs
export TOKENIZERS_PARALLELISM=false               # silences a warning you'll otherwise see 10000 times
```

`HF_HOME` on the wrong disk is the single most common way to fill a root partition. Check `df -h` before your first download.

---

## 5. Watching a run

```bash
# from the Mac, thanks to the LocalForward lines in ~/.ssh/config:
#   TensorBoard → http://localhost:6006
ssh a10 'cd ~/LLM_fine_tuning_from_zero_to_hero && tensorboard --logdir artifacts/ --port 6006'

# logs
ssh a10 tail -f ~/LLM_fine_tuning_from_zero_to_hero/artifacts/<run-id>/train.log

# GPU state
ssh a10 watch -n 2 nvidia-smi
ssh a10 nvidia-smi dmon          # live power/clocks/utilization — catches thermal throttling
```

---

## 6. Debugging a run that has gone wrong

Keep this list handy; Phase 12 explains what each one is for.

```bash
# what is the process actually doing right now? (works on a hung run)
py-spy dump --pid <pid>

# async CUDA errors report at the wrong line by default; this fixes attribution
CUDA_LAUNCH_BLOCKING=1 python train.py

# multi-GPU hangs
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ...

# memory forensics — see fragmentation instead of guessing at it
# (in code) torch.cuda.memory._record_memory_history() then _dump_snapshot()
```

---

## 7. The pre-flight checklist

Before launching anything that will run for hours:

- [ ] Inside `tmux`
- [ ] Smoke test passed on this exact config (`./scripts/smoke_test.sh`)
- [ ] Predicted VRAM and tok/s written into the phase doc (rule 2)
- [ ] `df -h` shows room for the checkpoints this run will write
- [ ] Config committed, so the run's git SHA means something
- [ ] `nvidia-smi` shows both GPUs idle — no leftover process holding VRAM
- [ ] The run writes to `artifacts/<run-id>/` and `train.log` is being teed
