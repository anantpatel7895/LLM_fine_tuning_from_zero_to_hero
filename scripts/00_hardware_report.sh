#!/usr/bin/env bash
# Phase 00 — hardware report. Run this ON THE A10 BOX and paste the output into docs/HARDWARE.md.
#
# Read-only: queries the driver and the Python environment, changes nothing.

set -uo pipefail   # deliberately no -e: a missing optional tool should not abort the report

hr() { printf '\n%s\n%s\n' "=== $1 ===" "$(printf '%.0s-' {1..70})"; }

printf '%s\n' "Hardware report — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
printf '%s\n' "host: $(hostname)"

hr "1. GPUs"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap,power.limit \
           --format=csv 2>/dev/null || echo "nvidia-smi not available"

hr "2. Interconnect topology  (looking for PCIe/SYS vs NVLink)"
# NV# in this matrix means NVLink; PIX/PXB/PHB/NODE/SYS all mean PCIe of varying distance.
nvidia-smi topo -m 2>/dev/null || echo "topology unavailable"

hr "3. Clocks and power (idle baseline — re-check DURING a run to see throttling)"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit \
           --format=csv 2>/dev/null

hr "4. CPU and RAM"
if [ -r /proc/cpuinfo ]; then
    echo "cores:  $(nproc)"
    echo "model:  $(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo | xargs)"
fi
free -h 2>/dev/null || vm_stat 2>/dev/null | head -5

hr "5. Disk — check HF_HOME is NOT on a small root partition"
echo "HF_HOME=${HF_HOME:-<unset — models will land in ~/.cache/huggingface>}"
df -h "${HF_HOME:-$HOME}" 2>/dev/null
echo
df -h . 2>/dev/null

hr "6. Relevant environment variables"
for v in HF_HOME HF_HUB_ENABLE_HF_TRANSFER PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM \
         CUDA_VISIBLE_DEVICES NCCL_DEBUG; do
    printf '%-32s %s\n' "$v" "${!v:-<unset>}"
done

hr "7. Python / torch / capability matrix"
python - <<'PY' 2>&1
import importlib.metadata as md
import sys

print(f"python                      {sys.version.split()[0]}")
print(f"python executable           {sys.executable}")

# "not installed" and "installed but broken" need completely different fixes, so never
# collapse them into one message. A missing package means `pip install`; a failing import
# usually means a corrupt install, a truncated download (often from a full disk), or a
# CUDA library mismatch -- and pip will happily report "Successfully installed" for a
# package that cannot actually be imported.
try:
    dist_version = md.version("torch")
except md.PackageNotFoundError:
    dist_version = None

if dist_version is None:
    print("torch                       NOT INSTALLED (no distribution metadata)")
    print("                            -> pip install torch==2.14.0 --index-url ...cu128")
    raise SystemExit(0)

try:
    import torch
except Exception as exc:                      # not just ImportError -- CUDA libs raise OSError
    print(f"torch                       INSTALLED ({dist_version}) but IMPORT FAILED")
    print(f"                            {type(exc).__name__}: {exc}")
    print()
    print("The package is present but unusable. Most likely causes, in order:")
    print("  1. a truncated/corrupt install -- often a disk that filled mid-download")
    print("  2. mismatched nvidia-* CUDA wheels (e.g. cu13 libs against a cu12 torch)")
    print("  3. a conflicting torch pin dragged in by another package (torchvision, unsloth)")
    print("Check `df -h /` first, then reinstall torch alone and import it before anything else.")
    raise SystemExit(0)

print(f"torch                       {torch.__version__}")
print(f"torch CUDA build            {torch.version.cuda}")
print(f"cuDNN                       {torch.backends.cudnn.version()}")
print(f"CUDA available              {torch.cuda.is_available()}")
print(f"device count                {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    props = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {props.name}  sm_{cap[0]}{cap[1]}  "
          f"{props.total_memory / 1024**3:.1f} GiB  {props.multi_processor_count} SMs")

print()
print("--- capability matrix, derived from the reported compute capability ---")
major, minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
cc = major * 10 + minor
checks = [
    ("bf16 supported",        torch.cuda.is_bf16_supported()),
    ("TF32 matmul enabled",   torch.backends.cuda.matmul.allow_tf32),
    ("FP8 (needs sm_89+)",    cc >= 89),
    ("FlashAttention-2 (sm_80+)", cc >= 80),
    ("FlashAttention-3 (sm_90 only)", cc >= 90),
]
for name, ok in checks:
    print(f"  {'YES' if ok else 'no ':<4} {name}")

print()
print("--- key libraries ---")
for pkg in ("transformers", "peft", "trl", "accelerate", "bitsandbytes",
            "datasets", "deepspeed", "flash_attn", "vllm", "tensorboard"):
    try:
        print(f"  {pkg:<16} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"  {pkg:<16} not installed")

print()
print("--- import checks (a version number does not prove it works) ---")
try:
    import flash_attn  # noqa: F401
    print("  flash_attn imports OK")
except Exception as exc:
    print(f"  flash_attn import FAILED: {type(exc).__name__}: {exc}")
try:
    import bitsandbytes as bnb
    print(f"  bitsandbytes imports OK (CUDA support: {bnb.COMPILED_WITH_CUDA if hasattr(bnb, 'COMPILED_WITH_CUDA') else 'unknown'})")
except Exception as exc:
    print(f"  bitsandbytes import FAILED: {type(exc).__name__}: {exc}")
PY

hr "done"
echo "Paste this into docs/HARDWARE.md sections 1, 2 and 5."
