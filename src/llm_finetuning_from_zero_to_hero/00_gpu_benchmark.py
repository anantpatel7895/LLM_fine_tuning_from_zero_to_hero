"""Phase 00 — measure what the box actually does, rather than what the spec sheet claims.

Four measurements, in order of how much they matter for this curriculum:

1. ``allreduce``  — PCIe bandwidth between the two GPUs. THE most important number here:
                    it decides every parallelism choice in Phase 10. See NUMBERS.md section 6.
2. ``matmul``     — achieved bf16 TFLOPS, i.e. your real MFU ceiling.
3. ``bandwidth``  — HBM read/write bandwidth, which sets the decode-side roofline.
4. ``memory``     — how much of the nominal 24 GB you can actually allocate.

Run on the box::

    # single-process measurements
    python -m llm_finetuning_from_zero_to_hero.00_gpu_benchmark

    # the all-reduce test needs both GPUs, so it re-launches itself under torchrun.
    # to run it directly:
    torchrun --nproc_per_node=2 -m llm_finetuning_from_zero_to_hero.00_gpu_benchmark --only allreduce

Paste the output into docs/HARDWARE.md section 3.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

# A10 (GA102) spec-sheet figures, for the predicted-vs-measured comparison.
A10_PEAK_BF16_TFLOPS = 125.0
A10_PEAK_BANDWIDTH_GBPS = 600.0

SEPARATOR = "-" * 72


def _sync() -> None:
    torch.cuda.synchronize()


def _timed(fn, warmup: int = 5, iters: int = 20) -> float:
    """Return median seconds per call, after warmup. Median, not mean — one scheduler
    hiccup should not move the number."""
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


# --------------------------------------------------------------------------------------
# 1. matmul throughput
# --------------------------------------------------------------------------------------
def bench_matmul(device: int = 0) -> list[dict]:
    """Achieved bf16 matmul TFLOPS at three shapes.

    Small shapes fall well short of peak because there is not enough work to hide memory
    latency. That gap is exactly why micro-batch size affects MFU in Phase 04.
    """
    print(f"\n=== 1. matmul throughput (bf16, device {device}) ===")
    print(SEPARATOR)
    torch.cuda.set_device(device)
    results = []

    for n in (2048, 4096, 8192):
        a = torch.randn(n, n, device=f"cuda:{device}", dtype=torch.bfloat16)
        b = torch.randn(n, n, device=f"cuda:{device}", dtype=torch.bfloat16)
        seconds = _timed(lambda: torch.matmul(a, b))
        # one n*n*n matmul is 2*n^3 FLOPs: one multiply and one add per element pair
        tflops = (2 * n**3) / seconds / 1e12
        pct = 100 * tflops / A10_PEAK_BF16_TFLOPS
        print(f"  {n:>5}^3   {seconds * 1e3:8.3f} ms   {tflops:7.2f} TFLOPS   {pct:5.1f}% of peak")
        results.append({"shape": n, "ms": seconds * 1e3, "tflops": tflops, "pct_peak": pct})
        del a, b
        torch.cuda.empty_cache()

    print(f"\n  spec-sheet peak: {A10_PEAK_BF16_TFLOPS} TFLOPS bf16")
    print("  NOTE: sustained numbers in a real 3-hour run will be lower — a 150 W card throttles.")
    print("        Check with `nvidia-smi dmon` DURING training, not at idle.")
    return results


# --------------------------------------------------------------------------------------
# 2. memory bandwidth
# --------------------------------------------------------------------------------------
def bench_bandwidth(device: int = 0) -> dict:
    """HBM bandwidth via a large device-to-device copy.

    This sets the decode-side roofline: autoregressive generation is bandwidth-bound
    (NUMBERS.md section 11), so this number caps your Phase 16 serving throughput.
    """
    print(f"\n=== 2. memory bandwidth (device {device}) ===")
    print(SEPARATOR)
    torch.cuda.set_device(device)

    n_bytes = 2 * 1024**3  # 2 GiB
    src = torch.empty(n_bytes // 2, device=f"cuda:{device}", dtype=torch.float16)
    dst = torch.empty_like(src)

    seconds = _timed(lambda: dst.copy_(src), warmup=3, iters=10)
    # a copy touches each byte twice: one read + one write
    gbps = (2 * n_bytes) / seconds / 1e9
    pct = 100 * gbps / A10_PEAK_BANDWIDTH_GBPS
    print(f"  2 GiB d2d copy   {seconds * 1e3:8.3f} ms   {gbps:7.1f} GB/s   {pct:5.1f}% of peak")
    print(f"\n  spec-sheet peak: {A10_PEAK_BANDWIDTH_GBPS} GB/s")
    print(f"  ridge point = {A10_PEAK_BF16_TFLOPS * 1e12 / (gbps * 1e9):.0f} FLOP/byte "
          f"(measured)   vs {A10_PEAK_BF16_TFLOPS * 1e12 / (A10_PEAK_BANDWIDTH_GBPS * 1e9):.0f} (spec)")
    print("  Above the ridge point = compute-bound (training). Below = memory-bound (decode).")

    del src, dst
    torch.cuda.empty_cache()
    return {"gbps": gbps, "pct_peak": pct}


# --------------------------------------------------------------------------------------
# 3. usable memory
# --------------------------------------------------------------------------------------
def bench_usable_memory(device: int = 0) -> dict:
    """How much of the nominal 24 GB can you actually allocate?

    Answers Q00.3. The gap is CUDA context, allocator granularity, and (with NCCL
    initialised) communication buffers. Budget the measured number, not 24 GB.
    """
    print(f"\n=== 3. usable memory (device {device}) ===")
    print(SEPARATOR)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    total = torch.cuda.get_device_properties(device).total_memory
    free_before, _ = torch.cuda.mem_get_info(device)

    # Establish the context cost, then grow a single allocation until it fails.
    _probe = torch.empty(1, device=f"cuda:{device}")
    free_after_ctx, _ = torch.cuda.mem_get_info(device)
    context_bytes = free_before - free_after_ctx

    allocated = []
    chunk = 256 * 1024**2  # 256 MiB
    try:
        while True:
            allocated.append(torch.empty(chunk, device=f"cuda:{device}", dtype=torch.uint8))
    except torch.cuda.OutOfMemoryError:
        pass

    usable = len(allocated) * chunk
    del allocated, _probe
    torch.cuda.empty_cache()

    print(f"  nominal total          {total / 1024**3:7.2f} GiB")
    print(f"  free before any alloc  {free_before / 1024**3:7.2f} GiB")
    print(f"  CUDA context cost      {context_bytes / 1024**2:7.1f} MiB")
    print(f"  largest sum allocated  {usable / 1024**3:7.2f} GiB   "
          f"({100 * usable / total:.1f}% of nominal)")
    print("\n  Budget the measured number in every VRAM prediction. NOT 24 GB.")
    print("  With NCCL initialised (multi-GPU) expect a few hundred MiB less again.")
    return {
        "total_gib": total / 1024**3,
        "context_mib": context_bytes / 1024**2,
        "usable_gib": usable / 1024**3,
    }


# --------------------------------------------------------------------------------------
# 4. all-reduce over the interconnect  —  THE important one
# --------------------------------------------------------------------------------------
def bench_allreduce() -> list[dict]:
    """PCIe all-reduce bandwidth. Must run under torchrun with 2 processes.

    This is the number that decides Phase 10. Plug it into NUMBERS.md section 6 and
    re-derive whether DDP or ZeRO-3 wins for each training regime.

    The 84 MB payload is not arbitrary: it is exactly one rank-16 LoRA adapter for an
    8B model (NUMBERS.md section 5), so its timing is the real per-step DDP cost of
    every LoRA run in this curriculum.
    """
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"\n=== 4. all-reduce over the interconnect (world_size={world}) ===")
        print(SEPARATOR)
        print("  payload            time       algo BW      bus BW     note")

    payloads = [
        (4 * 1024**2, "4 MiB"),
        (84 * 1024**2, "84 MiB  <- one r=16 LoRA adapter on an 8B"),
        (1024**3, "1 GiB"),
    ]

    results = []
    for n_bytes, label in payloads:
        tensor = torch.ones(n_bytes // 2, device=f"cuda:{local_rank}", dtype=torch.float16)

        dist.barrier()
        seconds = _timed(lambda: dist.all_reduce(tensor), warmup=5, iters=20)

        # Ring all-reduce moves 2*(N-1)/N of the payload per rank across the wire.
        algo_gbps = n_bytes / seconds / 1e9
        bus_gbps = algo_gbps * 2 * (world - 1) / world

        if rank == 0:
            head, _, note = label.partition("  <-")
            print(f"  {head:<10} {seconds * 1e3:9.3f} ms  "
                  f"{algo_gbps:7.2f} GB/s  {bus_gbps:7.2f} GB/s   {note.strip()}")
            results.append({
                "bytes": n_bytes,
                "ms": seconds * 1e3,
                "algo_gbps": algo_gbps,
                "bus_gbps": bus_gbps,
            })

        del tensor
        torch.cuda.empty_cache()

    if rank == 0 and results:
        # Re-derive NUMBERS.md section 6 from the measurement rather than the estimate.
        bw = max(r["bus_gbps"] for r in results) * 1e9
        lora_bytes = 2 * 41.9e6 * 2       # 2 * Psi * bytes_per_param, r=16 on 8B
        full_bytes = 2 * 8.03e9 * 2       # same, full fine-tune
        print(f"\n  --- NUMBERS.md section 6, re-derived from YOUR measured {bw / 1e9:.1f} GB/s ---")
        print(f"  DDP step comms, LoRA r=16 on 8B  (Psi=42M):  "
              f"{lora_bytes / 1e6:6.0f} MB  ->  {1e3 * lora_bytes / bw:7.1f} ms   "
              f"-> negligible against a ~1 s step. DDP is correct.")
        print(f"  DDP step comms, full FT of 8B  (Psi=8.03B):  "
              f"{full_bytes / 1e9:6.1f} GB  ->  {full_bytes / bw:7.2f} s    "
              f"-> dominates the step. Sharding a big model over PCIe is a bad trade.")
        print("\n  => Treat GPU 1 as a second EXPERIMENT SLOT, not a second worker,")
        print("     unless the arithmetic says otherwise. Confirm in Phase 10.")

    dist.barrier()
    dist.destroy_process_group()
    return results


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--only",
        choices=["matmul", "bandwidth", "memory", "allreduce", "all"],
        default="all",
        help="run a single benchmark instead of the whole set",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", type=str, default=None, help="also write results to this path")
    args = parser.parse_args()

    # Already under torchrun? Then the only sensible job is the collective benchmark.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        bench_allreduce()
        return 0

    if not torch.cuda.is_available():
        print("CUDA is not available. This script must run on the A10 box, not the Mac.")
        return 1

    print(f"torch {torch.__version__} · CUDA {torch.version.cuda} "
          f"· {torch.cuda.device_count()} device(s)")
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"  [{i}] {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}")

    results: dict[str, object] = {}
    if args.only in ("matmul", "all"):
        results["matmul"] = bench_matmul(args.device)
    if args.only in ("bandwidth", "all"):
        results["bandwidth"] = bench_bandwidth(args.device)
    if args.only in ("memory", "all"):
        results["usable_memory"] = bench_usable_memory(args.device)

    if args.only in ("allreduce", "all"):
        if torch.cuda.device_count() < 2:
            print("\n=== 4. all-reduce — SKIPPED: fewer than 2 GPUs visible ===")
            print("    (check CUDA_VISIBLE_DEVICES, and that no other process holds a card)")
        else:
            print("\nRelaunching the all-reduce benchmark under torchrun (needs 2 processes)...")
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node=2", "--nnodes=1",
                "-m", "llm_finetuning_from_zero_to_hero.00_gpu_benchmark",
                "--only", "allreduce",
            ]
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                print(f"\n  torchrun exited {proc.returncode}. Run it by hand to see the error:")
                print(f"  {' '.join(cmd)}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")

    print("\n" + SEPARATOR)
    print("Now fill in the MEASURED columns in docs/HARDWARE.md section 3,")
    print("and the prediction-vs-measurement table in docs/phases/phase-00.md section 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
