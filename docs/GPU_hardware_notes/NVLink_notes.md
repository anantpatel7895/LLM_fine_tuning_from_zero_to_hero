# NVLink — Simple Explanation

## What is NVLink?

**NVLink is a high-speed interconnect used mainly for communication between GPUs.**

```text
GPU 0                         GPU 1
┌──────────────┐              ┌──────────────┐
│ HBM / Memory │              │ HBM / Memory │
│   Compute    │              │   Compute    │
└──────┬───────┘              └──────┬───────┘
       │                              │
       └────────── NVLink ────────────┘
```

---

## Is NVLink between GPU memory and GPU compute?

**No.**

Communication between a GPU's HBM and its compute units happens through the GPU's internal memory/fabric system.

```text
GPU
├── HBM
└── Compute
     ↑
 Internal GPU connection
```

---

## Is NVLink between CPU RAM and GPU memory?

**Generally, no.**

CPU-to-GPU communication commonly uses **PCIe** or other platform-specific interconnects.

```text
CPU RAM
   │
   │ PCIe
   ▼
  GPU
   │
  HBM
```

---

# Real-World Example: Running a Large LLM

Suppose you have a **140 GB LLM** and two GPUs:

```text
GPU 0 → 80 GB HBM
GPU 1 → 80 GB HBM
```

The model is too large for one GPU, so we split it across both GPUs.

```text
                 140 GB LLM
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
         GPU 0                GPU 1
         70 GB                70 GB
           │                    │
          HBM                  HBM
```

Now suppose GPU 0 needs some data that GPU 1 has.

The GPUs need to communicate:

```text
GPU 0                         GPU 1
┌──────────────┐              ┌──────────────┐
│ HBM          │              │ HBM          │
│              │              │              │
│ Compute      │              │ Compute      │
└──────┬───────┘              └──────┬───────┘
       │                              │
       └────────── NVLink ────────────┘
                    ↑
             Data exchange
```

**NVLink provides the high-speed connection between the GPUs.**

---

## Why is this useful?

Imagine:

```text
GPU 0: "I need some data from GPU 1."
```

Instead of going through:

```text
GPU 0
  ↓
CPU
  ↓
CPU RAM
  ↓
CPU
  ↓
GPU 1
```

the GPUs can communicate directly through the GPU interconnect:

```text
GPU 0
  │
  │ NVLink
  ▼
GPU 1
```

This can provide much faster GPU-to-GPU communication than routing data through the CPU path.

---

# Simple Mental Model

Think of a multi-GPU server as two factories:

```text
Factory 1                    Factory 2
 GPU 0                        GPU 1
  │                             │
  │       High-speed road       │
  └────────── NVLink ───────────┘
```

Each factory has its own warehouse:

```text
GPU 0 → HBM 0
GPU 1 → HBM 1
```

NVLink is the **high-speed road between the two factories**.

It is **not** the road from the warehouse to the machines inside each factory.

---

# Remember

```text
CPU RAM → GPU
    ↓
  PCIe / platform interconnect

GPU HBM → GPU Compute
    ↓
  Internal GPU fabric

GPU → GPU
    ↓
  NVLink
```

### One-line definition

> **NVLink is a high-speed GPU-to-GPU interconnect that allows multiple GPUs to exchange data efficiently, which is particularly important when large AI models are distributed across multiple GPUs.**
