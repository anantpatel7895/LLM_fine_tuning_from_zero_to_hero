# CPU RAM vs GPU Memory vs Disk Storage

## 1. Overview

A modern computer has several different types of memory and storage.

For AI/ML and especially LLM inference, three are particularly important:

1. **CPU RAM (System Memory)**
2. **GPU Memory (VRAM / HBM)**
3. **Disk Storage (SSD / HDD)**

They are different in:

* Where they physically exist
* Who primarily uses them
* Capacity
* Bandwidth
* Latency
* Cost
* Purpose
* How data moves between them

A useful simplified picture is:

```text
                    COMPUTER
                       |
          +------------+------------+
          |                         |
         CPU                       GPU
          |                         |
       CPU RAM                 GPU Memory
       (DRAM)                 (VRAM / HBM)
          |                         |
          +------------+------------+
                       |
                 SSD / HDD
                  Storage
```

For LLM inference, an even more useful picture is:

```text
                  SSD / HDD
                Model on disk
                     |
                     | Load
                     v
                  CPU RAM
                     |
                     | Transfer
                     | PCIe / NVLink
                     v
                GPU HBM/VRAM
                     |
                     | Read weights,
                     | KV cache, tensors
                     v
                GPU Compute
                CUDA Cores /
                Tensor Cores
                     |
                     v
                   Output
```

---

# 2. First Important Concept: Memory vs Storage

Before discussing CPU RAM and GPU memory, we need to distinguish **memory** from **storage**.

## Memory

Memory is used to hold data that the computer is actively working with.

Examples:

* CPU RAM
* GPU HBM
* GPU cache
* CPU cache
* Registers

Memory is generally much faster than persistent storage.

---

## Storage

Storage is used to persist data.

Examples:

* SSD
* HDD
* NVMe SSD

If you shut down your computer:

```text
RAM
```

loses its normal working contents.

But:

```text
SSD
```

keeps your files.

For example:

```text
SSD
|
+-- model.safetensors
+-- tokenizer.json
+-- config.json
+-- Python files
+-- Operating system
```

---

# 3. CPU RAM

CPU RAM is usually called:

* RAM
* System RAM
* Main memory
* DRAM

Suppose your machine has:

```text
32 GB RAM
```

This is memory available primarily to the CPU and the operating system.

For example:

```text
CPU
 |
 +----> CPU RAM
          |
          +-- Python objects
          +-- DataFrames
          +-- OS processes
          +-- Application data
          +-- Model data
```

---

# 4. What Does CPU RAM Do?

Suppose you run:

```python
x = [1, 2, 3, 4, 5]
```

The Python process needs memory.

The data and Python runtime are stored in system memory.

Conceptually:

```text
SSD
 |
 | Python program
 v
CPU RAM
 |
 +-- Python interpreter
 +-- Variables
 +-- Objects
 +-- Application data
```

The CPU accesses this RAM while executing the program.

---

# 5. GPU Memory

GPU memory is memory directly associated with the GPU.

Depending on the type of GPU, you may hear:

* VRAM
* GPU memory
* HBM
* GDDR
* HBM2e
* HBM3
* HBM3e

For modern data-center AI GPUs, HBM is extremely important.

For example, conceptually:

```text
H100
 |
 +-- GPU compute
 |
 +-- HBM
```

The GPU uses this memory to hold data needed for GPU computation.

For LLMs, this can include:

```text
GPU HBM
 |
 +-- Model weights
 |
 +-- Activations
 |
 +-- KV cache
 |
 +-- Intermediate tensors
 |
 +-- Temporary buffers
```

---

# 6. Why Does the GPU Need Its Own Memory?

GPUs perform enormous amounts of parallel computation.

They need extremely high data throughput.

Imagine:

```text
GPU Compute
   |
   | "Give me data!"
   |
   v
GPU Memory
```

If the GPU had to depend entirely on slower external storage for every operation, it would be unusably slow.

Instead, frequently used data is placed in GPU memory.

For an LLM:

```text
SSD
 |
 | Load model
 v
GPU HBM
 |
 | Read weights
 v
Tensor Cores
 |
 | Compute
 v
Next token
```

---

# 7. GPU Memory Is Not the Same as CPU RAM

This is one of the most important concepts.

Suppose your computer has:

```text
CPU RAM = 64 GB
GPU HBM = 80 GB
```

You do NOT have:

```text
64 + 80 = 144 GB
```

of one unified pool of ordinary RAM.

They are separate memory resources.

Conceptually:

```text
              CPU
               |
          CPU RAM
           64 GB


              GPU
               |
          GPU HBM
           80 GB
```

Data can be transferred between them, but they are different memory systems.

---

# 8. GPU Memory vs CPU RAM

A simplified comparison:

| Property                 | CPU RAM               | GPU Memory     |
| ------------------------ | --------------------- | -------------- |
| Primary user             | CPU                   | GPU            |
| Typical technology       | DRAM                  | HBM / GDDR     |
| Purpose                  | General system memory | GPU workloads  |
| Capacity                 | Often larger          | Often smaller  |
| Bandwidth                | High                  | Extremely high |
| Used for LLM weights     | Sometimes             | Very commonly  |
| Used for KV cache        | Sometimes             | Commonly       |
| Directly attached to GPU | No                    | Yes            |

The exact specifications vary significantly by hardware.

---

# 9. Disk Storage

Disk storage is different from both CPU RAM and GPU memory.

Common examples:

```text
SSD
NVMe SSD
HDD
```

Suppose you download a model:

```text
llama-model.safetensors
```

The model initially exists on storage:

```text
SSD
 |
 +-- llama-model.safetensors
```

The model is not automatically sitting inside GPU memory just because the file exists on your machine.

---

# 10. Example: Loading an LLM

Suppose you have:

```text
model.safetensors
```

stored on your SSD.

The simplified flow is:

```text
             SSD
              |
              | read file
              v
           CPU RAM
              |
              | transfer to GPU
              v
          GPU HBM
              |
              | computation
              v
          GPU cores
```

This is a critical mental model.

---

# 11. Three Different Questions

When dealing with memory, always ask three different questions.

## Question 1: How much can I store?

This is:

```text
Memory capacity
```

Example:

```text
80 GB HBM
```

---

## Question 2: How quickly can I move data?

This is:

```text
Memory bandwidth
```

Example:

```text
3.35 TB/s
```

---

## Question 3: How quickly can I perform calculations?

This is:

```text
Compute throughput
```

Example:

```text
989 TFLOPS
```

These are completely different concepts.

---

# 12. Capacity vs Bandwidth

This distinction is extremely important.

## Capacity

Capacity means:

> How much data can fit?

Example:

```text
GPU HBM = 80 GB
```

It can hold approximately 80 GB of data.

---

## Bandwidth

Bandwidth means:

> How much data can be transferred per second?

Example:

```text
3.35 TB/s
```

means the memory system can theoretically provide data at a rate of roughly 3.35 TB per second under the specified conditions.

---

# 13. Warehouse Analogy

Imagine a warehouse.

## Capacity

The warehouse can hold:

```text
80,000 boxes
```

That is its capacity.

---

## Bandwidth

The warehouse has trucks.

Suppose the trucks can move:

```text
3,350 boxes/second
```

That is analogous to bandwidth.

So:

```text
Capacity
= How much is inside?

Bandwidth
= How quickly can I move it?
```

This analogy is very useful for GPU memory.

---

# 14. Memory Bandwidth

Memory bandwidth is:

> The rate at which data can be transferred between a memory system and the compute system.

It is commonly measured in:

```text
GB/s
TB/s
```

For example:

```text
3.35 TB/s
```

means approximately:

```text
3.35 trillion bytes/second
```

of theoretical memory bandwidth.

---

# 15. Why GPU Memory Has Very High Bandwidth

GPUs perform massive parallel computation.

Imagine:

```text
GPU
 |
 +-- Tensor Core 1
 +-- Tensor Core 2
 +-- Tensor Core 3
 +-- Tensor Core 4
 +-- ...
 +-- Thousands of parallel execution resources
```

All these compute resources need data.

Therefore:

```text
Huge compute capability
        +
Huge data requirements
        =
Need very high memory bandwidth
```

This is one reason GPUs use technologies such as HBM.

---

# 16. The H100 Example

Consider the simplified numbers:

```text
H100 FP16 compute
≈ 989 TFLOPS

H100 memory bandwidth
≈ 3.35 TB/s
```

These describe two different resources.

### Compute

```text
989 TFLOPS
```

means:

> How many floating-point operations the GPU can perform per second.

### Memory bandwidth

```text
3.35 TB/s
```

means:

> How quickly data can be supplied to/from GPU memory.

---

# 17. Connecting Compute and Memory Bandwidth

We can calculate a useful ratio:

$$
\frac{989\ TFLOPS}{3.35\ TB/s}
$$

Approximately:

$$
295\ FLOPs/byte
$$

This is the approximate **machine balance** for the stated configuration.

It tells us the arithmetic intensity required to keep the compute capability fully utilized from a bandwidth perspective.

---

# 18. What Is Arithmetic Intensity?

Arithmetic intensity is:

$$
AI =
\frac{FLOPs}{Bytes\ moved}
$$

Example:

Suppose a workload moves:

```text
1 GB
```

and performs:

```text
100 billion FLOPs
```

Then:

$$
AI =
\frac{100 \times 10^9}
{1 \times 10^9}
$$

Therefore:

$$
AI = 100\ FLOPs/byte
$$

---

# 19. Memory-Bound Workload

Suppose:

```text
Arithmetic intensity = 50 FLOPs/byte
```

H100:

```text
Memory bandwidth = 3.35 TB/s
```

Maximum performance based on memory bandwidth:

$$
3.35 \times 50
=
167.5\ TFLOPS
$$

But H100 can theoretically provide:

```text
989 TFLOPS
```

Therefore, the workload cannot feed the compute units fast enough.

The bottleneck is memory.

```text
Memory
  |
  |  not enough data
  v
GPU Compute
  |
  | waiting
```

This is:

```text
MEMORY-BOUND
```

---

# 20. Compute-Bound Workload

Suppose:

```text
Arithmetic intensity = 500 FLOPs/byte
```

Then:

$$
3.35 \times 500
=
1675\ TFLOPS
$$

But the GPU can only compute:

```text
989 TFLOPS
```

Therefore:

```text
Compute = bottleneck
```

This is:

```text
COMPUTE-BOUND
```

---

# 21. The Roofline Equation

A useful simplified equation is:

$$
Performance =
\min(
Peak\ Compute,
Memory\ Bandwidth \times Arithmetic\ Intensity
)
$$

For our example:

$$
Performance =
\min(
989,
3.35 \times AI
)
$$

where performance is measured in TFLOPS if the units are handled consistently.

The balance point occurs around:

$$
AI \approx 295\ FLOPs/byte
$$

---

# 22. Disk Storage Is Different Again

Now consider an SSD.

Suppose:

```text
SSD capacity = 2 TB
```

This tells us:

> The SSD can permanently store approximately 2 TB of data.

It does NOT mean:

```text
2 TB/s bandwidth
```

Capacity and bandwidth are different.

For example:

```text
SSD
Capacity: 2 TB
Bandwidth: perhaps several GB/s
```

The exact bandwidth depends on the SSD and workload.

---

# 23. Why SSD Is Not Used Directly for Every GPU Operation

Suppose your model is:

```text
100 GB
```

stored on SSD.

You don't want the GPU doing:

```text
GPU → SSD → GPU → SSD → GPU
```

for every calculation.

That would be extremely slow compared with keeping frequently accessed data in GPU memory.

Instead:

```text
SSD
 |
 | Load
 v
CPU RAM
 |
 | Transfer
 v
GPU HBM
 |
 | Repeated access
 v
GPU Compute
```

---

# 24. Data Movement Between CPU and GPU

When CPU RAM and GPU memory are separate, data needs a connection between them.

A simplified architecture:

```text
CPU
 |
 CPU RAM
 |
 |
 +-------- PCIe / high-speed interconnect --------+
                                                   |
                                                   v
                                                  GPU
                                                   |
                                                GPU HBM
```

The transfer bandwidth of this connection is different from the GPU HBM bandwidth.

This distinction is critical.

---

# 25. GPU HBM Bandwidth vs PCIe Bandwidth

Suppose:

```text
GPU HBM bandwidth
≈ 3.35 TB/s
```

That does NOT mean:

```text
CPU RAM → GPU
= 3.35 TB/s
```

No.

There are different links.

```text
CPU RAM
   |
   | PCIe / interconnect
   | <-- different bandwidth
   |
   v
GPU
   |
   | HBM
   | <-- GPU memory bandwidth
   |
   v
GPU Compute
```

Therefore, remember:

```text
CPU RAM bandwidth
        ≠
CPU-GPU interconnect bandwidth
        ≠
GPU HBM bandwidth
```

---

# 26. PCIe

PCIe is a high-speed connection used to connect components such as:

* GPU
* SSD
* Network devices

A simplified view:

```text
CPU
 |
 +---- PCIe ---- GPU
 |
 +---- PCIe ---- NVMe SSD
```

PCIe provides a data path.

Its bandwidth is much lower than the aggregate bandwidth of high-end HBM systems.

Therefore, moving huge amounts of data repeatedly between CPU RAM and GPU memory can become a bottleneck.

---

# 27. NVLink

High-end NVIDIA GPU systems can also use specialized high-speed GPU interconnects such as NVLink.

This is especially relevant in multi-GPU systems.

Conceptually:

```text
GPU 0 HBM
    |
    | NVLink
    |
GPU 1 HBM
```

This allows GPUs to exchange data through a high-bandwidth interconnect.

Again:

```text
HBM bandwidth
```

and:

```text
NVLink bandwidth
```

are different things.

---

# 28. Multi-GPU LLM Inference

Suppose an LLM is too large for one GPU.

For example:

```text
Model = 140 GB

GPU 0 = 80 GB
GPU 1 = 80 GB
```

The model can potentially be distributed across GPUs.

Conceptually:

```text
             LLM
              |
       +------+------+
       |             |
       v             v
     GPU 0         GPU 1
     80 GB         80 GB
       |             |
      HBM           HBM
```

Now another important resource appears:

```text
GPU 0 <---- Interconnect ----> GPU 1
```

The communication between GPUs can affect performance.

---

# 29. LLM Model Weights

Let's connect all of this to LLMs.

Suppose:

```text
Parameters = 70 billion
```

Using FP16:

```text
FP16 = 2 bytes
```

Approximate weight memory:

$$
70B \times 2
=
140GB
$$

So:

```text
70B model
≈ 140 GB of FP16 weights
```

This is an approximate calculation and does not include additional memory required by the inference system.

---

# 30. Where Does the 140 GB Exist?

Initially:

```text
SSD
 |
 +-- model files
```

After loading:

```text
CPU RAM
 |
 +-- model data
```

During GPU inference:

```text
GPU HBM
 |
 +-- model weights
```

Potentially across multiple GPUs:

```text
GPU 0 HBM
 |
 +-- part of model

GPU 1 HBM
 |
 +-- another part
```

---

# 31. LLM Inference Memory

GPU memory isn't only model weights.

A more complete simplified picture is:

```text
GPU HBM
|
+-- Model weights
|
+-- KV cache
|
+-- Activations
|
+-- Intermediate tensors
|
+-- CUDA/runtime allocations
|
+-- Temporary buffers
```

Therefore:

```text
Required GPU memory
>
Model weight size
```

in many real inference scenarios.

---

# 32. KV Cache

During autoregressive generation, the model maintains a **KV cache**.

Conceptually:

```text
Prompt
  |
  v
Transformer
  |
  +---- Keys
  |
  +---- Values
          |
          v
       KV Cache
```

The KV cache can consume significant GPU memory, especially with:

* Long context
* Large batch size
* Large models
* Many concurrent requests

Therefore GPU memory capacity affects how much context and concurrency you can support.

---

# 33. Model Size vs GPU Memory

Suppose:

```text
Model weights = 140 GB
```

and:

```text
GPU HBM = 80 GB
```

The entire model cannot fit on one 80 GB GPU.

Possible approaches include:

```text
1. Quantization
2. Multiple GPUs
3. Model sharding
4. CPU offloading
5. Other memory optimization techniques
```

Each approach has tradeoffs.

---

# 34. Quantization and Memory

Suppose the same 70B model is stored approximately as:

### FP16

```text
2 bytes/parameter

70B × 2
≈ 140 GB
```

### INT8

```text
1 byte/parameter

70B × 1
≈ 70 GB
```

### 4-bit

Approximately:

```text
0.5 bytes/parameter

70B × 0.5
≈ 35 GB
```

These are simplified weight-only calculations and actual memory usage is higher because of metadata, scales, runtime buffers, KV cache, and other allocations.

But the important idea is:

```text
Lower precision
       |
       v
Smaller model weights
       |
       v
Less memory required
       |
       v
Less data to move
```

This is one reason quantization can be valuable for inference.

---

# 35. Why Quantization Can Help Memory-Bound Inference

Suppose the model's weights need to be read repeatedly.

If:

```text
FP16
= 2 bytes/parameter
```

and you reduce them to:

```text
4-bit
≈ 0.5 bytes/parameter
```

then substantially less weight data needs to be transferred.

Simplified:

```text
FP16
████████████████

4-bit
████
```

Less data movement can improve performance when memory bandwidth is the bottleneck.

However, actual performance depends on the implementation, kernels, hardware support, dequantization overhead, batching, and workload.

---

# 36. Why LLM Decode Can Be Memory-Bound

Consider autoregressive generation.

The model generates:

```text
Token 1
  ↓
Token 2
  ↓
Token 3
  ↓
Token 4
  ↓
...
```

For each decoding step, the model performs computation using its weights and cached state.

A simplified view:

```text
GPU HBM
   |
   | model weights / cache
   v
GPU Compute
   |
   v
Next token
```

For some decode workloads, the GPU's compute capability is not the main limiting factor.

The limiting factor can instead be:

```text
How quickly can the required data be supplied?
```

Therefore:

```text
Memory-bound
```

can become an important characteristic of LLM decoding.

---

# 37. Prefill vs Decode

LLM inference is often discussed in two phases.

## Prefill

The prompt is processed.

```text
Prompt
   |
   v
Transformer
   |
   v
KV Cache
```

There can be substantial parallel computation over the prompt tokens.

---

## Decode

New tokens are generated one at a time.

```text
Token
  |
  v
Transformer
  |
  v
Next token
  |
  v
Transformer
  |
  v
Next token
```

Decode can have different arithmetic intensity and memory behavior from prefill.

Therefore:

```text
Prefill performance
```

and:

```text
Decode performance
```

should not automatically be assumed to have the same bottleneck.

---

# 38. Memory Hierarchy

A modern system has multiple layers of memory.

A simplified hierarchy:

```text
                Fastest
                   |
              Registers
                   |
              L1 / L2 Cache
                   |
              GPU Cache
                   |
              GPU HBM
                   |
               CPU RAM
                   |
              NVMe SSD
                   |
                HDD
                   |
                Slowest
```

This is a conceptual hierarchy.

Each level differs in:

* Capacity
* Latency
* Bandwidth
* Cost
* Physical location

---

# 39. Latency vs Bandwidth

These two concepts should not be confused.

## Latency

Latency means:

> How long does it take for a particular memory request to begin returning/complete?

Think:

```text
"How long do I wait?"
```

---

## Bandwidth

Bandwidth means:

> How much data can be transferred per second?

Think:

```text
"How much data can I move per second?"
```

Analogy:

```text
Latency = How long until the truck arrives?

Bandwidth = How many trucks can pass per second?
```

A highway can have:

```text
High latency
+
High bandwidth
```

and another connection can have:

```text
Low latency
+
Lower bandwidth
```

They are different characteristics.

---

# 40. A Complete LLM Data Path

Let's put everything together.

Suppose you download a model.

### Step 1 — Model on SSD

```text
NVMe SSD
|
+-- model.safetensors
```

Storage is persistent.

---

### Step 2 — Load model

```text
SSD
 |
 | read
 v
CPU RAM
```

The operating system/application reads the model into system memory.

---

### Step 3 — Transfer model to GPU

```text
CPU RAM
 |
 | PCIe / high-speed interconnect
 v
GPU
```

---

### Step 4 — Store model in HBM

```text
GPU HBM
|
+-- Model weights
+-- KV cache
+-- Activations
+-- Buffers
```

---

### Step 5 — GPU computation

```text
GPU HBM
   |
   | data
   v
Tensor Cores / CUDA cores
   |
   | computation
   v
Output
```

---

# 41. The Most Important Bandwidths to Distinguish

When working with AI infrastructure, you may encounter several different bandwidths.

```text
1. CPU RAM bandwidth

2. CPU ↔ GPU interconnect bandwidth
   e.g. PCIe

3. GPU HBM bandwidth

4. GPU ↔ GPU interconnect bandwidth
   e.g. NVLink

5. SSD bandwidth

6. Network bandwidth
   e.g. Ethernet / InfiniBand
```

They are not interchangeable.

---

# 42. Example Architecture

A simplified multi-GPU server:

```text
                       Server
                         |
          +--------------+--------------+
          |                             |
        CPU                           GPU
          |                             |
      CPU RAM                     +-----+-----+
                                  |           |
                                GPU 0       GPU 1
                                  |           |
                                HBM         HBM
                                  |           |
                                  +-----+-----+
                                        |
                                     NVLink
                                        
                         |
                       NVMe
                       SSD
```

Now you can identify several different data paths.

---

# 43. Why Network Bandwidth Also Matters

In distributed inference or training, GPUs may be located in different machines.

Then:

```text
GPU 0
 |
 +-- Network / InfiniBand
 |
GPU 1
```

The network becomes another potential bottleneck.

So a distributed AI system may involve:

```text
SSD
 ↓
CPU RAM
 ↓
PCIe
 ↓
GPU HBM
 ↓
GPU compute
 ↓
NVLink
 ↓
Another GPU
 ↓
Network
 ↓
Another server
```

Each connection has its own performance characteristics.

---

# 44. A Practical Mental Model

When analyzing an AI workload, ask:

### Question 1

Where is the data?

```text
SSD?
CPU RAM?
GPU HBM?
GPU cache?
```

### Question 2

Where does the computation happen?

```text
CPU?
GPU?
Tensor Cores?
```

### Question 3

How much data needs to move?

```text
Bytes
```

### Question 4

How much computation is performed?

```text
FLOPs
```

### Question 5

What is the bandwidth of the relevant connection?

```text
GB/s
TB/s
```

### Question 6

What is the arithmetic intensity?

$$
\frac{FLOPs}{Bytes}
$$

This helps determine whether the workload is likely to be:

```text
Memory-bound
```

or:

```text
Compute-bound
```

---

# 45. Summary Table

| Component     | Main Purpose              | Example       | Persistent? | Important Metric  |
| ------------- | ------------------------- | ------------- | ----------- | ----------------- |
| CPU Registers | Immediate CPU computation | CPU registers | No          | Latency           |
| CPU Cache     | Fast CPU data access      | L1/L2/L3      | No          | Latency/bandwidth |
| CPU RAM       | General system memory     | DDR5          | No          | GB/s              |
| GPU Registers | Immediate GPU computation | Register file | No          | Latency           |
| GPU Cache     | Fast GPU data access      | L1/L2         | No          | Latency/bandwidth |
| GPU HBM       | High-bandwidth GPU memory | HBM3/HBM3e    | No          | TB/s              |
| GPU VRAM      | GPU memory                | GDDR/HBM      | No          | GB/s or TB/s      |
| SSD           | Persistent storage        | NVMe SSD      | Yes         | GB/s              |
| HDD           | Persistent storage        | Hard disk     | Yes         | MB/s              |

---

# 46. Interview Questions

## Q1. Is GPU memory the same as CPU RAM?

No.

They are separate memory resources.

```text
CPU → CPU RAM

GPU → GPU HBM/VRAM
```

Data can be transferred between them through an interconnect.

---

## Q2. Is GPU memory the same as SSD?

No.

GPU memory is high-speed working memory.

SSD is persistent storage.

```text
GPU HBM
= working memory

SSD
= persistent storage
```

---

## Q3. What is memory bandwidth?

Memory bandwidth is the rate at which data can be transferred between a memory system and the processor/compute system.

Usually measured in:

```text
GB/s
TB/s
```

---

## Q4. What is memory capacity?

Memory capacity is the amount of data that can be stored.

Example:

```text
80 GB HBM
```

---

## Q5. What is the difference between capacity and bandwidth?

```text
Capacity
= How much data can fit?

Bandwidth
= How quickly can data move?
```

---

## Q6. What does 3.35 TB/s mean?

It represents approximately:

```text
3.35 trillion bytes/second
```

of memory bandwidth under the relevant specification/conditions.

It does not mean the GPU has 3.35 TB of memory.

---

## Q7. Does 80 GB HBM mean 80 GB/s?

No.

These are completely different:

```text
80 GB
```

= capacity

while:

```text
3.35 TB/s
```

= bandwidth

---

## Q8. What is arithmetic intensity?

$$
AI =
\frac{FLOPs}{Bytes\ moved}
$$

It measures how much computation is performed relative to data movement.

---

## Q9. What is a memory-bound workload?

A workload where performance is primarily limited by data movement/memory bandwidth rather than compute throughput.

---

## Q10. What is a compute-bound workload?

A workload where the compute hardware is the primary bottleneck.

---

# 47. The Mental Picture to Remember

The most important diagram from this entire note is:

```text
                         COMPUTER
                            |
             +--------------+--------------+
             |                             |
            CPU                           GPU
             |                             |
         CPU RAM                       GPU HBM
             |                             |
             |                             |
             +-------- Interconnect -------+
                       PCIe/NVLink
                            |
                            |
                           GPU
                         Compute
                            |
                            v
                         Output


             SSD / NVMe
                  |
                  | persistent storage
                  |
                  v
               CPU RAM
                  |
                  v
               GPU HBM
                  |
                  v
              GPU Compute
```

And remember these three questions:

```text
                 MEMORY

       ┌────────────────────────┐
       │                        │
       │  CAPACITY              │
       │  "How much?"            │
       │                        │
       │  BANDWIDTH             │
       │  "How fast?"            │
       │                        │
       │  LATENCY               │
       │  "How long to wait?"    │
       │                        │
       └────────────────────────┘
```

For LLM/GPU performance, the next layer is:

```text
             Data
              |
              v
        Memory Bandwidth
              |
              v
        Arithmetic Intensity
              |
              v
       +------+------+
       |             |
 Memory-bound   Compute-bound
       |             |
       v             v
  Optimize data   Optimize
    movement      computation
```

---

# 48. Final Mental Model

If you remember only one thing, remember this:

```text
SSD
"Where is my model permanently stored?"
        |
        v
CPU RAM
"Where is my program/data while the CPU is working?"
        |
        | PCIe / interconnect
        v
GPU HBM
"Where is the data the GPU needs at very high speed?"
        |
        | Very high bandwidth
        v
GPU Compute
"Where are the mathematical operations performed?"
```

And:

```text
Capacity  → How much data can I hold?
Bandwidth → How much data can I move per second?
Latency   → How long do I wait for data?
Compute   → How many operations can I perform per second?
```

For LLM inference, these four concepts—**capacity, bandwidth, latency, and compute**—are fundamental to understanding why a model fits or doesn't fit on a GPU, why inference can be memory-bound, why quantization helps, why batching changes performance, and why multi-GPU systems need fast interconnects.
