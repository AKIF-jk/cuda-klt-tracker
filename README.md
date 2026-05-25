<div align="center">

<br/>

```
██╗  ██╗██╗  ████████╗    ██████╗ ██████╗ ██████╗ 
██║ ██╔╝██║  ╚══██╔══╝   ██╔════╝██╔════╝██╔══██╗
█████╔╝ ██║     ██║      ██║     ██║     ██████╔╝
██╔═██╗ ██║     ██║      ██║     ██║     ██╔═══╝ 
██║  ██╗███████╗██║      ╚██████╗╚██████╗██║     
╚═╝  ╚═╝╚══════╝╚═╝       ╚═════╝ ╚═════╝╚═╝     
```

# GPU-Accelerated KLT Feature Tracker

**Pushing a real-time computer vision pipeline from CPU baseline to 10.85× GPU speedup — across four iterative deliverables.**

<br/>

[![CUDA](https://img.shields.io/badge/CUDA-GPU_Kernels-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![OpenACC](https://img.shields.io/badge/OpenACC-Directive_Acceleration-00ADEF?style=for-the-badge)](https://www.openacc.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-Performance_Critical-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![HPC](https://img.shields.io/badge/HPC-FAST--NU_Server-FF6B35?style=for-the-badge)](https://github.com/AKIF-jk/HPC-CCP)

<br/>

> **Final speedup: `10.85×` (CUDA) · `1.32×` (OpenACC baseline) · Dataset: 100 frames @ 4K (3840×2160)**

<br/>

</div>

---

## The Problem

The **Kanade–Lucas–Tomasi (KLT) feature tracker** is a cornerstone of real-time computer vision — used in everything from autonomous vehicles to AR headsets. The original implementation? Pure CPU. At 4K resolution with 5000 features tracked across 100 frames, it was painfully slow.

This project takes that sequential C++ codebase and systematically dismantles every bottleneck — profiling, parallelizing, optimizing memory, and iterating — until the wall clock time drops by an order of magnitude.

---

## Results at a Glance

| Version | Approach | CPU Time | GPU Time | **Speedup** |
|---------|----------|----------|----------|-------------|
| D1 | Baseline CPU (profiling only) | 102.37s | — | `1.00×` |
| D2 | Naïve CUDA (4 functions offloaded) | 102.37s | 56.52s | `1.81×` |
| D3 | Optimized CUDA (shared + constant memory) | 81.29s | **7.49s** | **`10.85×`** |
| D4 | OpenACC rewrite | 47.74s | 36.17s | `1.32×` |

> D3 represents the peak — a **10.85× speedup** achieved through iterative GPU memory optimization.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  KLT Tracker Pipeline                    │
├──────────────┬──────────────┬──────────────┬────────────┤
│   D1: CPU    │  D2: Naïve   │  D3: Tuned   │ D4: OpenACC│
│   Profiling  │    CUDA      │    CUDA      │  Directives│
└──────┬───────┴──────┬───────┴──────┬───────┴─────┬──────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
  gprof +         4 kernels      Shared mem    #pragma acc
  gprof2dot       offloaded      Constant mem  parallel loop
  call graphs     to GPU         Coalesced     gang/vector
                                 access        layout
```

---

## Journey: Deliverable by Deliverable

### D1 — Know Before You Optimize

Before touching a single line of GPU code, we profiled exhaustively using `gprof` and `gprof2dot`. The call graph revealed the real culprits:

| Function | Self Time | Call Count | Parallelizable? |
|---|---|---|---|
| `_convolveImageVert` | ~38% | 63× | ✅ Yes |
| `_convolveImageHoriz` | ~8% | 63× | ✅ Yes |
| `_KLTSelectGoodFeatures` | ~23% | 1× | ✅ Yes |
| `_interpolate` | ~15% | 2,069,270× | ⚠️ Memory-bound |
| `_compute2x2GradientMatrix` | ~8% | 1,222,060× | ✅ Yes |

**Amdahl's Law** projected a theoretical maximum of ~13× speedup with 92.3% of the code parallelizable.

---

### D2 — First Contact with the GPU

We offloaded the four hottest functions to CUDA kernels. Initial results were... humbling.

```
GPU Version: 33.681s  ←  Slower than CPU (32.096s)
```

The `_interpolate` function was causing massive `cudaMemcpy` overhead — 99.8% of GPU memory time was just copying data, drowning out any compute gain. We removed `_interpolate` and `_compute2x2gradient` from GPU execution.

**After adjustment:**

```
30 frames  →  CPU: 32.095s  |  GPU: 18.507s  →  1.73×
100 frames →  CPU: 102.368s |  GPU: 56.517s  →  1.81×
```

Close to theoretical prediction (1.856×) for convolution alone. Validation that the approach was sound.

---

### D3 — Memory Mastery

With the GPU foothold established, we attacked every memory inefficiency:

**Problems identified:**
- Excessive `cudaMemcpy()` calls on every frame
- Uncoalesced memory access patterns
- Convolution kernel living in global memory (slow reads)
- Repeated `fwrite()` calls inflating total time

**Optimizations applied:**

```
1. Communication Reduction   →  Batch host↔device transfers
2. Coalesced Memory Access   →  Matrix transpose for convolveHoriz()
3. Shared Memory             →  Image frames cached on-chip
4. Constant Memory           →  Convolution kernel (read-only) moved to L1
5. Launch Config Tuning      →  (32,8) thread block after empirical sweep
```

**Result:**

```
Track features time — CPU: 81,289.998 ms
Track features time — GPU:  7,493.039 ms
                         ─────────────────
                Speedup:       10.85×  🚀
```

From 1.81× to 10.85× — a **6× improvement** from memory optimization alone.

---

### D4 — OpenACC: The Portability Tradeoff

As a research comparison, we rewrote the pipeline using **OpenACC** — replacing hand-written CUDA kernels with compiler directives. This explores the classic tradeoff: development velocity vs. raw performance.

**Key techniques:**
- `#pragma acc data` — persistent GPU residency for arrays (no repeated copies)
- `#pragma acc loop gang` — outer loop → CUDA thread blocks
- `#pragma acc loop vector` — inner loop → coalesced vector lanes
- `acc_malloc` for intermediate buffers, managed with `enter/exit data`

| Metric | OpenACC | Direct CUDA |
|--------|---------|-------------|
| Dev effort | Low (directives) | High (kernels, indexing) |
| Memory management | Unified / implicit | Manual `cudaMalloc`/`cudaMemcpy` |
| Parallelism control | Compiler-decided | Full warp-level control |
| Portability | High (multi-vendor) | NVIDIA-only |
| **Peak speedup** | **1.32×** | **10.85×** |

OpenACC achieved 1.32× on the same 4K, 100-frame, 5000-feature workload — demonstrating that when you need maximum throughput, there's no substitute for hand-tuned CUDA.

---

## Technical Stack

```
Language    C / C++
GPU API     CUDA (D2, D3) · OpenACC (D4)
Profiling   gprof · gprof2dot · NVIDIA Nsight Systems
Build       GNU Make (separate CPU/GPU rules per deliverable)
Hardware    FAST-NU HPC Server · NVIDIA GPU (8704 CUDA cores)
Dataset     3840×2160 (4K) image sequences · up to 5000 features
```

---

## Repository Structure

```
HPC-CCP/
├── D1/          ← CPU baseline + profiling Makefile rules
├── D2/          ← Naïve CUDA implementation
├── D3/          ← Optimized CUDA (shared + constant memory)
├── D4/          ← OpenACC implementation
└── README.md
```

Each deliverable compiles independently. Makefile includes rules for both CPU and GPU execution targets.

---

## Key Takeaways

- **Profile before parallelizing.** D1's gprof data prevented us from wasting effort on the wrong functions — and saved us from a dead-end path with `_interpolate`.
- **Memory is the bottleneck, not compute.** The jump from 1.81× (D2) to 10.85× (D3) came entirely from reducing data movement, not adding more GPU cores.
- **Abstraction has a cost.** OpenACC's 1.32× vs CUDA's 10.85× shows that portability and peak performance are genuinely in tension — the right choice depends on your constraints.
- **Empirical tuning matters.** Launch configuration (32,8) was chosen after testing multiple configurations — there's no substitute for measurement.

---

## Team

**Mohammad Faran Azam** (23i-0075) · **Saim Ahmad** (23i-0781) · **Akif Jawad** (23i-0583)

FAST–NUCES · High Performance Computing with GPUs · Fall 2025

---

<div align="center">

*Built at the intersection of parallel algorithms and low-level GPU memory architecture.*

</div>
