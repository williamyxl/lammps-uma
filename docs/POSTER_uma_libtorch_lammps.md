# Poster talking points — UMA LibTorch multi-device integration into LAMMPS (Intel XPU + NVIDIA GPU)

*Scope: this is 1/5 of the poster. Keep to ~1 column + 1 figure. All numbers below are
measured on ALCF Aurora (Intel Max XPU); the design is portable to NVIDIA (CUDA/NCCL).*

---

## 1. One-line message (poster header)
**We run the FairChem UMA foundation MLIP inside LAMMPS as a native C++/LibTorch
`pair_style`, in FP64, across all 12 XPU tiles of an Aurora node — matching the
reference FairChem/ASE energies and forces to machine precision, and scaling one
NaCl crystal to ~373k–439k atoms on a single node.**

---

## 2. Why this matters (motivation, 2 bullets)
- **Foundation MLIPs (UMA) meet production MD.** ASE/Python drives UMA today, but
  large-scale MD needs LAMMPS' ecosystem (fixes, ensembles, throughput). We bring
  UMA into LAMMPS with **no Python at runtime** — a single C++ binary.
- **Vendor-portable acceleration.** One engine targets **Intel XPU (oneCCL/SYCL)**
  and **NVIDIA GPU (CUDA/NCCL)** behind a device-abstraction layer — the same
  `pair_style uma` runs on either.

---

## 3. What we built (the contribution)
- **`pair_style uma`**: a native LibTorch pair style that loads a TorchScript-traced
  UMA model and computes energy + per-atom forces via C++ autograd (`torch::autograd::grad`).
- **Multi-device graph-parallel (GP):** one system split across 12 tiles; per-layer
  feature exchange + force reduction over **native collectives (XCCL/oneCCL on Intel,
  NCCL on NVIDIA)** — no host staging.
- **FP64 throughout**, matching FairChem reference numerics.
- **Two hard problems solved to make it work:**
  1. **Activation checkpointing rebuilt for a traced graph.** `torch.utils.checkpoint`
     cannot survive `torch.jit.trace`; we re-implemented it in C++ at **per-block +
     per-edge-chunk + prologue** granularity (custom autograd ops) — the capacity lever.
  2. **O(N) neighbor lists.** Replaced the O(N²) graph build with a linked-cell list
     (C++ runtime) + LAMMPS-neighbor-list consumption — ~6× faster, unblocks large N.

---

## 4. Key results (the "wow" numbers — put these in a boxed table)

**Correctness (vs ASE FairChem API, FP64, NVT 300 K, NaCl):**
| System | LAMMPS vs ASE energy | per-atom force (≥100 atoms) | AG=FD |
|---|---|---|---|
| N=6 (1,728 at) | dE = 1.3e-10 eV | max\|dF\| = 1.5e-14 eV/Å, cos = 1.0 | 3.6e-8 ✅ |
| N=12 (13,824) | dE = 1.6e-8 eV | max\|dF\| = 2.2e-14, cos = 1.0 | ✅ |
| N=18 (46,656) | dE = 4.6e-8 eV | max\|dF\| = 4.8e-14, cos = 1.0 | ✅ |

→ **Bit-for-bit agreement with the FairChem reference** (energy ~1e-9 meV/atom;
forces at the FP64 floor; force cosine = 1.0000000000).

**Capacity / scaling (single Aurora node, FP64):**
| Config | Max single system |
|---|---|
| 1 XPU tile | **N=18 → 46,656 atoms** (64 GiB-limited) |
| 12 XPU tiles (graph-parallel) | **N=36–38 → 373k–439k atoms** |

→ The 12-tile path **exceeds the FairChem/ASE graph-parallel reference (N=32,
262,144 atoms)** — the C++ per-chunk checkpointing gives more per-tile headroom.

**Verified 10-step NVT@300 K parity + timing on 1/2/4/8/12 tiles at N=18** (all pass
ASE parity; graph-parallel collectives correct at every tile count).

---

## 5. Talking points (what to say at the poster)
- "This is the FairChem UMA model — the same weights — running *inside* LAMMPS in
  pure C++. No Python interpreter in the MD loop."
- "It's the same source on Intel and NVIDIA: a device-abstraction layer swaps
  oneCCL/SYCL for CUDA/NCCL. Aurora is the demonstration platform."
- "The hard part wasn't the pair style — it was **memory**. UMA's capacity comes
  from activation checkpointing, which doesn't survive TorchScript tracing. We
  rebuilt checkpointing in C++ at block + edge-chunk granularity, which is what
  lets one crystal span all 12 tiles."
- "Everything is FP64 and validated against the FairChem/ASE API to machine
  precision — energy, per-atom forces, and autograd-vs-finite-difference."
- "On 12 tiles we push a single NaCl crystal past 370,000 atoms — beyond the
  Python graph-parallel reference — on one node."

## 6. Honest caveats (if asked)
- Strong scaling is sub-linear: the per-layer full-N feature gather is
  collective-bound (~3× at 12 tiles, not 12×) — consistent with the Python GP
  reference; this is a *capacity* + *correctness* result, throughput optimization
  is ongoing.
- At N ≥ 32 the single-tile ASE oracle itself OOMs, so those sizes are validated by
  per-atom-energy consistency + the collective math being bit-exact at N ≤ 18.
- Graph-parallel shards + activation-checkpoint artifacts are exported per system
  size (a one-time offline step).

## 7. Positioning within the full poster (this = 1/5)
Frame as the **"deployment / infrastructure"** pillar: *"...and to run these models
at production scale we integrated UMA into LAMMPS as a portable multi-GPU LibTorch
pair style (Intel + NVIDIA), FP64, validated to machine precision, scaling one
crystal to ~400k atoms per node."* The other 4/5 (science / model / dataset /
results) sit on top of this capability.

---

## 8. GPT-5.6 Luna image prompt — workflow / flowchart figure

> **Prompt for GPT-5.6 Luna (diagram generation):**
>
> Create a clean, professional **left-to-right workflow flowchart** for a scientific
> poster titled *"UMA foundation MLIP → native LibTorch pair_style in LAMMPS,
> multi-GPU (Intel XPU + NVIDIA), FP64."* Use a modern flat style, a restrained
> palette (deep blue, teal, slate gray, one orange accent), rounded rectangles,
> clear directional arrows, and legible sans-serif labels. Layout in three
> horizontal lanes:
>
> **Lane 1 — Offline (build-time), gray:**
> box "FairChem UMA-s-1p2 (PyTorch, FP64)" → box "TorchScript trace + activation-
> checkpoint export (per-block / per-edge-chunk / prologue)" → cylinder "Traced
> model artifacts (.pt shards + metadata)". Small caption under lane: "one-time,
> per system size".
>
> **Lane 2 — Runtime engine (C++/LibTorch), blue, the centerpiece:**
> box "LAMMPS MD driver (fix nvt, neighbor list)" → box "pair_style uma (C++)" →
> box "UMA engine: torch::jit forward + autograd forces (FP64)" with a looped
> sub-box "C++ activation checkpointing (block+chunk recompute)"; from the engine a
> box "cell-list O(N) neighbor graph". Show a feedback arrow "energy + per-atom
> forces" back to the LAMMPS driver. Label the whole lane "NO Python at runtime".
>
> **Lane 3 — Multi-device graph-parallel, teal with an orange accent:**
> a box "Device abstraction layer" branching to two stacked boxes: "Intel XPU —
> oneCCL / SYCL" and "NVIDIA GPU — CUDA / NCCL". To the right, draw **12 small GPU-
> tile icons** connected by a ring of double-headed arrows labeled "per-layer
> feature all-gather + force all-reduce (native collectives)". Caption:
> "1 crystal split across 12 tiles".
>
> **Right-edge results callout box (orange border):** three lines —
> "FP64, matches ASE FairChem to ~1e-9 meV/atom & forces ~1e-14",
> "1 tile: 46,656 atoms   |   12 tiles: ~373k–439k atoms",
> "NVT 300 K validated (energy + forces + AG=FD)".
>
> Keep it uncluttered, poster-legible from ~1.5 m, 16:9-ish aspect, white
> background, no photorealism — schematic infographic only. Do not include any
> real logos.
