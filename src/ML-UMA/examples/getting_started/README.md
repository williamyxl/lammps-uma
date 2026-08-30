# Getting started: LAMMPS + UMA (`pair_style uma/kk`)

This folder is a short path from “I have the `lammps-uma` tree” to
“I ran 10 NVT steps of NaCl with UMA on one GPU.”

**Precision:** FP64 only (`pair_style uma/kk precision double`). Mixed
precision is disabled for now (FP32 energy graph / upcast forces).

| File | What it does |
|------|----------------|
| [`build_uma.slurm`](build_uma.slurm) | Builds LAMMPS with Kokkos CUDA + UMA (exports **omat** FP64 if needed) |
| [`export_all_tasks_fp64.slurm`](export_all_tasks_fp64.slurm) | Exports **all** UMA energy-task artifacts in FP64 |
| [`in.nacl5_nvt`](in.nacl5_nvt) | NaCl 5×5×5 rocksalt, NVT @ 300 K, 10 steps |
| this README | How to build and run |

Export tool: `../../uma-engine/python/export_artifact.py`

---

## Before you start

1. You are on a **GPU** node (or about to submit a GPU job).
2. Conda env **`uma312`** is available (PyTorch + CUDA).
3. The UMA checkpoint exists (default on this machine):
   `$UMA_CHECKPOINT`

No extra data file is required: the input builds the crystal with
LAMMPS `lattice` / `create_atoms`.

---

## Step 1 — Build LAMMPS with UMA

From anywhere:

```bash
sbatch $ROOT/src/ML-UMA/examples/getting_started/build_uma.slurm
```

The job will:

1. Load CUDA + CMake
2. Activate `uma312`
3. Export the FP64 **omat** artifact via `export_artifact.py` (skipped if present)
4. Compile `build-uma/lmp` with `pair_style uma/kk`

When it finishes, you should have:

```text
lammps-uma/build-uma/lmp
lammps-uma/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64/
```

### Optional — export all UMA task heads (FP64)

The multihead checkpoint has several energy tasks (`omat`, `odac`, `omol`,
`oc20`, `oc22`, `oc25`, `omc`). Each needs its own TorchScript artifact:

```bash
sbatch $ROOT/src/ML-UMA/examples/getting_started/export_all_tasks_fp64.slurm
```

Equivalent manual command:

```bash
ENG=$ROOT/src/ML-UMA/uma-engine
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_artifact.py \
  --checkpoint $UMA_CHECKPOINT \
  --dtype float64 --all-tasks --skip-existing \
  --artifacts-root $ENG/artifacts
```

Writes `artifacts/uma-s-1p2-<task>-f64/`. LAMMPS `pair_coeff` still points at
**one** directory (getting_started uses **omat**).

Rebuild later the same way if the UMA sources change.

In the LAMMPS input:

```lammps
pair_style uma/kk precision double
pair_coeff * * .../artifacts/uma-s-1p2-omat-f64 Na Cl
```

---

## Step 2 — Run the NaCl NVT example

On a GPU allocation (interactive or batch), with the same modules/env:

```bash
cd $ROOT

source $CONDA_SETUP
conda activate uma312
module unload cudatoolkit 2>/dev/null || true
module load cuda/12.8

# Shared libraries LAMMPS needs at runtime
ENG=$PWD/src/ML-UMA/uma-engine
TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="${ENG}/third_party/vesin/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

cd src/ML-UMA/examples/getting_started

# One GPU, Kokkos package, UMA pair style
../../../../build-uma/lmp -k on g 1 -sf kk -in in.nacl5_nvt
```

You should see thermo output for steps 0–10 and a normal LAMMPS “Total wall time”.

---

## What the input does (LAMMPS view)

`in.nacl5_nvt` is a normal metal-units script:

- Builds **NaCl rocksalt** as two FCC lattices (Na + Cl), **5×5×5** cells → **1000 atoms**
- Uses **`pair_style uma/kk precision double`** (FP64 energies/forces)
- Points `pair_coeff` at the exported artifact directory and element map `Na Cl`
- Turns **`newton off`** (required for this pair style)
- Creates velocities at **300 K**, runs **`fix nvt`** for **10** steps (`timestep 0.001` → 1 fs)

Edit temperature, steps, or box size the same way you would in any other LAMMPS input.

---

## Notes for LAMMPS users

- Launch with **`-k on g 1 -sf kk`**. That turns on Kokkos on one GPU and applies the `/kk` suffixes.
- This build is **single MPI rank** (`BUILD_MPI=OFF`). Do not use `mpirun` for this binary.
- `pair_coeff` takes a **directory** (the artifact), not a single `.pt` file:
  `.../artifacts/uma-s-1p2-omat-f64`
- Element names after the artifact must match atom types in order (`1=Na`, `2=Cl`).
- For longer or multi-GPU campaigns, see `../multi_gpu_nacl6/` after you are comfortable with this smoke run.
