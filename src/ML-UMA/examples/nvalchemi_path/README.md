# ALCHEMI Toolkit path — 4th comparison path

NVIDIA [ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) as both
inference and MD engine, alongside the three existing paths:

| Path | Inference | MD engine |
|------|-----------|-----------|
| ASE FC FP64 | fairchem `MLIPPredictUnit` | ASE |
| FC LAMMPS | fairchem | LAMMPS |
| LibTorch UMA LAMMPS (**product, W8nk**) | TorchScript + MP workers | LAMMPS |
| **nvalchemi** (this) | `UMAWrapper` (fairchem under the hood) | `NVTNoseHoover` |

Same frozen boxes (NaCl 6×6×6 = 1728 atoms, water888 = 2592), same FP64 +
`merge_mole`, same merge oracle → numbers are directly comparable.

## Environment (separate on purpose)

Installed in its own conda env **`nvalchemi312`**, not `uma312`.

The toolkit's `uma` extra is declared **mutually exclusive** with `cu12`,
`cu13`, and `mace`: `fairchem-core` caps `torch<2.9`, while the CUDA extras
floor it at `torch>=2.11` via `nvalchemi-toolkit-ops`. Installing into
`uma312` would have re-resolved torch and broken the entire running campaign.

```bash
conda create -n nvalchemi312 python=3.12 pip
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
            --extra-index-url https://pypi.nvidia.com \
            'nvalchemi-toolkit[uma,ase]' 'setuptools<81'
```

Resolved: `nvalchemi-toolkit 0.2.0`, `torch 2.8.0+cu128`, `fairchem-core 2.21.0`,
`warp-lang 1.16.0`. Uses the cached local checkpoint, so no HuggingFace
gating is required.

## Protocol match

- **NVT**: `NVTNoseHoover` — matches LAMMPS `fix nvt`. (`NVTLangevin` is a
  *different ensemble* and would not be a like-for-like comparison.)
- **FP64**: set on both `InferenceSettings.base_precision_dtype` **and**
  `AtomicData.from_atoms(dtype=...)`, which defaults to float32 and would
  otherwise silently break parity before the model is called.
- **Steps**: NaCl 10, water 100 (campaign convention).
- **Parity**: `|ΔE| ≤ 1e-6 eV`, `max|ΔF| ≤ 1e-5 eV/Å` vs the merge oracle,
  plus `|Σ F| < 1e-6` net-force sanity.

## Multi-GPU

1 GPU runs the integrator directly. 2/4 GPUs use `DomainParallel` spatial
domain decomposition via `torchrun --nproc_per_node=N`. Note this is a
**different parallel strategy** from the product path, which uses
model-parallel process-per-rank workers with NCCL — so a speed difference
does not by itself indicate a better or worse implementation.

## Usage

```bash
sbatch smoke.slurm                    # 1-GPU validation first
python run_nvalchemi.py --sys nacl6 --ngpu 1
torchrun --nproc_per_node=4 run_nvalchemi.py --sys nacl6 --ngpu 4
```

## Status

Public beta, API subject to change. The 1-GPU smoke test gates whether the
2/4-GPU matrix is worth submitting.
