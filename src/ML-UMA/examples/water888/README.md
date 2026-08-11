# Water NVT 300 K — flexible smoke (LAMMPS data with velocities)

Structure: [`water_nvt_300K_atomic_metal.data`](water_nvt_300K_atomic_metal.data)
(`atom_style atomic`, `units metal`, **Velocities** section included; **648 atoms**,
type 1=O / type 2=H).

- **Ensemble:** NVT 300 K, `run 10`, timestep 1 fs — uses velocities from the data file
- **Molecules:** fully **flexible** — no `fix shake`, rigid, or bond constraints
- **Timing:** SLURM wall around the single `lmp … -in in.h2o888_nvt` line

```bash
sbatch --export=ALL,RECOMPILE=1,NGPUS=1,UMA_DEVICES=1 run_h2o888_nvt.slurm
```

### ASE FairChem FP64 NVT

Same geometry / T / dt / tdamp / 10 steps via `NoseHooverChainNVT` + `FAIRChemCalculator`
(float64). Timed region is `dyn.run(...)` only (`load_predict_unit` untimed).

```bash
sbatch --export=ALL,NGPUS=1,FAIRCHEM_WORKERS=1,NSTEPS=10 run_ase_nvt.slurm
# or on an interactive GPU node:
python run_ase_nvt.py
```

### Three separate SLURM paths (first-frame parity + NVT timing)

| Script | What |
|--------|------|
| `run_path_ase.slurm` | First-frame E+F + `NoseHooverChainNVT` **100** steps (post-warmup) |
| `run_path_fc.slurm` | First-frame `run 0` E+F + NVT **100** steps (single `run`) |
| `run_path_uma.slurm` | First-frame dump E+F + NVT **100** Pair ms/step (cold start excluded) |

Parity uses **only the first frame** (initial geometry). No per-step NVT E/F dumps.

```bash
cd .../examples/water888
sbatch --chdir=$PWD --export=ALL,FAIRCHEM_WORKERS=1,NSTEPS=100 run_path_ase.slurm
sbatch --chdir=$PWD --export=ALL,FAIRCHEM_WORKERS=1,NSTEPS=100 run_path_fc.slurm
sbatch --chdir=$PWD --export=ALL,RECOMPILE=0,NGPUS=1,UMA_DEVICES=1,NSTEPS=100 run_path_uma.slurm
python refresh_compare.py
```

Canonical table: [`results/COMPARE.md`](results/COMPARE.md) / [`results/COMPARE_multigpu.md`](results/COMPARE_multigpu.md).

### Multi-GPU (2 / 4)

1. Export MP shards for n=648: `sbatch export_mp_water888.slurm`
2. Then for each N in 2 4:
   ```bash
   sbatch --chdir=$PWD --gpus-per-node=$N --export=ALL,NGPUS=$N,FAIRCHEM_WORKERS=$N,NSTEPS=100 run_path_ase.slurm
   sbatch --chdir=$PWD --gpus-per-node=$N --export=ALL,NGPUS=$N,FAIRCHEM_WORKERS=$N,NSTEPS=100 run_path_fc.slurm
   sbatch --chdir=$PWD --gpus-per-node=$N --export=ALL,NGPUS=$N,UMA_DEVICES=$N,NSTEPS=100,UMA_PEER_TRANSPORT=nccl run_path_uma.slurm
   ```
3. `python gather_multigpu.py`

### ASE and velocities

Yes — ASE can read this file **with velocities**:

```python
from ase.io import read
atoms = read("water_nvt_300K_atomic_metal.data",
             format="lammps-data", atom_style="atomic", units="metal")
vel = atoms.get_velocities()  # shape (N, 3), Å/ps in metal units
```

(`prep_data.py` from extxyz is optional legacy; the NVT job no longer needs it.)
