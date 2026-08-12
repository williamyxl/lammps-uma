# Phase P0 - Polaris single-node validation (1/2/4 GPUs)

LAMMPS + LibTorch-UMA (precision double, devices 1/2/4) vs ASE-FairChem FP64 (task=omat).

Gates: |dE| <= 1e-06 eV (or rel <= 1e-09); per-atom max|dF| <= 1e-05 eV/A.

Each LAMMPS path is compared to the ASE-FC FP64 oracle in the MATCHING recipe: devices 1 = general (traced model); devices 2/4 = umas_fast_pytorch + merge_mole (MP shards).

| system | N | GPUs | recipe | E_lammps (eV) | |dE| | max|dF| | mean|dF| | NVT ms/step | speedup | verdict |
|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|:--:|
| nacl666 | 1728 | 1 | general | -5830.923720167 | 1.282e-10 | 3.911e-14 | 7.246e-15 | 314.47 | 1.00x | PASS |
| nacl666 | 1728 | 2 | fastmerge | -5830.923741338 | 1.210e-10 | 1.377e-14 | 2.562e-15 | 161.88 | 1.94x | PASS |
| nacl666 | 1728 | 4 | fastmerge | -5830.923741338 | 1.201e-10 | 1.391e-14 | 2.579e-15 | 103.34 | 3.04x | PASS |
| water888 | 648 | 1 | general | -3143.389356118 | 8.640e-12 | 3.021e-13 | 7.630e-14 | 324.43 | 1.00x | PASS |
| water888 | 648 | 2 | fastmerge | -3143.389377472 | 8.640e-12 | 2.571e-13 | 1.886e-14 | 168.74 | 1.92x | PASS |
| water888 | 648 | 4 | fastmerge | -3143.389377472 | 8.640e-12 | 2.594e-13 | 1.893e-14 | 106.45 | 3.05x | PASS |

**P0 PASS**

