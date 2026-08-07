# NaCl FP64 single-point (legacy)

Prefer [`../multi_gpu_nacl6/`](../multi_gpu_nacl6/) for ASE / FC / `uma/kk`
double parity on NaCl6.

Remaining: `in.nacl_sp` / `data.nacl` + `run_sp.sh`. Use artifact
`uma-s-1p2-omat-f64` and `precision double` if you reuse these inputs.

```bash
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=$ROOT/src/ML-UMA/uma-engine
CKPT=/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt

PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_artifact.py \
  --checkpoint $CKPT --dtype float64 --task omat \
  --output $ENG/artifacts/uma-s-1p2-omat-f64
```
