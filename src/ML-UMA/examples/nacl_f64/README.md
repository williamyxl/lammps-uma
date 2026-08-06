# NaCl FP64 parity (positions + energy float64)

ASE ground truth uses FairChem with `base_precision_dtype=float64`.
Artifact: `uma-engine/artifacts/uma-s-1p2-omat-f64` (from uma-lmp root).

```bash
# from uma-lmp root
# Export (once)
PYTHONPATH=uma-engine/python:$PYTHONPATH python uma-engine/python/export_omat.py \
  --dtype float64 --output uma-engine/artifacts/uma-s-1p2-omat-f64

# Python ASE vs export
PYTHONPATH=uma-engine/python:$PYTHONPATH python uma-engine/python/parity_nacl.py --dtype float64

# Full report (ASE + Python + C++ + LAMMPS run 0)
python lammps/src/ML-UMA/examples/nacl_f64/run_parity_f64.py
```
