# NaCl FP64 parity (positions + energy float64)

ASE ground truth uses FairChem with `base_precision_dtype=float64`.
Artifact: `lammps/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64`.

```bash
# from uma-lmp root
ENG=lammps/src/ML-UMA/uma-engine

# Export (once)
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_omat.py \
  --dtype float64 --output $ENG/artifacts/uma-s-1p2-omat-f64

# Python ASE vs export
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/parity_nacl.py --dtype float64

# Full report (ASE + Python + C++ + LAMMPS run 0)
python lammps/src/ML-UMA/examples/nacl_f64/run_parity_f64.py
```
