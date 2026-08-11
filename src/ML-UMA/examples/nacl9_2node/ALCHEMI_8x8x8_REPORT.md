# ALCHEMI on NaCl 8×8×8 (4096 atoms) — 4 GPU vs 8 GPU

FP64, `graph_partition`, single point + NVT 300 K. Ground truth is the LibTorch UMA 4-GPU run (job 21026029) on the **same** box (`nacl8_rattle`).

**Ground truth (LibTorch, 4 GPU):** E = `-13821.798173425` eV, NVT `189.7` ms/step, forces (4096,3), |F|max `0.5368`

_No ALCHEMI run on the nacl8 ground-truth box yet (jobs 21038856 / 21038857 pending)._
