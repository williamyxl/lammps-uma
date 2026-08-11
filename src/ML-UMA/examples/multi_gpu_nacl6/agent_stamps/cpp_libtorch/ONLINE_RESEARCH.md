# Online research notes — LibTorch multi-GPU inference (2026-08-07)

## FairChem official
- Multi-GPU = workers=N + Ray. Multi-node needs Ray cluster (fairchem#1949).
- UMA not thread-safe (MoLE); one model instance per GPU/process (#1481).

## Patterns that fit
1. Custom autograd for collectives (pytorch#40690/#40702); TORCH_LIBRARY + register_autograd for export.
2. Export-time FairChem → TorchScript + opaque peer ops; runtime C++ registers ops.
3. Pipeline parallel wrong for atom/edge GP.
4. Same-node peer memcpy / Kokkos fence; single process can own all devices.
5. N× serial TorchScript is an anti-pattern.
