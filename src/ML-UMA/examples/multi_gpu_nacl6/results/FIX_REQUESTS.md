# FIX_REQUESTS — multi_gpu_nacl6 (coordinator)

**Jobs:** g1=`20898588` OK · g2=`20898818` OK except FC · FC-only=`20899729` CANCELLED · ngpu4 deferred

## F3 — FC @ workers=2 process-group (partially fixed)

Original full run: PG double-init after ASE → E=0.  
Fix applied: `dist.destroy_process_group()` in teardown.  
FC-only `20899729` showed correct `PotEng=-5830.9237` then hung/cancelled on `destroy_process_group (FC)`.

**Still needed (WRITE):**

1. Make teardown non-blocking / timeout (NCCL destroy can hang); always write `parity.json` **before** teardown.
2. Resubmit FC-only (`ONLY_PATHS=fc`) until COMPLETED with parity on disk.
3. Prefer subprocess isolation for FC when `workers>1` (NEXT_ROUND item 2).

## F4 — ONLY_PATHS=fc clobbered `results/ngpu2/`

FC-only wiped `ngpu2/parity.json` / forces (ASE+uma rows lost from disk; still in `SUMMARY.json`).

**Minimal WRITE fix:** when `ONLY_PATHS` is a subset, merge into existing `parity.json` or write `parity_fc.json` under a non-destructive path; never unlink sibling path artifacts.
