# Settings documents (E / force / timing)

**Stamp:** 2026-08-09T20:41

| Document | `execution_mode` | `merge_mole` |
|----------|------------------|--------------|
| [SETTINGS_no_merge_no_fast.md](SETTINGS_no_merge_no_fast.md) | `general` | False |
| [SETTINGS_merge_no_fast.md](SETTINGS_merge_no_fast.md) | `general` | True |
| [SETTINGS_fast_no_merge.md](SETTINGS_fast_no_merge.md) | `umas_fast_pytorch` | False (**illegal**) |
| [SETTINGS_fast_and_merge.md](SETTINGS_fast_and_merge.md) | `umas_fast_pytorch` | True |

Each doc: ASE · FC · uma/kk × NaCl6 · water888 × @1/@2/@4 — energy, per-atom force parity, timing.

Regenerate: `python ../regenerate_settings_docs.py` (optionally `--ingest-matrix`).
