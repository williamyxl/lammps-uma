#!/usr/bin/env python3
"""Write meta.json for one harness run.

Usage: meta.py OUTDIR REPO [--lmp PATH] [--artifact DIR] [--tag TAG] [--extra K=V ...]
Records git state, binary sha256, artifact id, toolchain, hosts and PBS info.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    try:
        return subprocess.run(
            cmd, cwd=cwd, check=False, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_id(art):
    meta = Path(art) / "metadata.json"
    if not meta.is_file():
        return None
    files = sorted(
        str(p.relative_to(art)) for p in Path(art).rglob("*") if p.is_file()
    )
    h = hashlib.sha256()
    h.update(meta.read_bytes())
    h.update("\n".join(files).encode())
    return h.hexdigest()


def artifact_bytes(art):
    total = 0
    root = Path(art)
    if not root.is_dir():
        return None
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def manifest_sha(art):
    meta = Path(art) / "metadata.json"
    if not meta.is_file():
        return None
    return hashlib.sha256(meta.read_bytes()).hexdigest()


def fairchem_version():
    import importlib.metadata as md
    for dist in ("fairchem-core", "fairchem"):
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
    return None


def nitro_flags(log_path):
    if not log_path:
        return None
    path = Path(log_path)
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if "NITRO_FLAGS:" in line:
            return line.split("NITRO_FLAGS:", 1)[1].strip()
    return None


def resolved_libsycl():
    explicit = os.environ.get("NITRO_LIBSYCL")
    if explicit:
        return str(Path(explicit).resolve()) if Path(explicit).exists() else explicit
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not entry:
            continue
        for name in ("libsycl.so.8", "libsycl.so.9", "libsycl.so"):
            cand = Path(entry) / name
            if cand.exists():
                return str(cand.resolve())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("repo")
    ap.add_argument("--lmp")
    ap.add_argument("--artifact")
    ap.add_argument("--tag")
    ap.add_argument("--require-base", action="store_true",
                    help="exit 1 when the nitro-base tag or --tag is missing (timing jobs)")
    ap.add_argument("--log", help="LAMMPS log to read the NITRO_FLAGS line from")
    ap.add_argument("--artifact-kept", choices=("yes", "no"), default="yes")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()

    repo = a.repo
    base_sha = run(["git", "rev-parse", "-q", "--verify", "nitro-base^{commit}"], repo)
    if a.require_base and not base_sha:
        raise SystemExit("FAIL: nitro-base tag is not set (required for timing jobs)")
    if a.require_base and not a.tag:
        raise SystemExit("FAIL: --tag is required for timing jobs")
    nodefile = os.environ.get("PBS_NODEFILE")
    hosts = []
    if nodefile and Path(nodefile).is_file():
        hosts = sorted(set(Path(nodefile).read_text().split()))
    meta = {
        "git_describe": run(["git", "describe", "--tags", "--always", "--dirty"], repo),
        "git_sha": run(["git", "rev-parse", "HEAD"], repo),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo),
        "tag": a.tag,
        "nitro_tag": a.tag,
        "nitro_base_sha": base_sha or None,
        "lmp": a.lmp,
        "lmp_sha256": sha256_file(a.lmp) if a.lmp and Path(a.lmp).is_file() else None,
        "artifact": a.artifact,
        "artifact_id": artifact_id(a.artifact) if a.artifact else None,
        "artifact_bytes": artifact_bytes(a.artifact) if a.artifact else None,
        "artifact_manifest_sha": manifest_sha(a.artifact) if a.artifact else None,
        "artifact_kept": a.artifact_kept == "yes",
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "pbs_queue": os.environ.get("PBS_QUEUE"),
        "hosts": hosts,
        "icpx": run(["icpx", "--version"]).splitlines()[:1],
        "icpx_version": (run(["icpx", "--version"]).splitlines() or [None])[0],
        "libsycl": resolved_libsycl(),
        "mpich": os.environ.get("MPICH") or os.environ.get("NITRO_MPICH"),
        "oneccl_root": os.environ.get("ONECCL_ROOT") or os.environ.get("CONDA_PREFIX"),
        "fairchem": fairchem_version(),
        "fairchem_expected": "2.21.0",
        "nitro_flags": nitro_flags(a.log),
        "python": sys.version.split()[0],
        "env": {
            k: os.environ.get(k)
            for k in (
                "UMA_AC", "UMA_MN_CKPT", "UMA_MP_PERF", "UMA_CHECKPOINT",
                "OMP_NUM_THREADS", "CCL_WORKER_COUNT", "FI_PROVIDER",
                "LD_LIBRARY_PATH",
            )
        },
        "extra": dict(kv.split("=", 1) for kv in a.extra if "=" in kv),
    }
    try:
        import torch  # noqa: WPS433

        meta["torch"] = torch.__version__
    except ImportError:
        meta["torch"] = None
    meta["fairchem_match"] = meta["fairchem"] == meta["fairchem_expected"]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"META_OK {out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
