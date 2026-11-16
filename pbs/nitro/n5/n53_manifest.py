#!/usr/bin/env python3
"""Compare an empty-flag export to the stock S3 artifact.

metadata.json carries exporter_git_sha, which changes with the commit. Every
other field must match. That is the manifest-hash check for step 22 (N5.3).
"""
import hashlib
import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    new = Path(sys.argv[1])
    ref = Path(sys.argv[2])
    ok = True
    pairs = [("metadata.json", "metadata.json")]
    for rank in range(12):
        rel = f"w12/r{rank}/metadata.json"
        pairs.append((rel, rel))
    for rel_new, rel_ref in pairs:
        a = load(new / rel_new)
        b = load(ref / rel_ref)
        keys = sorted(set(a) | set(b))
        differ = [k for k in keys if a.get(k) != b.get(k)]
        if differ != ["exporter_git_sha"]:
            ok = False
            print(f"N53_MANIFEST_DIFF {rel_new} keys={differ}", flush=True)
        else:
            print(
                f"N53_MANIFEST_OK {rel_new} git_new={a.get('exporter_git_sha')} "
                f"git_ref={b.get('exporter_git_sha')}",
                flush=True,
            )
    for rank in range(12):
        rel = f"w12/r{rank}/block_export_report.json"
        ga = load(new / rel).get("graph_check")
        gb = load(ref / rel).get("graph_check")
        if ga != gb:
            ok = False
            print(f"N53_GRAPH_CHECK_DIFF {rel} new={ga} ref={gb}", flush=True)
        else:
            print(f"N53_GRAPH_CHECK_OK {rel} {ga}", flush=True)
    for name in ("model_traced.pt", "model_chunk_0.pt", "model_block_0.pt"):
        pn = new / "w12" / "r0" / name
        pr = ref / "w12" / "r0" / name
        if not pn.is_file() or not pr.is_file():
            print(f"N53_GRAPH_MISSING {name}", flush=True)
            ok = False
            continue
        hn, hr = sha_file(pn), sha_file(pr)
        same = int(hn == hr)
        print(f"N53_GRAPH {name} equal={same} new={hn} ref={hr}", flush=True)
        if not same:
            ok = False
    if ok:
        print("N53_MANIFEST_EQUAL", flush=True)
        return 0
    print("N53_MANIFEST_FAIL", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
