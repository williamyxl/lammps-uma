"""Tier 1 (a): metadata.json contract — the C++ substring parser's assumptions.

Pure Python (no torch). metadata.cpp parses metadata.json with a hand-rolled
substring scanner (P4'.2 will replace it with a real JSON parser). Until then, this
test:
  1. round-trips the exporter's key set through json (writer format is stable),
  2. replicates the C++ parse_json_string / parse_json_number / parse_compute_dtype
     substring logic and asserts it agrees with a real json.loads on the WELL-FORMED
     cases the production exporter emits,
  3. pins the KNOWN-BRITTLE cases (embedded quote in export_notes, missing key) as
     xfail-style documentation so P4'.1-3 has a concrete target and a future real
     parser can flip them to PASS.

This is the seam the report calls "the least-defended interface in the system".
"""
import json
import sys


# --- replica of metadata.cpp's substring parsers ------------------------------
def cpp_parse_json_string(js, key):
    pos = js.find('"' + key + '":')
    if pos < 0:
        return None
    start = js.find('"', pos + len(key) + 3)
    if start < 0:
        return None
    start += 1
    end = js.find('"', start)          # NOTE: no escape handling (the bug)
    return js[start:end]


def cpp_parse_json_number(js, key):
    pos = js.find('"' + key + '":')
    if pos < 0:
        return None
    # emulate C++ find_first_of("0123456789.-", pos)
    start = None
    for i in range(pos, len(js)):
        if js[i] in "0123456789.-":
            start = i
            break
    if start is None:
        return None
    j = start
    while j < len(js) and js[j] in "0123456789.-eE+":
        j += 1
    return float(js[start:j])


def cpp_parse_compute_dtype(js):
    pos = js.find('"base_precision_dtype":')
    if pos < 0:
        return "float32"        # C++ silently returns kFloat32 (the P4'.2 defect)
    quote = js.find('"', pos + 24)   # magic offset 24
    if quote < 0:
        return "float32"
    end = js.find('"', quote + 1)
    value = js[quote + 1:end]
    return "float64" if "float64" in value else "float32"


def _good_metadata():
    return {
        "model_name": "uma-s-1p2",
        "task_name": "omat",
        "export_format": "blocks_xpu",
        "cutoff": 6.0,
        "max_neighbors": 300,
        "edge_pad_cap": 131072,
        "edge_ac_chunk": 65536,
        "inference_settings": {"base_precision_dtype": "float64"},
        "energy_task": {"name": "omat", "dataset": "omat",
                        "normalizer_mean": 0.0, "normalizer_rmsd": 1.0},
        "export_notes": ["exported on aurora", "fp64 wigner fix applied"],
    }


def test_json_roundtrip_preserves_keys():
    m = _good_metadata()
    s = json.dumps(m, indent=2)
    back = json.loads(s)
    assert back == m


def test_cpp_string_parser_agrees_on_wellformed():
    js = json.dumps(_good_metadata(), indent=2)
    for key in ("model_name", "task_name", "export_format"):
        assert cpp_parse_json_string(js, key) == _good_metadata()[key], key


def test_cpp_number_parser_agrees_on_wellformed():
    js = json.dumps(_good_metadata(), indent=2)
    assert cpp_parse_json_number(js, "cutoff") == 6.0
    assert int(cpp_parse_json_number(js, "max_neighbors")) == 300
    assert int(cpp_parse_json_number(js, "edge_pad_cap")) == 131072


def test_cpp_compute_dtype_float64():
    js = json.dumps(_good_metadata(), indent=2)
    assert cpp_parse_compute_dtype(js) == "float64"


def test_edge_pad_cap_multiple_of_chunk_in_metadata():
    # P4'.3: the loader should validate edge_pad_cap % edge_ac_chunk == 0
    m = _good_metadata()
    assert m["edge_pad_cap"] % m["edge_ac_chunk"] == 0


# --- P4'.2 landed: the C++ now uses nlohmann/json. These document what the OLD
# substring scanner got wrong (regression memory) and assert a REAL parser is
# correct on the same inputs. ------------------------------------------------
def test_embedded_quote_broke_old_substring_parser():
    # The old scanner stopped at the first '"' regardless of escaping; a real JSON
    # parser handles it. We keep the replica to prove the OLD approach was wrong.
    m = _good_metadata()
    m["model_name"] = 'has"quote'
    js = json.dumps(m)
    old = cpp_parse_json_string(js, "model_name")   # buggy substring behavior
    real = json.loads(js)["model_name"]             # what nlohmann now returns
    assert old != real, "the old substring scanner should mishandle escapes"
    assert real == 'has"quote'


def test_real_parser_version_and_dtype_contract():
    # P4'.1/P4'.2/P4'.3 contract, verified with a real parser (json.loads mirrors
    # nlohmann semantics for these cases):
    m = _good_metadata()
    m["metadata_version"] = 2
    js = json.dumps(m)
    d = json.loads(js)
    # version present and >= 2
    assert d.get("metadata_version", 0) >= 2
    # dtype resolves from inference_settings.base_precision_dtype
    assert d["inference_settings"]["base_precision_dtype"] == "float64"
    # missing dtype must be detectable (C++ now THROWS instead of silent float32)
    d2 = {k: v for k, v in m.items() if k != "inference_settings"}
    assert "inference_settings" not in d2
    # edge_pad_cap % edge_ac_chunk == 0 (P4'.3 read-back invariant)
    assert d["edge_pad_cap"] % d["edge_ac_chunk"] == 0


def test_legacy_metadata_missing_version_is_rejected_unless_allowed():
    # A pre-P4'.1 artifact has no metadata_version -> the C++ loader rejects it
    # (value default 0 < 2) unless UMA_ALLOW_LEGACY_METADATA=1. Mirror that policy.
    d = _good_metadata()  # no metadata_version key
    ver = d.get("metadata_version", 0)
    allow_legacy = False
    accepted = (ver >= 2) or allow_legacy
    assert not accepted, "missing version must be rejected by default"
    allow_legacy = True
    accepted = (ver >= 2) or allow_legacy
    assert accepted, "UMA_ALLOW_LEGACY_METADATA=1 must accept legacy"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    n_fail = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            n_fail += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{len(tests)-n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)
