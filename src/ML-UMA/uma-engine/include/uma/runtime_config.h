#pragma once

// In-process UMA knobs. pair_style uma keywords write these; the engine
// reads them.
//
// Keyword names (UMA_AC, UMA_DD, UMA_ENGINE_BUILD_GRAPH, UMA_COMPUTE_VIRIAL)
// come only from runtime_set. They never fall back to the process environment.
// An allowlist of harness perf/debug and functional names, plus any
// UMA_NITRO_* arm toggle, falls back to getenv when the map has no value.
// The list is in runtime_config.cpp and pbs/nitro/ENV_AUDIT.md.

namespace uma {

// Empty or null value unsets the name.
void runtime_set(const char *name, const char *value);
const char *runtime_get(const char *name);

inline bool runtime_flag_1(const char *name)
{
  const char *e = runtime_get(name);
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}

}  // namespace uma
