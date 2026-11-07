#pragma once

// In-process UMA knobs. pair_style uma keywords write these; the engine
// reads them. Process environment (getenv) is not a user API.

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
