#include "uma/runtime_config.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// libc's environment vector. This must stay at global scope: an extern inside
// namespace uma is a different symbol, and linking lmp then fails.
extern "C" char **environ;

namespace uma {
namespace {

std::mutex g_mu;
std::unordered_map<std::string, std::unique_ptr<std::string>> g_vars;

// Harness perf/debug and functional names. Keyword names are absent on purpose:
// UMA_AC, UMA_DD, UMA_ENGINE_BUILD_GRAPH, UMA_COMPUTE_VIRIAL.
bool allowlisted_exact(const char *name)
{
  static const char *kNames[] = {
      "UMA_ALLOW_LEGACY_METADATA",
      "UMA_MEM_PEAK",
      "UMA_MN_CKPT",
      "UMA_MP_NATOMS",
      "UMA_MP_PERF",
  };
  for (const char *n : kNames) {
    if (std::strcmp(name, n) == 0) return true;
  }
  return false;
}

// UMA_NITRO_<ID> with a non-empty <ID>. Unset means the BASE path.
bool is_nitro_toggle(const char *name)
{
  static constexpr char kPrefix[] = "UMA_NITRO_";
  constexpr std::size_t n = sizeof(kPrefix) - 1;
  return std::strncmp(name, kPrefix, n) == 0 && name[n] != '\0';
}

bool allowlisted(const char *name)
{
  return allowlisted_exact(name) || is_nitro_toggle(name);
}

// Launcher rank, if the process was started under one. No rank variable means
// a single process, which logs.
bool log_from_this_rank()
{
  static const char *kRankKeys[] = {
      "PMI_RANK", "PMIX_RANK", "PALS_RANKID", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK",
  };
  for (const char *k : kRankKeys) {
    const char *v = std::getenv(k);
    if (v != nullptr && v[0] != '\0') return std::strcmp(v, "0") == 0;
  }
  return true;
}

void log_nitro_flags_once()
{
  static std::once_flag once;
  std::call_once(once, [] {
    if (!log_from_this_rank()) return;
    std::vector<std::string> flags;
    for (char **e = ::environ; e != nullptr && *e != nullptr; ++e) {
      const char *s = *e;
      if (std::strncmp(s, "UMA_NITRO_", 10) != 0) continue;
      const char *eq = std::strchr(s, '=');
      if (eq == nullptr || eq == s + 10) continue;
      flags.emplace_back(s);
    }
    std::sort(flags.begin(), flags.end());
    std::cerr << "NITRO_FLAGS:";
    if (flags.empty()) {
      std::cerr << " (none)";
    } else {
      for (const std::string &f : flags) std::cerr << ' ' << f;
    }
    std::cerr << '\n';
  });
}

}  // namespace

void runtime_set(const char *name, const char *value)
{
  if (name == nullptr || name[0] == '\0') return;
  std::lock_guard<std::mutex> lock(g_mu);
  if (value == nullptr || value[0] == '\0') {
    g_vars.erase(name);
    return;
  }
  g_vars[name] = std::make_unique<std::string>(value);
}

const char *runtime_get(const char *name)
{
  if (name == nullptr || name[0] == '\0') return nullptr;
  log_nitro_flags_once();
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_vars.find(name);
  if (it != g_vars.end()) return it->second->c_str();
  if (!allowlisted(name)) return nullptr;
  const char *env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') return nullptr;
  // Copy into the map so the returned pointer stays valid, same as a runtime_set value.
  auto stored = std::make_unique<std::string>(env);
  const char *out = stored->c_str();
  g_vars.emplace(name, std::move(stored));
  return out;
}

}  // namespace uma
