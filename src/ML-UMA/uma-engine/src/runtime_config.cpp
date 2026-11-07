#include "uma/runtime_config.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace uma {
namespace {

std::mutex g_mu;
std::unordered_map<std::string, std::unique_ptr<std::string>> g_vars;

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
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_vars.find(name);
  if (it == g_vars.end()) return nullptr;
  return it->second->c_str();
}

}  // namespace uma
