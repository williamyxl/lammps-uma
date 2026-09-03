#include "uma/metadata.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace uma {
namespace {

using json = nlohmann::json;

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

bool env_flag(const char* name) {
  const char* e = std::getenv(name);
  return e && e[0] == '1' && e[1] == '\0';
}

// P4'.2: real JSON accessors (replace the hand-rolled substring scanners which
// had no escape handling, no nesting, and a magic pos+24 offset). Missing
// required keys throw with the artifact path + key named.
const json& require(const json& j, const char* key, const std::string& path) {
  auto it = j.find(key);
  if (it == j.end()) {
    throw std::runtime_error("metadata.json (" + path + "): missing required key '" +
                             std::string(key) + "'");
  }
  return *it;
}

// A2/S2 fix (audit rev 26 §G.21.5): null-tolerant optional-string read. nlohmann's
// j.value("k", default) substitutes the default ONLY when the key is ABSENT; if
// the key is present but JSON null it throws type_error.302. Exporters legitimately
// emit null for un-set provenance (checkpoint_sha256, atom_refs, ...), so a null
// optional string must read as empty, not abort the whole artifact load. This bug
// made the fresh v2 DD artifact unloadable ("type must be string, but is null").
std::string opt_string(const json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null() || !it->is_string()) return std::string();
  return it->get<std::string>();
}

torch::ScalarType parse_compute_dtype(const json& j, const std::string& path) {
  // base_precision_dtype lives under inference_settings. P4'.2: on any parse
  // difficulty THROW (the old code silently returned kFloat32, which would run a
  // float64 artifact in float32 -> wrong physics with no diagnostic).
  auto is_it = j.find("inference_settings");
  if (is_it == j.end() || !is_it->is_object()) {
    throw std::runtime_error("metadata.json (" + path +
                             "): missing inference_settings (cannot determine "
                             "base_precision_dtype)");
  }
  auto dt_it = is_it->find("base_precision_dtype");
  if (dt_it == is_it->end() || !dt_it->is_string()) {
    throw std::runtime_error("metadata.json (" + path +
                             "): missing/!string inference_settings."
                             "base_precision_dtype");
  }
  const std::string value = dt_it->get<std::string>();
  if (value.find("float64") != std::string::npos) return torch::kFloat64;
  if (value.find("float32") != std::string::npos) return torch::kFloat32;
  throw std::runtime_error("metadata.json (" + path +
                           "): unrecognized base_precision_dtype '" + value + "'");
}

torch::Tensor parse_double_array(const json& obj, const char* key) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return {};
  if (!it->is_array()) {
    throw std::runtime_error(std::string("metadata.json: key '") + key +
                             "' is not an array");
  }
  std::vector<double> values = it->get<std::vector<double>>();
  if (values.empty()) return {};
  return torch::tensor(values, torch::kFloat64);
}

}  // namespace

ArtifactMetadata load_artifact_metadata(const std::string& metadata_path) {
  const std::string text = read_file(metadata_path);
  json j;
  try {
    j = json::parse(text);
  } catch (const json::parse_error& e) {
    throw std::runtime_error("metadata.json (" + metadata_path +
                             "): JSON parse error: " + e.what());
  }

  ArtifactMetadata meta;

  // ---- P4'.1: schema version + provenance --------------------------------
  const bool allow_legacy = env_flag("UMA_ALLOW_LEGACY_METADATA");
  meta.metadata_version = j.value("metadata_version", 0);
  if (meta.metadata_version < 2 && !allow_legacy) {
    throw std::runtime_error(
        "metadata.json (" + metadata_path + "): metadata_version=" +
        std::to_string(meta.metadata_version) +
        " is pre-P4'.1 (expected >= 2). Re-export with the current exporter, or "
        "set UMA_ALLOW_LEGACY_METADATA=1 to load an old artifact at your own risk.");
  }
  if (meta.metadata_version >= 2 && j.contains("metadata_version") &&
      !j["metadata_version"].is_number_integer()) {
    throw std::runtime_error("metadata.json (" + metadata_path +
                             "): metadata_version must be an integer");
  }
  // opt_string (not j.value): null-tolerant, so a provenance field written as
  // JSON null reads empty instead of throwing (see opt_string above).
  meta.fairchem_version = opt_string(j, "fairchem_version");
  meta.torch_version = opt_string(j, "torch_version");
  meta.exporter_git_sha = opt_string(j, "exporter_git_sha");
  meta.checkpoint_sha256 = opt_string(j, "checkpoint_sha256");

  // ---- required core fields ----------------------------------------------
  meta.model_name = require(j, "model_name", metadata_path).get<std::string>();
  meta.task_name = require(j, "task_name", metadata_path).get<std::string>();
  meta.export_format = require(j, "export_format", metadata_path).get<std::string>();
  meta.cutoff = require(j, "cutoff", metadata_path).get<double>();
  meta.max_neighbors = require(j, "max_neighbors", metadata_path).get<int>();

  const json& et = require(j, "energy_task", metadata_path);
  meta.normalizer_mean = require(et, "normalizer_mean", metadata_path).get<double>();
  meta.normalizer_rmsd = require(et, "normalizer_rmsd", metadata_path).get<double>();
  meta.compute_dtype = parse_compute_dtype(j, metadata_path);
  meta.element_references = parse_double_array(et, "element_references");

  // ---- optional fields ----------------------------------------------------
  meta.checkpoint_path = opt_string(j, "checkpoint_path");
  meta.edge_ac_chunk = j.value("edge_ac_chunk", 0);
  meta.edge_pad_cap = j.value("edge_pad_cap", 0);
  meta.edge_pad_atom = j.value("edge_pad_atom", 0);
  meta.dd_halo_width = j.value("dd_halo_width", 0);

  // ---- P4'.3: read back what the exporter wrote and validate it ----------
  // edge_pad_cap must be a positive multiple of edge_ac_chunk when both are set
  // (the padding invariant the traced chunk loop relies on). A mismatch here is a
  // corrupt/hand-edited artifact -> hard error instead of a step-1 crash.
  if (meta.edge_pad_cap > 0 && meta.edge_ac_chunk > 0 &&
      (meta.edge_pad_cap % meta.edge_ac_chunk) != 0) {
    throw std::runtime_error(
        "metadata.json (" + metadata_path + "): edge_pad_cap (" +
        std::to_string(meta.edge_pad_cap) + ") is not a multiple of edge_ac_chunk (" +
        std::to_string(meta.edge_ac_chunk) + ")");
  }
  // world/rank sanity (present on GP artifacts). If present they must be coherent.
  if (j.contains("world") && j.contains("rank")) {
    const int world = j.value("world", 1);
    const int rank = j.value("rank", 0);
    if (world < 1 || rank < 0 || rank >= world) {
      throw std::runtime_error(
          "metadata.json (" + metadata_path + "): incoherent world/rank (world=" +
          std::to_string(world) + ", rank=" + std::to_string(rank) + ")");
    }
  }

  return meta;
}

}  // namespace uma
