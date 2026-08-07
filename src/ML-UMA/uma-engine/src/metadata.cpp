#include "uma/metadata.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace uma {
namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

std::string parse_json_string(const std::string& json, const std::string& key) {
  const auto pos = json.find("\"" + key + "\":");
  if (pos == std::string::npos) {
    throw std::runtime_error("Missing JSON key: " + key);
  }
  auto start = json.find('"', pos + key.size() + 3);
  if (start == std::string::npos) {
    throw std::runtime_error("Malformed JSON string for key: " + key);
  }
  ++start;
  const auto end = json.find('"', start);
  return json.substr(start, end - start);
}

double parse_json_number(const std::string& json, const std::string& key) {
  const auto pos = json.find("\"" + key + "\":");
  if (pos == std::string::npos) {
    throw std::runtime_error("Missing JSON key: " + key);
  }
  auto start = json.find_first_of("0123456789.-", pos);
  if (start == std::string::npos) {
    throw std::runtime_error("Malformed JSON number for key: " + key);
  }
  return std::stod(json.substr(start));
}

int parse_json_int(const std::string& json, const std::string& key) {
  return static_cast<int>(parse_json_number(json, key));
}

std::string extract_object(const std::string& json, const std::string& key) {
  const auto key_pos = json.find("\"" + key + "\":");
  if (key_pos == std::string::npos) {
    throw std::runtime_error("Missing JSON object: " + key);
  }
  auto brace = json.find('{', key_pos);
  if (brace == std::string::npos) {
    throw std::runtime_error("Malformed JSON object: " + key);
  }
  int depth = 0;
  for (size_t i = brace; i < json.size(); ++i) {
    if (json[i] == '{') {
      ++depth;
    } else if (json[i] == '}') {
      --depth;
      if (depth == 0) {
        return json.substr(brace, i - brace + 1);
      }
    }
  }
  throw std::runtime_error("Unterminated JSON object: " + key);
}

torch::Tensor parse_json_double_array(const std::string& json,
                                      const std::string& key) {
  const auto key_pos = json.find("\"" + key + "\":");
  if (key_pos == std::string::npos) {
    return {};
  }
  auto start = json.find('[', key_pos);
  auto end = json.find(']', start);
  if (start == std::string::npos || end == std::string::npos) {
    throw std::runtime_error("Malformed JSON array: " + key);
  }
  std::vector<double> values;
  std::istringstream stream(json.substr(start + 1, end - start - 1));
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.find_first_of("0123456789.-") == std::string::npos) {
      continue;
    }
    values.push_back(std::stod(token));
  }
  if (values.empty()) {
    return {};
  }
  return torch::tensor(values, torch::kFloat64);
}

torch::ScalarType parse_compute_dtype(const std::string& json) {
  const auto pos = json.find("\"base_precision_dtype\":");
  if (pos == std::string::npos) {
    return torch::kFloat32;
  }
  auto quote = json.find('"', pos + 24);
  if (quote == std::string::npos) {
    return torch::kFloat32;
  }
  ++quote;
  const auto end = json.find('"', quote);
  const std::string value = json.substr(quote, end - quote);
  if (value.find("float64") != std::string::npos) {
    return torch::kFloat64;
  }
  return torch::kFloat32;
}

}  // namespace

ArtifactMetadata load_artifact_metadata(const std::string& metadata_path) {
  const auto json = read_file(metadata_path);
  const auto energy_task = extract_object(json, "energy_task");

  ArtifactMetadata meta;
  meta.model_name = parse_json_string(json, "model_name");
  meta.task_name = parse_json_string(json, "task_name");
  meta.export_format = parse_json_string(json, "export_format");
  meta.cutoff = parse_json_number(json, "cutoff");
  meta.max_neighbors = parse_json_int(json, "max_neighbors");
  meta.normalizer_mean = parse_json_number(energy_task, "normalizer_mean");
  meta.normalizer_rmsd = parse_json_number(energy_task, "normalizer_rmsd");
  meta.compute_dtype = parse_compute_dtype(json);
  meta.element_references =
      parse_json_double_array(energy_task, "element_references");
  // Optional: used by GraphParallelRuntime (devices>1) eager FairChem path.
  try {
    meta.checkpoint_path = parse_json_string(json, "checkpoint_path");
  } catch (const std::exception&) {
    meta.checkpoint_path.clear();
  }
  return meta;
}

}  // namespace uma
