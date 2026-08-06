#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

namespace uma {

struct ArtifactMetadata {
  std::string model_name;
  std::string task_name;
  std::string export_format;
  double cutoff = 6.0;
  int max_neighbors = 300;
  double normalizer_mean = 0.0;
  double normalizer_rmsd = 1.0;
  torch::ScalarType compute_dtype = torch::kFloat32;
  torch::Tensor element_references;  // [num_elements] or empty
};

ArtifactMetadata load_artifact_metadata(const std::string& metadata_path);

}  // namespace uma
