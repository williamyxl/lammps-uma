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
  /// Optional path to scientific UMA checkpoint (eager GP for devices>1).
  std::string checkpoint_path;
  /// P2.1 edge-cap padding: the fixed traced edge capacity (a multiple of
  /// edge_ac_chunk) and the dummy pad atom index. The runtime pads its per-step
  /// edge_index up to edge_pad_cap on atom edge_pad_atom (self-loops beyond
  /// cutoff -> zero contribution) so the traced per-chunk loop count is invariant
  /// to per-step edge drift. 0 => padding disabled (legacy drift-prone path).
  int edge_pad_cap = 0;
  int edge_pad_atom = 0;
  /// DD k=4: per-node feature width (sph_feature_size * sphere_channels). LAMMPS
  /// must size its forward/reverse comm buffer to this many doubles/atom before
  /// the run (via comm_forward/comm_reverse in init_style). 0 => not a DD artifact.
  int dd_halo_width = 0;
};

ArtifactMetadata load_artifact_metadata(const std::string& metadata_path);

}  // namespace uma
