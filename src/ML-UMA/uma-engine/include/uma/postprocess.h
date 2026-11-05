#pragma once

#include <torch/torch.h>

namespace uma {

torch::Tensor denorm_energy(const torch::Tensor& normed,
                            double mean,
                            double rmsd);

torch::Tensor undo_element_references(const torch::Tensor& energy,
                                      const torch::Tensor& atomic_numbers,
                                      const torch::Tensor& batch,
                                      const torch::Tensor& element_references);

}  // namespace uma
