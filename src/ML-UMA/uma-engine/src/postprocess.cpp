#include "uma/postprocess.h"

namespace uma {

torch::Tensor denorm_energy(const torch::Tensor& normed,
                            double mean,
                            double rmsd) {
  // Preserve compute dtype: FP32 for mixed, FP64 for double.
  // Previously always cast to FP32, which ~1 meV vs ASE after float32
  // element-reference accumulation on NaCl 3x3x3.
  const auto dtype = normed.scalar_type();
  auto rmsd_t = torch::tensor(rmsd, normed.options().dtype(dtype));
  auto mean_t = torch::tensor(mean, normed.options().dtype(dtype));
  return normed.to(dtype) * rmsd_t + mean_t;
}

torch::Tensor undo_element_references(
    const torch::Tensor& energy,
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& batch,
    const torch::Tensor& element_references) {
  if (!element_references.defined() || element_references.numel() == 0) {
    return energy;
  }
  const auto z = atomic_numbers.to(torch::kLong);
  const auto b = batch.to(torch::kLong);
  const auto refs = element_references.to(energy.device(), energy.scalar_type());
  const auto per_atom = refs.index({z});
  auto out = energy.clone();
  return out.scatter_add_(0, b, per_atom);
}

}  // namespace uma
