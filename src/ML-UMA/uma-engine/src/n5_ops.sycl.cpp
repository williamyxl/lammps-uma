// Step 22 (N5.3). Registers the five fused-op schemas and links them into
// uma_engine. The bodies are the export-time stand-in: they must not run in
// the model. An empty flag set never traces a call, so a call here is a bug.

#include "uma/n5_ops.h"

#include <torch/library.h>

#include <iostream>

namespace {

at::Tensor n5_not_in_model(const at::Tensor& x) {
  TORCH_CHECK(false, "uma_n5 op called; the empty flag set does not use it");
  return x;
}

}  // namespace

extern "C" void uma_n5_ops_linked_anchor() {
  std::cerr << "N53_LINKED " UMA_N5_OP_LIST "\n" << std::flush;
}

TORCH_LIBRARY(uma_n5, m) {
  m.def(UMA_N5_F1_SCHEMA);
  m.def(UMA_N5_F2_SCHEMA);
  m.def(UMA_N5_F3_SCHEMA);
  m.def(UMA_N5_F4_SCHEMA);
  m.def(UMA_N5_F5_SCHEMA);
}

TORCH_LIBRARY_IMPL(uma_n5, CompositeExplicitAutograd, m) {
  m.impl("f1", &n5_not_in_model);
  m.impl("f2", &n5_not_in_model);
  m.impl("f3", &n5_not_in_model);
  m.impl("f4", &n5_not_in_model);
  m.impl("f5", &n5_not_in_model);
}
