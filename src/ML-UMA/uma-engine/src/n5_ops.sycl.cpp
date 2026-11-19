// Step 22 (N5.3). Registers the five fused-op schemas and links them into
// uma_engine. The bodies are the export-time stand-in: they must not run in
// the model. An empty flag set never traces a call, so a call here is a bug.

#include "uma/n5_ops.h"

#include <torch/library.h>

#include <iostream>


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

// f3 is implemented in libuma_f3.so (step 27). The identity stub lived here
// until that schema gained weights.
