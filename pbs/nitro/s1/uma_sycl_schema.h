#pragma once

// One schema string for the step 12 op. The Python stand-in in step 13 reads
// this line. A mismatch makes the traced graph fail to load.

#define UMA_SYCL_SILU_AFFINE_SCHEMA \
  "silu_affine(Tensor x, Tensor w, Tensor b) -> Tensor"
