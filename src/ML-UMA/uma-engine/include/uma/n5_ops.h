#pragma once

// Step 22 (N5.3). One schema string per fused op. The Python stand-in reads
// these lines. A later block stage replaces the body; the name stays. An empty
// UMA_NITRO_FUSED does not call any of them, so the traced graph stays the
// stock graph.

#define UMA_N5_F1_SCHEMA "f1(Tensor x) -> Tensor"
#define UMA_N5_F2_SCHEMA "f2(Tensor g, Tensor x, Tensor expand) -> Tensor"
#define UMA_N5_F3_SCHEMA "f3(Tensor x) -> Tensor"
#define UMA_N5_F4_SCHEMA "f4(Tensor x) -> Tensor"
#define UMA_N5_F5_SCHEMA "f5(Tensor x) -> Tensor"
#define UMA_N5_OP_LIST "f1,f2,f3,f4,f5"
