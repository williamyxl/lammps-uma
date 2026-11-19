#pragma once

// Step 22 (N5.3). One schema string per fused op. The Python stand-in reads
// these lines. A later block stage replaces the body; the name stays. An empty
// UMA_NITRO_FUSED does not call any of them, so the traced graph stays the
// stock graph.

#define UMA_N5_F1_SCHEMA "f1(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor"
#define UMA_N5_F2_SCHEMA "f2(Tensor g, Tensor x, Tensor expand) -> Tensor"
#define UMA_N5_F3_SCHEMA "f3(Tensor x, Tensor W1, Tensor b1, Tensor ln1_w, Tensor ln1_b, Tensor W2, Tensor b2, Tensor ln2_w, Tensor ln2_b, Tensor W3, Tensor b3, float eps) -> Tensor"
#define UMA_N5_F4_SCHEMA "f4(Tensor x, Tensor edge_index, Tensor wigner) -> Tensor"
#define UMA_N5_F5_SCHEMA "f5(Tensor x, Tensor wigner_inv, Tensor edge_index, int n_nodes, int node_offset) -> Tensor"
#define UMA_N5_OP_LIST "f1,f2,f3,f4,f5"
