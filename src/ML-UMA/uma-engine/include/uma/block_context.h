#pragma once

// G7/S7: the full dated design narrative below is retained inline for context; the
// migrated design history lives in `docs/activation_checkpointing.md` (P6.1).
//
// Per-block activation checkpointing for traced UMA inference.
//
// The traced top module (model_traced.pt) records calls to a custom op:
//   uma_ckpt::block(int idx, Tensor x, Tensor edge_distance_vec,
//                   Tensor edge_distance, Tensor atomic_numbers,
//                   Tensor edge_index, Tensor sys_node_emb) -> Tensor x
// Each escn message-passing block (with balance_channels folded in) is also
// exported to disk as model_block_{i}.pt (a TorchScript module) with:
//   forward(x, edge_distance_vec, edge_distance, atomic_numbers, edge_index,
//           sys_node_emb) -> x
//
// KEY MEMORY FIX (design doc option (i), 2026-08-23 "TRUE ROOT of the ~6x
// discrepancy"): the OLD block interface passed the edge-sized loop-INVARIANTS
// wigner [E,25,25] (6.5 GiB), wigner_inv_env (6.5 GiB) and x_edge (4 GiB) as
// inputs, and BlockCheckpointFn.save_for_backward retained ALL of them per
// block -> ~62 GiB OOM at N=18 (~1.4M edges). The block sub-module now takes
// only SMALL precursors (edge_distance_vec [E,3], edge_distance [E]) + ints and
// RECOMPUTES wigner/x_edge INTERNALLY (created and freed within block->forward
// in both the no_grad forward and the backward recompute). Nothing edge-sized
// is ever saved across the forward/backward boundary.
//
// This file wires the op to a process-wide BlockContext that owns the loaded
// per-block sub-modules and runs block idx under BlockCheckpointFn: forward runs
// the sub-module under NoGradGuard (no activations retained), backward recomputes
// the block forward with grad enabled and produces the vjp for every FLOAT input
// that depends on positions. Only ONE block's activations live at once, matching
// the eager AC memory profile (~1/num_layers peak) while staying pure C++.
//
// KEY MEMORY FIX (design doc option (j), 2026-08-23): per-BLOCK checkpoint is
// STILL insufficient. Each block's intra-block edge-chunk loop (~85 chunks at
// N=18) traces as a FLAT sequence (checkpoint_passthrough), so under the C++
// BlockCheckpointFn backward RECOMPUTE (grad ON) ALL ~85 chunks' SO2
// intermediates are retained at once -> 63 GiB OOM. The real memory unit is the
// edge-CHUNK. The fix (option j): each block emits, per edge-chunk, a call to a
// NEW op
//   uma_ckpt::chunk(int block_idx, Tensor x_full, Tensor edge_distance_vec,
//                   Tensor edge_distance, Tensor atomic_numbers,
//                   Tensor edge_index, int node_offset, int mole_start,
//                   int natoms) -> Tensor
// and each block is exported to disk as model_chunk_{i}.pt (a TorchScript module)
// whose forward runs SO2 conv over ONE edge-chunk, scattered back to nodes:
//   chunk_module(x_full, edge_distance_vec, edge_distance, atomic_numbers,
//                edge_index, node_offset:int, mole_start:int)
//       -> partial[natoms,...]
// where edge_distance_vec [Ec,3] and edge_distance [Ec] are the CHUNK precursors
// (small), atomic_numbers [natoms] int, edge_index [2,Ec] int, x_full [natoms,..]
// node-sized. The chunk module builds THIS chunk's wigner [Ec,25,25] (~0.07 GiB)
// INTERNALLY from those precursors -> NO wigner/x_edge is ever passed in.
// The block output is the SUM of the chunk partials (done in the traced block
// graph). C++ wraps EACH chunk in ChunkCheckpointFn -> only ONE chunk is ever
// live: the chunk forward runs under NoGradGuard (one chunk's wigner + SO2
// intermediates built and freed) and its backward recompute is independent per
// chunk. x_full is node-sized; the precursors are chunk-small. Nothing
// edge-sized-huge (no full-edge [E,25,25] wigner) is ever passed OR saved -- this
// removes the LAST full-edge-wigner transient. This is the whole option-(j) fix;
// it matches eager AC exactly.
//
// Forces = -dE/dpos, and pos flows into x (loop-carried), edge_distance_vec,
// edge_distance, and sys_node_emb; so backward must return grads for those four
// float tensors. atomic_numbers and edge_index are integer -> no grad.

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

namespace uma {

// Process-wide registry of the per-block TorchScript sub-modules. Mirrors the
// PeerContext singleton pattern: one instance for the whole process, protected
// by a mutex, storing raw jit modules + the device they live on.
class BlockContext {
 public:
  static BlockContext& instance();

  // Load model_block_{i}.pt for i in [0, num_blocks) from artifact_dir onto
  // `device`, eval() each, and store. Replaces any previously loaded blocks.
  void load_blocks(const std::string& artifact_dir, int num_blocks,
                   torch::Device device);

  void clear();

  int num_blocks() const;
  bool loaded() const;
  torch::Device device() const;

  // Non-owning pointer to block idx (used by BlockCheckpointFn to store an
  // int64 handle in the AutogradContext, mirroring CheckpointModuleFn).
  torch::jit::script::Module* block_ptr(int idx);

  // Option (j): load model_chunk_{i}.pt for i in [0, num_chunks) from
  // artifact_dir onto `device`, eval() each, and store. This is the PRIMARY path
  // now (per-chunk AC). Replaces any previously loaded chunk modules.
  void load_chunks(const std::string& artifact_dir, int num_chunks,
                   torch::Device device);

  int num_chunks() const;
  bool chunks_loaded() const;

  // Non-owning pointer to the chunk module for block idx (used by
  // ChunkCheckpointFn to store an int64 handle in the AutogradContext).
  torch::jit::script::Module* chunk_ptr(int idx);

  // Option (P1-b): load the SINGLE prologue edge_degree chunk module
  // model_edgedeg_chunk.pt from artifact_dir onto `device`, eval() it, and store.
  // EXACTLY analogous to a single chunk module but the module runs
  // edge_degree_embedding.forward_chunk over ONE edge-chunk. Replaces any
  // previously loaded edge_degree module.
  void load_edgedeg(const std::string& artifact_dir, torch::Device device);

  bool edgedeg_loaded() const;

  // Non-owning pointer to the (single) edge_degree chunk module (used by
  // EdgeDegreeCheckpointFn to store an int64 handle in the AutogradContext).
  torch::jit::script::Module* edgedeg_ptr();

 private:
  BlockContext() = default;
  mutable std::mutex mu_;
  std::vector<torch::jit::script::Module> blocks_;
  std::vector<torch::jit::script::Module> chunks_;
  std::vector<torch::jit::script::Module> edgedeg_;
  torch::Device device_ = torch::Device(torch::kCPU);
  bool loaded_ = false;
  bool chunks_loaded_ = false;
  bool edgedeg_loaded_ = false;
};

// Per-block checkpoint autograd Function for the SMALL-precursor escn block
// interface. Differentiates w.r.t. every float input that depends on pos
// (x, edge_distance_vec, edge_distance, sys_node_emb). atomic_numbers and
// edge_index are integer and receive a null grad.
//
// The whole point is memory: save_for_backward saves ONLY the small inputs
// (node-sized x/sys_node_emb, edge-index-sized edge_index, and edge_distance_vec
// [E,3] = ~34 MB / edge_distance [E]). NO edge-sized-huge tensor ([E,25,25]
// wigner et al.) is ever an input, so none is saved; wigner/x_edge are
// recomputed inside block->forward and freed with the block's activations.
struct BlockCheckpointFn
    : public torch::autograd::Function<BlockCheckpointFn> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::jit::script::Module* block,
                               torch::Tensor x,
                               torch::Tensor edge_distance_vec,
                               torch::Tensor edge_distance,
                               torch::Tensor atomic_numbers,
                               torch::Tensor edge_index,
                               torch::Tensor sys_node_emb) {
    ctx->saved_data["block"] = reinterpret_cast<int64_t>(block);
    // Save ONLY the small precursors. Nothing edge-sized-huge is saved: wigner
    // ([E,25,25]) is recomputed inside the block, never an input here.
    ctx->save_for_backward({x, edge_distance_vec, edge_distance, atomic_numbers,
                            edge_index, sys_node_emb});
    torch::NoGradGuard no_grad;  // do not retain the forward graph
    std::vector<torch::jit::IValue> args = {
        x,             edge_distance_vec, edge_distance,
        atomic_numbers, edge_index,       sys_node_emb};
    return block->forward(args).toTensor();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto* block = reinterpret_cast<torch::jit::script::Module*>(
        ctx->saved_data["block"].toInt());
    auto saved = ctx->get_saved_variables();
    // Detach + require grad on the FLOAT pos-derived inputs that need grads for
    // forces. atomic_numbers and edge_index are integer constants -> no grad.
    auto x = saved[0].detach().set_requires_grad(true);
    auto edge_distance_vec = saved[1].detach().set_requires_grad(true);
    auto edge_distance = saved[2].detach().set_requires_grad(true);
    auto atomic_numbers = saved[3];  // integer: constant, no grad
    auto edge_index = saved[4];      // integer: constant, no grad
    auto sys_node_emb = saved[5].detach().set_requires_grad(true);

    torch::Tensor out;
    {
      torch::autograd::AutoGradMode grad_on(true);
      // wigner/x_edge are recomputed INSIDE this forward and freed with the
      // block's activations once grad() returns -> peak == one block.
      std::vector<torch::jit::IValue> args = {
          x,             edge_distance_vec, edge_distance,
          atomic_numbers, edge_index,       sys_node_emb};
      out = block->forward(args).toTensor();
    }

    std::vector<torch::Tensor> inputs = {x, edge_distance_vec, edge_distance,
                                         sys_node_emb};
    auto grads = torch::autograd::grad({out}, inputs, {grad_outputs[0]},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/true);
    auto grad_or_zero = [](const torch::Tensor& g,
                           const torch::Tensor& like) -> torch::Tensor {
      return g.defined() ? g : torch::zeros_like(like);
    };
    // Return order matches forward args: block ptr, x, edge_distance_vec,
    // edge_distance, atomic_numbers, edge_index, sys_node_emb. The two integer
    // inputs (atomic_numbers, edge_index) -> null grad.
    return {torch::Tensor(),
            grad_or_zero(grads[0], x),
            grad_or_zero(grads[1], edge_distance_vec),
            grad_or_zero(grads[2], edge_distance),
            torch::Tensor(),
            torch::Tensor(),
            grad_or_zero(grads[3], sys_node_emb)};
  }
};

// Per-CHUNK checkpoint autograd Function (design doc option (j)). Mirrors
// BlockCheckpointFn but the checkpoint boundary is ONE edge-chunk, so only one
// chunk's SO2 intermediates ever live.
//
// The op uma_ckpt::chunk has INT args (block_idx, node_offset, mole_start,
// natoms) that are NOT tensors. An autograd Function's apply()/forward() only
// meaningfully differentiates its TENSOR arguments, so ChunkCheckpointFn::apply
// takes ONLY the tensor args plus the two ints that chunk_module->forward needs
// (node_offset, mole_start). Those two ints are threaded through ctx->saved_data
// (NOT as autograd inputs) so the backward recompute can pass them to
// chunk_module->forward. block_idx and natoms never reach the module here
// (block_idx selects the module in the autograd shim; natoms only shapes the
// traced sum), so they are not passed to apply at all.
//
// MEMORY: save_for_backward saves ONLY the module-forward inputs -- the FLOAT
// pos-derived precursors (x_full node-sized, edge_distance_vec [Ec,3] and
// edge_distance [Ec], both CHUNK-small) + the integer atomic_numbers/edge_index.
// NO wigner is passed OR saved: the chunk module builds THIS chunk's wigner
// [Ec,25,25] (~0.07 GiB) INTERNALLY, and it is freed with the chunk's SO2
// intermediates once the no_grad forward returns (and rebuilt+freed in the
// backward recompute). This removes the LAST full-edge-wigner transient; every
// saved tensor is chunk-precursor-small or node-sized. Each chunk's forward runs
// under NoGradGuard and its backward recompute is independent, so peak == one
// chunk (== eager AC).
//
// Forces = -dE/dpos: x_full derives from x (block-carried, from pos);
// edge_distance_vec and edge_distance are the chunk slices derived from pos via
// the prologue. The THREE float grads (x_full, edge_distance_vec, edge_distance)
// must return so pos accumulates; x_full's grad chains back to the block input.
// atomic_numbers and edge_index are integer -> no grad.
struct ChunkCheckpointFn
    : public torch::autograd::Function<ChunkCheckpointFn> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::jit::script::Module* chunk_module,
                               torch::Tensor x_full,
                               torch::Tensor edge_distance_vec,
                               torch::Tensor edge_distance,
                               torch::Tensor atomic_numbers,
                               torch::Tensor edge_index,
                               int64_t node_offset,
                               int64_t mole_start) {
    ctx->saved_data["chunk_module"] = reinterpret_cast<int64_t>(chunk_module);
    // Ints threaded via saved_data (NOT autograd inputs); the module forward
    // needs them in both the no_grad forward and the backward recompute.
    ctx->saved_data["node_offset"] = node_offset;
    ctx->saved_data["mole_start"] = mole_start;
    // Save ONLY the module-forward inputs: FLOAT pos-derived precursors
    // (x_full, edge_distance_vec, edge_distance) + integer atomic_numbers/
    // edge_index. NO wigner is saved -- the chunk builds its own [Ec,25,25]
    // wigner internally, so no full-edge wigner is ever an input here.
    ctx->save_for_backward({x_full, edge_distance_vec, edge_distance,
                            atomic_numbers, edge_index});
    torch::NoGradGuard no_grad;  // free this chunk's wigner + SO2 intermediates
    std::vector<torch::jit::IValue> args = {
        x_full,     edge_distance_vec, edge_distance, atomic_numbers,
        edge_index, node_offset,       mole_start};
    return chunk_module->forward(args).toTensor();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto* chunk_module = reinterpret_cast<torch::jit::script::Module*>(
        ctx->saved_data["chunk_module"].toInt());
    const int64_t node_offset = ctx->saved_data["node_offset"].toInt();
    const int64_t mole_start = ctx->saved_data["mole_start"].toInt();
    auto saved = ctx->get_saved_variables();
    // Detach + require grad on the FLOAT pos-derived inputs. atomic_numbers and
    // edge_index are integer -> constants, no grad.
    auto x_full = saved[0].detach().set_requires_grad(true);
    auto edge_distance_vec = saved[1].detach().set_requires_grad(true);
    auto edge_distance = saved[2].detach().set_requires_grad(true);
    auto atomic_numbers = saved[3];  // integer: constant, no grad
    auto edge_index = saved[4];      // integer: constant, no grad

    torch::Tensor out;
    {
      torch::autograd::AutoGradMode grad_on(true);
      // Recompute ONE chunk's wigner + SO2 conv with grad on; freed once grad()
      // returns -> peak == one chunk.
      std::vector<torch::jit::IValue> args = {
          x_full,     edge_distance_vec, edge_distance, atomic_numbers,
          edge_index, node_offset,       mole_start};
      out = chunk_module->forward(args).toTensor();
    }

    std::vector<torch::Tensor> inputs = {x_full, edge_distance_vec,
                                         edge_distance};
    auto grads = torch::autograd::grad({out}, inputs, {grad_outputs[0]},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/true);
    auto grad_or_zero = [](const torch::Tensor& g,
                           const torch::Tensor& like) -> torch::Tensor {
      return g.defined() ? g : torch::zeros_like(like);
    };
    // Return order matches apply() args: chunk_module ptr, x_full,
    // edge_distance_vec, edge_distance, atomic_numbers, edge_index, node_offset,
    // mole_start. Arity == 8 (one entry per apply arg). The module ptr, the two
    // integer tensors (atomic_numbers, edge_index), and the two int args
    // (node_offset, mole_start) receive null grad.
    return {torch::Tensor(),
            grad_or_zero(grads[0], x_full),
            grad_or_zero(grads[1], edge_distance_vec),
            grad_or_zero(grads[2], edge_distance),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  }
};

// Per-CHUNK checkpoint autograd Function for the PROLOGUE edge_degree_embedding
// (design doc P1-b). IDENTICAL structure to ChunkCheckpointFn, but the module
// runs edge_degree_embedding.forward_chunk over ONE edge-chunk and accumulates
// that chunk's scatter INTO x (the [natoms,...] node accumulator), returning the
// updated x. The traced prologue loop emits one uma_ckpt::edge_degree call per
// edge-chunk, so only ONE chunk's SO2/wigner intermediates are ever live.
//
// The op uma_ckpt::edge_degree has INT args (node_offset, mole_start, natoms)
// that are NOT tensors, so EdgeDegreeCheckpointFn::apply takes ONLY the tensor
// args plus the two ints the module forward needs (node_offset, mole_start).
// Those two ints are threaded through ctx->saved_data (NOT as autograd inputs).
// natoms never reaches the module (x is already node-sized), so it is not passed
// to apply at all.
//
// MEMORY: save_for_backward saves ONLY the module-forward inputs -- the FLOAT
// pos-derived precursors (x node-sized, edge_distance_vec [Ec,3] and
// edge_distance [Ec], both CHUNK-small) + the integer atomic_numbers/edge_index.
// NO full-edge wigner is passed OR saved: the module builds THIS chunk's wigner
// [Ec,25,25] INTERNALLY and frees it with the chunk's SO2 intermediates once the
// no_grad forward returns (rebuilt+freed in the backward recompute). This is the
// 12.82 GiB full-edge prologue transient removed -- every saved tensor is
// chunk-precursor-small or node-sized.
//
// Forces = -dE/dpos: x is the node accumulator (grad chains prologue->x_message
// which flows to the blocks); edge_distance_vec and edge_distance are the chunk
// slices derived from pos via the prologue. The THREE float grads (x,
// edge_distance_vec, edge_distance) must return so pos accumulates.
// atomic_numbers and edge_index are integer -> no grad.
struct EdgeDegreeCheckpointFn
    : public torch::autograd::Function<EdgeDegreeCheckpointFn> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::jit::script::Module* edgedeg_module,
                               torch::Tensor x,
                               torch::Tensor edge_distance_vec,
                               torch::Tensor edge_distance,
                               torch::Tensor atomic_numbers,
                               torch::Tensor edge_index,
                               int64_t node_offset,
                               int64_t mole_start) {
    ctx->saved_data["edgedeg_module"] =
        reinterpret_cast<int64_t>(edgedeg_module);
    // Ints threaded via saved_data (NOT autograd inputs); the module forward
    // needs them in both the no_grad forward and the backward recompute.
    ctx->saved_data["node_offset"] = node_offset;
    ctx->saved_data["mole_start"] = mole_start;
    // Save ONLY the module-forward inputs: FLOAT pos-derived precursors
    // (x, edge_distance_vec, edge_distance) + integer atomic_numbers/edge_index.
    // NO wigner is saved -- the module builds this chunk's [Ec,25,25] wigner
    // internally, so no full-edge wigner is ever an input here.
    ctx->save_for_backward({x, edge_distance_vec, edge_distance, atomic_numbers,
                            edge_index});
    torch::NoGradGuard no_grad;  // free this chunk's wigner + SO2 intermediates
    std::vector<torch::jit::IValue> args = {
        x,          edge_distance_vec, edge_distance, atomic_numbers,
        edge_index, node_offset,       mole_start};
    return edgedeg_module->forward(args).toTensor();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto* edgedeg_module = reinterpret_cast<torch::jit::script::Module*>(
        ctx->saved_data["edgedeg_module"].toInt());
    const int64_t node_offset = ctx->saved_data["node_offset"].toInt();
    const int64_t mole_start = ctx->saved_data["mole_start"].toInt();
    auto saved = ctx->get_saved_variables();
    // Detach + require grad on the FLOAT pos-derived inputs. atomic_numbers and
    // edge_index are integer -> constants, no grad.
    auto x = saved[0].detach().set_requires_grad(true);
    auto edge_distance_vec = saved[1].detach().set_requires_grad(true);
    auto edge_distance = saved[2].detach().set_requires_grad(true);
    auto atomic_numbers = saved[3];  // integer: constant, no grad
    auto edge_index = saved[4];      // integer: constant, no grad

    torch::Tensor out;
    {
      torch::autograd::AutoGradMode grad_on(true);
      // Recompute ONE chunk's wigner + edge_degree scatter with grad on; freed
      // once grad() returns -> peak == one chunk.
      std::vector<torch::jit::IValue> args = {
          x,          edge_distance_vec, edge_distance, atomic_numbers,
          edge_index, node_offset,       mole_start};
      out = edgedeg_module->forward(args).toTensor();
    }

    std::vector<torch::Tensor> inputs = {x, edge_distance_vec, edge_distance};
    auto grads = torch::autograd::grad({out}, inputs, {grad_outputs[0]},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/true);
    auto grad_or_zero = [](const torch::Tensor& g,
                           const torch::Tensor& like) -> torch::Tensor {
      return g.defined() ? g : torch::zeros_like(like);
    };
    // Return order matches apply() args: edgedeg_module ptr, x,
    // edge_distance_vec, edge_distance, atomic_numbers, edge_index, node_offset,
    // mole_start. Arity == 8 (one entry per apply arg). The module ptr, the two
    // integer tensors (atomic_numbers, edge_index), and the two int args
    // (node_offset, mole_start) receive null grad.
    return {torch::Tensor(),
            grad_or_zero(grads[0], x),
            grad_or_zero(grads[1], edge_distance_vec),
            grad_or_zero(grads[2], edge_distance),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  }
};

// Registered as the Autograd kernel for uma_ckpt::chunk. Looks up chunks_[idx]
// and runs it under ChunkCheckpointFn. The int args block_idx/node_offset/
// mole_start/natoms are NOT tensors: block_idx selects the module, node_offset/
// mole_start are threaded into ChunkCheckpointFn (module needs them), and natoms
// only shapes the node-sized partial (unused by the C++ Function).
torch::Tensor uma_ckpt_chunk_autograd(int64_t block_idx,
                                      const torch::Tensor& x_full,
                                      const torch::Tensor& edge_distance_vec,
                                      const torch::Tensor& edge_distance,
                                      const torch::Tensor& atomic_numbers,
                                      const torch::Tensor& edge_index,
                                      int64_t node_offset, int64_t mole_start,
                                      int64_t natoms);

// Registered as the Autograd kernel for uma_ckpt::edge_degree. Looks up the
// single edgedeg module and runs it under EdgeDegreeCheckpointFn. The int args
// node_offset/mole_start/natoms are NOT tensors: node_offset/mole_start are
// threaded into EdgeDegreeCheckpointFn (module needs them), and natoms only
// shaped the traced node-sized accumulator (unused by the C++ Function; x is
// already node-sized).
torch::Tensor uma_ckpt_edge_degree_autograd(
    const torch::Tensor& x, const torch::Tensor& edge_distance_vec,
    const torch::Tensor& edge_distance, const torch::Tensor& atomic_numbers,
    const torch::Tensor& edge_index, int64_t node_offset, int64_t mole_start,
    int64_t natoms);

// Registered as the Autograd kernel for uma_ckpt::block. Looks up blocks_[idx]
// and runs it under BlockCheckpointFn.
torch::Tensor uma_ckpt_block_autograd(int64_t idx, const torch::Tensor& x,
                                      const torch::Tensor& edge_distance_vec,
                                      const torch::Tensor& edge_distance,
                                      const torch::Tensor& atomic_numbers,
                                      const torch::Tensor& edge_index,
                                      const torch::Tensor& sys_node_emb);

// Convenience: load blocks if model_block_0.pt exists in artifact_dir. Reads
// num_blocks from metadata.json ("num_blocks") when present, else counts
// model_block_*.pt files. No-op (returns false) when no block files exist.
bool maybe_load_blocks(const std::string& artifact_dir, torch::Device device);

// Option (j) PRIMARY path: load chunk modules if model_chunk_0.pt exists in
// artifact_dir. Reads num_blocks from metadata.json ("num_blocks") when present,
// else counts model_chunk_*.pt files. No-op (returns false) when none exist.
bool maybe_load_chunks(const std::string& artifact_dir, torch::Device device);

// Option (P1-b): load the single prologue edge_degree chunk module if
// model_edgedeg_chunk.pt exists in artifact_dir. Defensive: no-op (returns
// false) when the file is absent, so artifacts without a checkpointed prologue
// keep the un-checkpointed full-edge prologue path.
bool maybe_load_edgedeg(const std::string& artifact_dir, torch::Device device);

}  // namespace uma
