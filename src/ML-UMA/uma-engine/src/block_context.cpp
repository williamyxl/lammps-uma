#include "uma/block_context.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace uma {

namespace {
// Perf knobs: bypass the activation checkpoint for a given op and run its
// sub-module forward DIRECTLY under autograd (retain activations, no backward
// recompute). Removes that op's ~forward-sized recompute cost but retains its
// activations -> higher HBM. Numerically identical (same graph, retained vs
// recomputed). Default OFF (checkpointing on) so large systems that need AC to
// fit are unaffected.
//
// GRANULAR so the memory/speed tradeoff can be tuned per op level. Measured at
// N=32 W=12: the CHUNK activations (SO2 + [Ec,25,25] wigner x ~11 chunks x 4
// blocks) are the memory blocker -- full UMA_NO_RECOMPUTE=1 OOMs (62.5/64 GiB).
// The BLOCK and EDGE_DEGREE recomputes retain only node-sized activations, so
// skipping THOSE reclaims recompute time without the chunk memory blowup.
//   UMA_NO_RECOMPUTE=1        -> master: all three (block+chunk+edge_degree)
//   UMA_NO_RECOMPUTE_BLOCK=1  -> block only
//   UMA_NO_RECOMPUTE_CHUNK=1  -> chunk only (memory-heavy; use with care)
//   UMA_NO_RECOMPUTE_EDEG=1   -> edge_degree prologue only
// Each is evaluated once and cached.
bool env_flag_1(const char* name) {
  const char* e = std::getenv(name);
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}
bool no_recompute_block() {
  static const bool v =
      env_flag_1("UMA_NO_RECOMPUTE") || env_flag_1("UMA_NO_RECOMPUTE_BLOCK");
  return v;
}
bool no_recompute_chunk() {
  static const bool v =
      env_flag_1("UMA_NO_RECOMPUTE") || env_flag_1("UMA_NO_RECOMPUTE_CHUNK");
  return v;
}
bool no_recompute_edeg() {
  static const bool v =
      env_flag_1("UMA_NO_RECOMPUTE") || env_flag_1("UMA_NO_RECOMPUTE_EDEG");
  return v;
}

// opt5 P-1: SELECTIVE per-chunk retain. Full UMA_NO_RECOMPUTE_CHUNK=1 OOMs
// (retains every chunk's SO2 + [Ec,25,25] wigner at once). UMA_CHUNK_RETAIN_K=k
// retains only the first k edge-chunks PER BLOCK under autograd (no backward
// recompute) and checkpoints the rest -- a tunable memory/speed knob that claws
// back part of the chunk-recompute cost within the HBM budget. k<=0 (default) =>
// legacy (all chunks checkpointed, unless UMA_NO_RECOMPUTE_CHUNK forces all).
// Numerically identical either way (retain vs recompute = same gradient).
int chunk_retain_k() {
  static const int v = [] {
    const char* e = std::getenv("UMA_CHUNK_RETAIN_K");
    if (e == nullptr) return 0;
    try { return std::max(0, std::stoi(e)); } catch (...) { return 0; }
  }();
  return v;
}

// Per-(forward,block) chunk index counter. The traced top graph emits the block
// loop in order (block 0 chunks 0..C, block 1 chunks 0..C, ...); we count chunk
// calls per block_idx and reset a block's counter whenever it is (re)entered
// after a different block, so the first k chunks of each block are retained.
// thread_local: each rank's predict runs on one thread; backward reuses the
// forward's retained graph so the counter only needs to gate the FORWARD.
thread_local int g_last_block_idx = -1;
thread_local int g_chunk_in_block = 0;
bool retain_this_chunk(int block_idx) {
  const int k = chunk_retain_k();
  if (k <= 0) return false;
  if (block_idx != g_last_block_idx) {
    g_last_block_idx = block_idx;
    g_chunk_in_block = 0;
  }
  const bool retain = g_chunk_in_block < k;
  ++g_chunk_in_block;
  return retain;
}
}  // namespace

BlockContext& BlockContext::instance() {
  static BlockContext ctx;
  return ctx;
}

void BlockContext::load_blocks(const std::string& artifact_dir, int num_blocks,
                               torch::Device device) {
  if (num_blocks < 0) {
    throw std::runtime_error("BlockContext::load_blocks: num_blocks < 0");
  }
  std::lock_guard<std::mutex> lk(mu_);
  blocks_.clear();
  blocks_.reserve(static_cast<size_t>(num_blocks));
  device_ = device;
  for (int i = 0; i < num_blocks; ++i) {
    const std::string path =
        artifact_dir + "/model_block_" + std::to_string(i) + ".pt";
    auto m = torch::jit::load(path, device);
    m.eval();
    m.to(device);
    blocks_.push_back(std::move(m));
  }
  loaded_ = num_blocks > 0;
}

void BlockContext::load_chunks(const std::string& artifact_dir, int num_chunks,
                               torch::Device device) {
  if (num_chunks < 0) {
    throw std::runtime_error("BlockContext::load_chunks: num_chunks < 0");
  }
  std::lock_guard<std::mutex> lk(mu_);
  chunks_.clear();
  chunks_.reserve(static_cast<size_t>(num_chunks));
  device_ = device;
  for (int i = 0; i < num_chunks; ++i) {
    const std::string path =
        artifact_dir + "/model_chunk_" + std::to_string(i) + ".pt";
    auto m = torch::jit::load(path, device);
    m.eval();
    m.to(device);
    chunks_.push_back(std::move(m));
  }
  chunks_loaded_ = num_chunks > 0;
}

void BlockContext::load_edgedeg(const std::string& artifact_dir,
                                torch::Device device) {
  std::lock_guard<std::mutex> lk(mu_);
  edgedeg_.clear();
  edgedeg_.reserve(1);
  device_ = device;
  const std::string path = artifact_dir + "/model_edgedeg_chunk.pt";
  auto m = torch::jit::load(path, device);
  m.eval();
  m.to(device);
  edgedeg_.push_back(std::move(m));
  edgedeg_loaded_ = true;
}

void BlockContext::clear() {
  std::lock_guard<std::mutex> lk(mu_);
  blocks_.clear();
  chunks_.clear();
  edgedeg_.clear();
  loaded_ = false;
  chunks_loaded_ = false;
  edgedeg_loaded_ = false;
}

int BlockContext::num_blocks() const {
  std::lock_guard<std::mutex> lk(mu_);
  return static_cast<int>(blocks_.size());
}

bool BlockContext::loaded() const {
  std::lock_guard<std::mutex> lk(mu_);
  return loaded_;
}

torch::Device BlockContext::device() const {
  std::lock_guard<std::mutex> lk(mu_);
  return device_;
}

torch::jit::script::Module* BlockContext::block_ptr(int idx) {
  std::lock_guard<std::mutex> lk(mu_);
  if (idx < 0 || idx >= static_cast<int>(blocks_.size())) {
    throw std::runtime_error("BlockContext::block_ptr: idx " +
                             std::to_string(idx) + " out of range (have " +
                             std::to_string(blocks_.size()) + " blocks)");
  }
  return &blocks_[static_cast<size_t>(idx)];
}

int BlockContext::num_chunks() const {
  std::lock_guard<std::mutex> lk(mu_);
  return static_cast<int>(chunks_.size());
}

bool BlockContext::chunks_loaded() const {
  std::lock_guard<std::mutex> lk(mu_);
  return chunks_loaded_;
}

torch::jit::script::Module* BlockContext::chunk_ptr(int idx) {
  std::lock_guard<std::mutex> lk(mu_);
  if (idx < 0 || idx >= static_cast<int>(chunks_.size())) {
    throw std::runtime_error("BlockContext::chunk_ptr: idx " +
                             std::to_string(idx) + " out of range (have " +
                             std::to_string(chunks_.size()) + " chunk modules)");
  }
  return &chunks_[static_cast<size_t>(idx)];
}

bool BlockContext::edgedeg_loaded() const {
  std::lock_guard<std::mutex> lk(mu_);
  return edgedeg_loaded_;
}

torch::jit::script::Module* BlockContext::edgedeg_ptr() {
  std::lock_guard<std::mutex> lk(mu_);
  if (edgedeg_.empty()) {
    throw std::runtime_error(
        "BlockContext::edgedeg_ptr: edge_degree module not loaded");
  }
  return &edgedeg_[0];
}

torch::Tensor uma_ckpt_block_autograd(int64_t idx, const torch::Tensor& x,
                                      const torch::Tensor& edge_distance_vec,
                                      const torch::Tensor& edge_distance,
                                      const torch::Tensor& atomic_numbers,
                                      const torch::Tensor& edge_index,
                                      const torch::Tensor& sys_node_emb) {
  auto* block = BlockContext::instance().block_ptr(static_cast<int>(idx));
  if (no_recompute_block()) {
    // Retain-activations fast path: run the block forward under autograd (no
    // NoGradGuard, no checkpoint) so backward reuses retained activations
    // instead of recomputing. Same math, more memory, less compute.
    std::vector<torch::jit::IValue> args = {
        x, edge_distance_vec, edge_distance, atomic_numbers, edge_index,
        sys_node_emb};
    return block->forward(args).toTensor();
  }
  return BlockCheckpointFn::apply(block, x, edge_distance_vec, edge_distance,
                                  atomic_numbers, edge_index, sys_node_emb);
}

torch::Tensor uma_ckpt_chunk_autograd(int64_t block_idx,
                                      const torch::Tensor& x_full,
                                      const torch::Tensor& edge_distance_vec,
                                      const torch::Tensor& edge_distance,
                                      const torch::Tensor& atomic_numbers,
                                      const torch::Tensor& edge_index,
                                      int64_t node_offset, int64_t mole_start,
                                      int64_t /*natoms*/) {
  auto* chunk_module =
      BlockContext::instance().chunk_ptr(static_cast<int>(block_idx));
  // block_idx selects the module; node_offset/mole_start are threaded into the
  // Function (module needs them); natoms only shaped the traced node-sized sum.
  // wigner/x_edge are NO LONGER passed: the chunk module recomputes them from
  // the per-chunk precursors (edge_distance_vec, edge_distance, atomic_numbers)
  // INTERNALLY, so no full-edge wigner transient ever crosses this boundary.
  // opt5 P-1: retain this chunk (run directly under autograd, no backward
  // recompute) if either the global chunk-retain flag is set (all chunks) OR the
  // selective budget UMA_CHUNK_RETAIN_K retains the first k chunks of this block.
  if (no_recompute_chunk() || retain_this_chunk(static_cast<int>(block_idx))) {
    // Retain-activations fast path. Retains this chunk's SO2 intermediates +
    // [Ec,25,25] wigner (the memory per-chunk AC avoids), so only safe within the
    // HBM budget -- UMA_CHUNK_RETAIN_K caps how many chunks/block are retained.
    // Same math as the checkpoint (retain vs recompute = identical gradient).
    std::vector<torch::jit::IValue> args = {
        x_full,     edge_distance_vec, edge_distance, atomic_numbers,
        edge_index, node_offset,       mole_start};
    return chunk_module->forward(args).toTensor();
  }
  return ChunkCheckpointFn::apply(chunk_module, x_full, edge_distance_vec,
                                  edge_distance, atomic_numbers, edge_index,
                                  node_offset, mole_start);
}

torch::Tensor uma_ckpt_edge_degree_autograd(
    const torch::Tensor& x, const torch::Tensor& edge_distance_vec,
    const torch::Tensor& edge_distance, const torch::Tensor& atomic_numbers,
    const torch::Tensor& edge_index, int64_t node_offset, int64_t mole_start,
    int64_t /*natoms*/) {
  auto* edgedeg_module = BlockContext::instance().edgedeg_ptr();
  // node_offset/mole_start are threaded into the Function (module needs them);
  // natoms only shaped the traced node-sized accumulator (x is already
  // node-sized). wigner/x_edge are NOT passed: the module recomputes them from
  // the per-chunk precursors (edge_distance_vec, edge_distance, atomic_numbers)
  // INTERNALLY, so no full-edge wigner transient ever crosses this boundary.
  if (no_recompute_edeg()) {
    // Retain-activations fast path (see no_recompute_edeg). The prologue
    // accumulates into x; running it directly under autograd retains its
    // intermediates for backward instead of recomputing. Same math.
    std::vector<torch::jit::IValue> args = {
        x,          edge_distance_vec, edge_distance, atomic_numbers,
        edge_index, node_offset,       mole_start};
    return edgedeg_module->forward(args).toTensor();
  }
  return EdgeDegreeCheckpointFn::apply(edgedeg_module, x, edge_distance_vec,
                                       edge_distance, atomic_numbers, edge_index,
                                       node_offset, mole_start);
}

namespace {

bool file_exists(const std::string& path) {
  std::ifstream in(path);
  return in.good();
}

// Minimal JSON int lookup (mirrors metadata.cpp parse_json_int) without adding
// a dependency; returns -1 when the key is absent or unparseable.
int parse_optional_json_int(const std::string& json, const std::string& key) {
  const auto pos = json.find("\"" + key + "\":");
  if (pos == std::string::npos) return -1;
  auto start = json.find_first_of("0123456789-", pos + key.size() + 3);
  if (start == std::string::npos) return -1;
  try {
    return static_cast<int>(std::stol(json.substr(start)));
  } catch (const std::exception&) {
    return -1;
  }
}

int count_block_files(const std::string& artifact_dir) {
  int n = 0;
  while (file_exists(artifact_dir + "/model_block_" + std::to_string(n) +
                     ".pt")) {
    ++n;
  }
  return n;
}

int count_chunk_files(const std::string& artifact_dir) {
  int n = 0;
  while (file_exists(artifact_dir + "/model_chunk_" + std::to_string(n) +
                     ".pt")) {
    ++n;
  }
  return n;
}

int read_num_blocks(const std::string& artifact_dir) {
  const std::string metadata_path = artifact_dir + "/metadata.json";
  std::ifstream in(metadata_path);
  if (in.good()) {
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    const int meta_n = parse_optional_json_int(json, "num_blocks");
    if (meta_n > 0) return meta_n;
  }
  return count_block_files(artifact_dir);
}

// Option (j): one chunk module per block, so the count is num_blocks. Prefer
// metadata num_blocks, else scan model_chunk_*.pt.
int read_num_chunks(const std::string& artifact_dir) {
  const std::string metadata_path = artifact_dir + "/metadata.json";
  std::ifstream in(metadata_path);
  if (in.good()) {
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    const int meta_n = parse_optional_json_int(json, "num_blocks");
    if (meta_n > 0) return meta_n;
  }
  return count_chunk_files(artifact_dir);
}

}  // namespace

bool maybe_load_blocks(const std::string& artifact_dir, torch::Device device) {
  if (!file_exists(artifact_dir + "/model_block_0.pt")) {
    return false;  // no per-block artifacts; legacy monolithic path
  }
  const int num_blocks = read_num_blocks(artifact_dir);
  if (num_blocks <= 0) return false;
  BlockContext::instance().load_blocks(artifact_dir, num_blocks, device);
  return true;
}

bool maybe_load_chunks(const std::string& artifact_dir, torch::Device device) {
  if (!file_exists(artifact_dir + "/model_chunk_0.pt")) {
    return false;  // no per-chunk artifacts; fall back to block/monolithic path
  }
  const int num_chunks = read_num_chunks(artifact_dir);
  if (num_chunks <= 0) return false;
  BlockContext::instance().load_chunks(artifact_dir, num_chunks, device);
  return true;
}

bool maybe_load_edgedeg(const std::string& artifact_dir, torch::Device device) {
  if (!file_exists(artifact_dir + "/model_edgedeg_chunk.pt")) {
    return false;  // no checkpointed prologue; un-checkpointed full-edge path
  }
  BlockContext::instance().load_edgedeg(artifact_dir, device);
  return true;
}

}  // namespace uma

// Custom op: the traced top module calls this per message-passing block. The
// integer idx selects the loaded sub-module; the six tensors are the escn block
// interface with SMALL edge precursors (wigner/x_edge are recomputed inside the
// block, never passed as inputs). Registered on the Autograd key so per-block
// checkpointing composes with the traced graph's autograd (forces flow back
// through x/edge_distance_vec/edge_distance/sys_node_emb; atomic_numbers and
// edge_index are integer).
TORCH_LIBRARY(uma_ckpt, m) {
  m.def(
      "block(int idx, Tensor x, Tensor edge_distance_vec, "
      "Tensor edge_distance, Tensor atomic_numbers, Tensor edge_index, "
      "Tensor sys_node_emb) -> Tensor");
  // Option (j) per-CHUNK op. The traced block loop emits one call per edge-chunk.
  // int args block_idx/node_offset/mole_start/natoms are NOT tensors; the three
  // FLOAT tensors (x_full, edge_distance_vec, edge_distance) carry pos grads;
  // atomic_numbers and edge_index are integer. edge_distance_vec [Ec,3] and
  // edge_distance [Ec] are the CHUNK precursors (small); the chunk module builds
  // this chunk's wigner [Ec,25,25] INTERNALLY, so NO wigner/x_edge is passed and
  // no full-edge wigner transient ever crosses this boundary.
  m.def(
      "chunk(int block_idx, Tensor x_full, Tensor edge_distance_vec, "
      "Tensor edge_distance, Tensor atomic_numbers, Tensor edge_index, "
      "int node_offset, int mole_start, int natoms) -> Tensor");
  // Option (P1-b) PROLOGUE per-CHUNK op. The traced prologue loop emits one call
  // per edge-chunk. Same shape as chunk (minus block_idx: the prologue has a
  // single edge_degree module). int args node_offset/mole_start/natoms are NOT
  // tensors; the three FLOAT tensors (x, edge_distance_vec, edge_distance) carry
  // pos grads; atomic_numbers and edge_index are integer. The module builds this
  // chunk's wigner [Ec,25,25] INTERNALLY and accumulates the scatter INTO x,
  // returning updated x -- so NO full-edge wigner transient (the 12.82 GiB
  // prologue alloc) ever crosses this boundary.
  m.def(
      "edge_degree(Tensor x, Tensor edge_distance_vec, "
      "Tensor edge_distance, Tensor atomic_numbers, Tensor edge_index, "
      "int node_offset, int mole_start, int natoms) -> Tensor");
}

TORCH_LIBRARY_IMPL(uma_ckpt, Autograd, m) {
  m.impl("block", TORCH_FN(uma::uma_ckpt_block_autograd));
  m.impl("chunk", TORCH_FN(uma::uma_ckpt_chunk_autograd));
  m.impl("edge_degree", TORCH_FN(uma::uma_ckpt_edge_degree_autograd));
}
