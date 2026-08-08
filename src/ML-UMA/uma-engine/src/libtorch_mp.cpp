#include "uma/libtorch_mp.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <sys/stat.h>  // mkdir

#if defined(UMA_ENGINE_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "uma/graph_shard.h"
#include "uma/kokkos_peer.h"
#include "uma/neighbor_list.h"
#include "uma/payload_shm.h"
#include "uma/peer_context.h"
#include "uma/postprocess.h"
#include "uma/shared_peer.h"
#include "uma/vesin_nl.h"

namespace uma {
namespace {

bool file_exists(const std::string& path) {
  std::ifstream in(path);
  return static_cast<bool>(in);
}

void disable_torchscript_texpr_once() {
  static const bool done = [] {
    torch::jit::setTensorExprFuserEnabled(false);
    torch::jit::setGraphExecutorOptimize(false);
    return true;
  }();
  (void)done;
}

void write_all(int fd, const void* src, size_t n) {
  const char* p = static_cast<const char*>(src);
  size_t off = 0;
  while (off < n) {
    const ssize_t w = ::write(fd, p + off, n - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("write_all: ") + std::strerror(errno));
    }
    if (w == 0) throw std::runtime_error("write_all: EOF");
    off += static_cast<size_t>(w);
  }
}

void read_all(int fd, void* dst, size_t n) {
  char* p = static_cast<char*>(dst);
  size_t off = 0;
  while (off < n) {
    const ssize_t r = ::read(fd, p + off, n - off);
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("read_all: ") + std::strerror(errno));
    }
    if (r == 0) throw std::runtime_error("read_all: EOF");
    off += static_cast<size_t>(r);
  }
}

}  // namespace

bool LibtorchMpRuntime::artifacts_present(const std::string& artifact_dir,
                                          int num_devices) {
  if (num_devices < 2) return false;
  // Legacy wN_rR or n-specific wN_n{N}_rR (via UMA_MP_NATOMS).
  std::string ntag;
  if (const char* e = std::getenv("UMA_MP_NATOMS")) {
    if (*e) ntag = std::string("_n") + e;
  }
  for (int r = 0; r < num_devices; ++r) {
    const std::string p_n = artifact_dir + "/model_mp_w" + std::to_string(num_devices) +
                            ntag + "_r" + std::to_string(r) + ".pt";
    const std::string p = artifact_dir + "/model_mp_w" + std::to_string(num_devices) +
                          "_r" + std::to_string(r) + ".pt";
    if (!file_exists(p_n) && !file_exists(p)) return false;
  }
  return true;
}

struct LibtorchMpRuntime::Worker {
  pid_t pid = -1;
  int to_child = -1;
  int from_child = -1;
};

// Out-of-line storage for workers / shared slot (keep header small).
struct LibtorchMpRuntime::Impl {
  std::string artifact_dir;
  std::string shm_path;
  std::string payload_path;
  kokkos_peer::SharedPeerGatherSlot* shared = nullptr;
  PayloadShm* payload = nullptr;
  std::vector<Worker> workers;
  int64_t part_check_n = -1;
  int64_t part_check_e = -1;
};

std::unique_ptr<LibtorchMpRuntime> LibtorchMpRuntime::try_create(
    const std::string& artifact_dir, const ArtifactMetadata& metadata,
    int num_devices, torch::ScalarType compute_dtype) {
  if (num_devices < 2) {
    throw std::runtime_error("LibtorchMpRuntime requires num_devices > 1");
  }
  if (compute_dtype != torch::kFloat64) {
    throw std::runtime_error("LibtorchMpRuntime: FP64 only");
  }
  if (!artifacts_present(artifact_dir, num_devices)) {
    return nullptr;
  }

  register_uma_peer_ops();
  disable_torchscript_texpr_once();

  auto rt = std::unique_ptr<LibtorchMpRuntime>(
      new LibtorchMpRuntime(num_devices, metadata, compute_dtype));
  rt->impl_ = std::make_unique<Impl>();
  rt->impl_->artifact_dir = artifact_dir;

  // File-backed shm so workers can exec+mmap (anonymous shm does not survive exec).
  rt->impl_->shm_path =
      "/dev/shm/uma_mp_" + std::to_string(static_cast<int>(getpid())) + "_" +
      std::to_string(num_devices);
  {
    const int transport = kokkos_peer::SharedPeerGatherSlot::select_transport();
    const size_t bytes =
        kokkos_peer::SharedPeerGatherSlot::map_bytes_for(num_devices, transport);
    int fd = ::open(rt->impl_->shm_path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
      throw std::runtime_error(std::string("shm open: ") + std::strerror(errno));
    }
    if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      close(fd);
      throw std::runtime_error("shm ftruncate failed");
    }
    void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) {
      throw std::runtime_error("shm mmap failed");
    }
    auto* shm = static_cast<kokkos_peer::SharedPeerGatherSlot::Shm*>(mem);
    std::memset(shm, 0, bytes);
    shm->world = num_devices;
    shm->transport = transport;
    pthread_mutexattr_t mattr;
    pthread_condattr_t cattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_init(&cattr);
    pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&shm->mu, &mattr);
    pthread_cond_init(&shm->cv, &cattr);
    pthread_mutexattr_destroy(&mattr);
    pthread_condattr_destroy(&cattr);
    rt->impl_->shared =
        kokkos_peer::SharedPeerGatherSlot::attach(shm, bytes);  // parent unmaps on destroy
    std::cerr << "uma LibtorchMpRuntime: peer_transport="
              << kokkos_peer::SharedPeerGatherSlot::transport_name(transport)
              << " shm_bytes=" << bytes << "\n";
  }
  // Geometry/edge payload shm (pipe-tax cut).
  {
    rt->impl_->payload_path = rt->impl_->shm_path + "_payload";
    rt->impl_->payload = PayloadShm::create_file(rt->impl_->payload_path.c_str());
    rt->impl_->payload->hdr->world = num_devices;
    std::cerr << "uma LibtorchMpRuntime: payload_shm_bytes="
              << rt->impl_->payload->bytes << "\n";
  }
  PeerContext::instance().reset_shared(rt->impl_->shared);

  // Resolve worker binary next to this process or via env.
  std::string worker_bin;
  if (const char* e = std::getenv("UMA_LIBTORCH_MP_WORKER")) {
    worker_bin = e;
  } else {
#ifdef UMA_ENGINE_PYTHON_DIR
    // uma-engine/python -> uma-engine/build-cpp-mp/uma_libtorch_mp_worker (best-effort)
    worker_bin = std::string(UMA_ENGINE_PYTHON_DIR) +
                 "/../build-cpp-mp/uma_libtorch_mp_worker";
#endif
    if (worker_bin.empty() || !file_exists(worker_bin)) {
      worker_bin = "uma_libtorch_mp_worker";
    }
  }

  rt->impl_->workers.resize(static_cast<size_t>(num_devices));
  for (int r = 0; r < num_devices; ++r) {
    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
      throw std::runtime_error("LibtorchMpRuntime: pipe failed");
    }
    const pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("LibtorchMpRuntime: fork failed");
    if (pid == 0) {
      // Fresh CUDA context via exec (not fork-inherited).
      dup2(to_child[0], STDIN_FILENO);
      dup2(from_child[1], STDOUT_FILENO);
      close(to_child[0]);
      close(to_child[1]);
      close(from_child[0]);
      close(from_child[1]);
      const std::string rank_s = std::to_string(r);
      const std::string world_s = std::to_string(num_devices);
      const std::string bytes_s =
          std::to_string(rt->impl_->shared->map_bytes());
      // Each worker sees a single GPU as cuda:0 (matches export bake).
      ::setenv("CUDA_VISIBLE_DEVICES", rank_s.c_str(), 1);
      // CUDA_LAUNCH_BLOCKING is a large N>1 tax; opt-in for assert debug only.
      // UMA_CUDA_LAUNCH_BLOCKING=1 → force "1"; =0 → unset; unset → leave env as-is
      // (default: do NOT force on — required for multi-GPU self-scaling).
      if (const char* blk = std::getenv("UMA_CUDA_LAUNCH_BLOCKING")) {
        if (blk[0] == '1' && blk[1] == '\0') {
          ::setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
        } else if (blk[0] == '0' && blk[1] == '\0') {
          ::unsetenv("CUDA_LAUNCH_BLOCKING");
        }
      } else {
        ::unsetenv("CUDA_LAUNCH_BLOCKING");
      }
      // Per-rank stderr log (CUDA asserts / TORCH errors).
      std::string log_dir = "/tmp/uma_mp_logs";
      if (const char* e = std::getenv("UMA_MP_LOG_DIR")) {
        if (*e) log_dir = e;
      }
      ::setenv("UMA_MP_LOG_DIR", log_dir.c_str(), 1);
      ::setenv("UMA_MP_PAYLOAD_SHM", rt->impl_->payload_path.c_str(), 1);
      ::setenv("UMA_MP_PAYLOAD_BYTES",
               std::to_string(rt->impl_->payload->bytes).c_str(), 1);
      {
        ::mkdir(log_dir.c_str(), 0755);  // best-effort; may already exist
        const std::string log_path = log_dir + "/worker_r" + rank_s + ".log";
        int logfd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (logfd >= 0) {
          dup2(logfd, STDERR_FILENO);
          close(logfd);
        }
      }
      execl(worker_bin.c_str(), worker_bin.c_str(), rank_s.c_str(), world_s.c_str(),
            artifact_dir.c_str(), rt->impl_->shm_path.c_str(), bytes_s.c_str(),
            static_cast<char*>(nullptr));
      std::cerr << "execl " << worker_bin << " failed: " << std::strerror(errno) << "\n";
      _exit(127);
    }
    close(to_child[0]);
    close(from_child[1]);
    rt->impl_->workers[static_cast<size_t>(r)].pid = pid;
    rt->impl_->workers[static_cast<size_t>(r)].to_child = to_child[1];
    rt->impl_->workers[static_cast<size_t>(r)].from_child = from_child[0];
  }

  for (int r = 0; r < num_devices; ++r) {
    uint8_t ready = 0;
    read_all(rt->impl_->workers[static_cast<size_t>(r)].from_child, &ready, 1);
    if (!ready) {
      int32_t mlen = 0;
      std::string msg;
      try {
        read_all(rt->impl_->workers[static_cast<size_t>(r)].from_child, &mlen,
                 sizeof(mlen));
        if (mlen > 0 && mlen < 100000) {
          msg.resize(static_cast<size_t>(mlen));
          read_all(rt->impl_->workers[static_cast<size_t>(r)].from_child, msg.data(),
                   static_cast<size_t>(mlen));
        }
      } catch (...) {
        msg = "(no detail; is uma_libtorch_mp_worker on PATH / UMA_LIBTORCH_MP_WORKER?)";
      }
      throw std::runtime_error("LibtorchMpRuntime: worker " + std::to_string(r) +
                               " failed ready: " + msg);
    }
  }

  std::cerr << "uma LibtorchMpRuntime: backend=" << rt->backend_
            << " devices=" << num_devices << " (process-per-rank exec)"
            << " artifact=" << artifact_dir << "\n";
  (void)metadata;
  return rt;
}

LibtorchMpRuntime::LibtorchMpRuntime(int num_devices, ArtifactMetadata metadata,
                                     torch::ScalarType compute_dtype)
    : num_devices_(num_devices),
      metadata_(std::move(metadata)),
      compute_dtype_(compute_dtype) {
  if (metadata_.element_references.defined()) {
    element_refs_ =
        metadata_.element_references.to(torch::kCPU, compute_dtype_).contiguous();
  }
}

LibtorchMpRuntime::~LibtorchMpRuntime() {
  if (!impl_) return;
  for (auto& w : impl_->workers) {
    if (w.to_child >= 0) {
      try {
        int32_t cmd = 0;
        write_all(w.to_child, &cmd, sizeof(cmd));
      } catch (...) {
      }
      close(w.to_child);
      w.to_child = -1;
    }
    if (w.from_child >= 0) {
      close(w.from_child);
      w.from_child = -1;
    }
    if (w.pid > 0) {
      int status = 0;
      waitpid(w.pid, &status, 0);
      w.pid = -1;
    }
  }
  if (impl_->shared) {
    auto* shm = impl_->shared->raw();
    const size_t nbytes = impl_->shared->map_bytes();
    impl_->shared->destroy();  // deletes wrapper only (owns=false)
    if (shm) {
      pthread_mutex_destroy(&shm->mu);
      pthread_cond_destroy(&shm->cv);
      munmap(shm, nbytes);
    }
    impl_->shared = nullptr;
  }
  if (impl_->payload) {
    impl_->payload->destroy();
    impl_->payload = nullptr;
  }
  if (!impl_->shm_path.empty()) {
    ::unlink(impl_->shm_path.c_str());
  }
  if (!impl_->payload_path.empty()) {
    ::unlink(impl_->payload_path.c_str());
  }
  PeerContext::instance().clear();
}

void LibtorchMpRuntime::rebuild_neighbors_full(torch::Device build_dev) {
#if defined(VESIN_ROOT)
  if (build_dev.is_cuda() && torch::cuda::is_available() && pos0_.is_cuda()) {
    auto vg = vesin_nl::vesin_build_graph_cuda(
        pos0_, cell0_, pbc0_, metadata_.cutoff, metadata_.max_neighbors,
        /*full_directed=*/true, compute_dtype_);
    auto center_i = vg.edge_index.index({0});
    auto neighbor_j = vg.edge_index.index({1});
    edge_index_ = torch::stack({neighbor_j, center_i}, 0).contiguous();
    cell_offsets_ = vg.shifts.to(build_dev, compute_dtype_).contiguous();
    return;
  }
#endif
  NeighborListConfig config;
  config.cutoff = metadata_.cutoff;
  config.max_neighbors = metadata_.max_neighbors;
  auto graph = build_neighbor_graph(pos0_.to(torch::kCPU), cell0_.to(torch::kCPU),
                                    pbc0_.to(torch::kCPU), config);
  edge_index_ = graph.edge_index.to(build_dev);
  cell_offsets_ = graph.cell_offsets.to(build_dev, compute_dtype_);
}

Prediction LibtorchMpRuntime::predict(const torch::Tensor& pos,
                                      const torch::Tensor& atomic_numbers,
                                      const torch::Tensor& cell,
                                      const torch::Tensor& pbc, int64_t charge,
                                      int64_t spin) {
  if (!impl_ || impl_->workers.empty()) {
    throw std::runtime_error("LibtorchMpRuntime: workers not started");
  }
  if (!pos.defined() || pos.dim() != 2 || pos.size(1) != 3) {
    throw std::runtime_error("LibtorchMpRuntime: pos must be [N,3]");
  }
  const int64_t n = pos.size(0);
  torch::Device build_dev(torch::kCUDA, 0);
  const auto dtype = compute_dtype_;

  pos0_ = pos.to(build_dev, dtype).contiguous();
  z0_ = atomic_numbers.to(build_dev, torch::kLong).contiguous();
  auto cell_in = cell.to(build_dev, dtype).contiguous();
  if (cell_in.dim() == 3) cell_in = cell_in.squeeze(0);
  cell0_ = cell_in;
  auto pbc_in = pbc.to(build_dev, torch::kBool).contiguous();
  if (pbc_in.dim() == 2) pbc_in = pbc_in.squeeze(0);
  pbc0_ = pbc_in;
  n_cached_ = n;

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();

  pos0_ = wrap_positions_to_cell(pos0_, cell0_, pbc0_);
  rebuild_neighbors_full(build_dev);
  const auto t_nl = clock::now();

  auto eidx_full = edge_index_.to(torch::kCPU).contiguous();
  auto coff_full = cell_offsets_.to(torch::kCPU, torch::kFloat64).contiguous();
  if (n != impl_->part_check_n || eidx_full.size(1) != impl_->part_check_e) {
    if (!graph_shard::partitions_cover_all_edges(eidx_full, n, num_devices_)) {
      throw std::runtime_error(
          "LibtorchMpRuntime: edge partitions do not cover graph");
    }
    impl_->part_check_n = n;
    impl_->part_check_e = eidx_full.size(1);
  }

  if (n > PayloadShm::kMaxN) {
    throw std::runtime_error("LibtorchMpRuntime: n exceeds PayloadShm::kMaxN");
  }
  if (num_devices_ > PayloadShm::kMaxWorld) {
    throw std::runtime_error("LibtorchMpRuntime: world exceeds PayloadShm::kMaxWorld");
  }
  if (!impl_->payload) {
    throw std::runtime_error("LibtorchMpRuntime: payload shm missing");
  }

  auto pos_cpu = pos0_.to(torch::kCPU, torch::kFloat64).contiguous();
  auto z_cpu = z0_.to(torch::kCPU, torch::kLong).contiguous();
  auto cell_cpu = cell0_.to(torch::kCPU, torch::kFloat64).contiguous();
  int32_t pbc32[3] = {pbc0_[0].item<bool>() ? 1 : 0, pbc0_[1].item<bool>() ? 1 : 0,
                      pbc0_[2].item<bool>() ? 1 : 0};

  // Publish geometry + shards once into payload shm (pipe only wakes workers).
  PayloadShm* pay = impl_->payload;
  pthread_mutex_lock(&pay->hdr->mu);
  pay->hdr->n = static_cast<int32_t>(n);
  pay->hdr->world = num_devices_;
  pay->hdr->charge = charge;
  pay->hdr->spin = spin;
  std::memcpy(pay->hdr->cell, cell_cpu.data_ptr<double>(), 9 * sizeof(double));
  std::memcpy(pay->hdr->pbc, pbc32, sizeof(pbc32));
  std::memcpy(pay->pos_ptr(), pos_cpu.data_ptr<double>(),
              static_cast<size_t>(n) * 3 * sizeof(double));
  std::memcpy(pay->z_ptr(), z_cpu.data_ptr<int64_t>(),
              static_cast<size_t>(n) * sizeof(int64_t));
  for (int r = 0; r < num_devices_; ++r) {
    auto shard = graph_shard::shard_edges(eidx_full, coff_full, n, num_devices_, r);
    const int64_t ne = shard.edge_index.size(1);
    if (ne > PayloadShm::kMaxE) {
      pthread_mutex_unlock(&pay->hdr->mu);
      throw std::runtime_error("LibtorchMpRuntime: nedges exceeds PayloadShm::kMaxE");
    }
    pay->hdr->nedges[r] = static_cast<int32_t>(ne);
    if (ne > 0) {
      auto eidx_cpu = shard.edge_index.contiguous();
      auto coff_cpu =
          shard.cell_offsets.defined()
              ? shard.cell_offsets.to(torch::kFloat64).contiguous()
              : torch::zeros({ne, 3}, torch::kFloat64);
      // Store as [2, E] flat (row-major): eidx[0,E) then eidx[E,2E)
      std::memcpy(pay->eidx_ptr(r), eidx_cpu.data_ptr<int64_t>(),
                  static_cast<size_t>(ne) * 2 * sizeof(int64_t));
      std::memcpy(pay->coff_ptr(r), coff_cpu.data_ptr<double>(),
                  static_cast<size_t>(ne) * 3 * sizeof(double));
    }
  }
  pay->hdr->result_gen = -1;
  const int32_t gen = ++pay->hdr->gen;
  pthread_mutex_unlock(&pay->hdr->mu);
  const auto t_pub = clock::now();

  // Tiny wake: cmd=1 + gen (no geometry on pipe).
  const int32_t cmd = 1;
  for (int r = 0; r < num_devices_; ++r) {
    auto& w = impl_->workers[static_cast<size_t>(r)];
    write_all(w.to_child, &cmd, sizeof(cmd));
    write_all(w.to_child, &gen, sizeof(gen));
  }

  // Collect status from all ranks; energy+forces from payload (rank0).
  double energy = 0.0;
  for (int r = 0; r < num_devices_; ++r) {
    auto& w = impl_->workers[static_cast<size_t>(r)];
    uint8_t ok = 0;
    read_all(w.from_child, &ok, 1);
    if (!ok) {
      int32_t mlen = 0;
      read_all(w.from_child, &mlen, sizeof(mlen));
      std::string msg(static_cast<size_t>(mlen), '\0');
      if (mlen > 0) read_all(w.from_child, msg.data(), static_cast<size_t>(mlen));
      throw std::runtime_error("LibtorchMpRuntime worker " + std::to_string(r) +
                               ": " + msg);
    }
  }
  const auto t_wait = clock::now();

  pthread_mutex_lock(&pay->hdr->mu);
  if (pay->hdr->result_gen != gen) {
    pthread_mutex_unlock(&pay->hdr->mu);
    throw std::runtime_error("LibtorchMpRuntime: payload result_gen mismatch");
  }
  energy = pay->hdr->energy;
  auto forces = torch::from_blob(pay->forces_ptr(), {n, 3}, torch::kFloat64).clone();
  pthread_mutex_unlock(&pay->hdr->mu);

  const double ms_nl =
      std::chrono::duration<double, std::milli>(t_nl - t0).count();
  const double ms_pub =
      std::chrono::duration<double, std::milli>(t_pub - t_nl).count();
  const double ms_wait =
      std::chrono::duration<double, std::milli>(t_wait - t_pub).count();
  const double ms_tot =
      std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  std::cerr << "PERF_PARENT world=" << num_devices_ << " ms_nl=" << ms_nl
            << " ms_pub=" << ms_pub << " ms_wait_workers=" << ms_wait
            << " ms_total=" << ms_tot << " gen=" << gen << "\n";

  Prediction out;
  out.energy = energy;
  out.forces = forces;
  return out;
}

Prediction LibtorchMpRuntime::predict_host(int n, const double* pos_xyz,
                                           const int* atomic_numbers,
                                           const double* cell_3x3,
                                           const int* pbc_3,
                                           double* forces_out_optional) {
  auto pos = torch::from_blob(const_cast<double*>(pos_xyz), {n, 3}, torch::kFloat64)
                 .clone();
  auto z = torch::empty({n}, torch::kLong);
  auto z_acc = z.accessor<int64_t, 1>();
  for (int i = 0; i < n; ++i) z_acc[i] = atomic_numbers[i];
  auto cell = torch::empty({3, 3}, torch::kFloat64);
  auto cell_acc = cell.accessor<double, 2>();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) cell_acc[i][j] = cell_3x3[3 * i + j];
  auto pbc = torch::tensor({pbc_3[0] != 0, pbc_3[1] != 0, pbc_3[2] != 0}, torch::kBool);
  auto pred = predict(pos, z, cell, pbc, 0, 0);
  if (forces_out_optional) {
    auto f_cpu = pred.forces.to(torch::kCPU).contiguous();
    std::memcpy(forces_out_optional, f_cpu.data_ptr<double>(),
                sizeof(double) * static_cast<size_t>(n) * 3);
  }
  return pred;
}

}  // namespace uma
