// Persistent process-per-rank LibTorch MP worker (exec'd; CUDA-safe).
// argv: uma_libtorch_mp_worker <rank> <world> <artifact_dir> <shm_path> <shm_bytes>
// stdin/stdout = parent pipes

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(UMA_ENGINE_USE_CUDA)
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

#include <c10/core/AutogradState.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "uma/graph_shard.h"
#include "uma/kokkos_peer.h"
#include "uma/metadata.h"
#include "uma/payload_shm.h"
#include "uma/peer_context.h"
#include "uma/postprocess.h"
#include "uma/shared_peer.h"

namespace {

void write_all(int fd, const void* src, size_t n) {
  const char* p = static_cast<const char*>(src);
  size_t off = 0;
  while (off < n) {
    const ssize_t w = ::write(fd, p + off, n - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("write: ") + std::strerror(errno));
    }
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
      throw std::runtime_error(std::string("read: ") + std::strerror(errno));
    }
    if (r == 0) throw std::runtime_error("read: EOF");
    off += static_cast<size_t>(r);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: uma_libtorch_mp_worker rank world artifact_dir shm_path shm_bytes\n";
    return 2;
  }
  const int rank = std::stoi(argv[1]);
  const int world = std::stoi(argv[2]);
  const std::string artifact_dir = argv[3];
  const std::string shm_path = argv[4];
  const size_t shm_bytes = static_cast<size_t>(std::stoull(argv[5]));
  const bool verbose = [] {
    const char* e = std::getenv("UMA_MP_VERBOSE");
    return e && e[0] == '1' && e[1] == '\0';
  }();

  // Banner to per-rank stderr log (parent may have already dup2'd STDERR).
  std::cerr << "uma_libtorch_mp_worker START rank=" << rank << " world=" << world
            << " artifact=" << artifact_dir << " shm=" << shm_path
            << " CUDA_VISIBLE_DEVICES="
            << (std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES")
                                                    : "(unset)")
            << " CUDA_LAUNCH_BLOCKING="
            << (std::getenv("CUDA_LAUNCH_BLOCKING") ? std::getenv("CUDA_LAUNCH_BLOCKING")
                                                    : "(unset)")
            << std::endl
            << std::flush;

  try {
    torch::jit::setTensorExprFuserEnabled(false);
    torch::jit::setGraphExecutorOptimize(false);
    // Mid-backward uma_peer collectives must run in deterministic reverse-forward
    // order on both ranks. Autograd's thread pool would desync the shm barrier.
    c10::AutogradState::get_tls_state().set_multithreading_enabled(false);
    uma::register_uma_peer_ops();

    int fd = ::open(shm_path.c_str(), O_RDWR);
    if (fd < 0) {
      throw std::runtime_error(std::string("open shm: ") + std::strerror(errno));
    }
    void* mem = mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) {
      throw std::runtime_error(std::string("mmap shm: ") + std::strerror(errno));
    }
    auto* shm = static_cast<uma::kokkos_peer::SharedPeerGatherSlot::Shm*>(mem);
    auto* slot = uma::kokkos_peer::SharedPeerGatherSlot::attach(shm, shm_bytes);
    uma::PeerContext::instance().reset_shared(slot);
    uma::PeerContext::set_thread_rank(rank);

#if defined(UMA_ENGINE_USE_CUDA)
    // Parent sets CUDA_VISIBLE_DEVICES=<rank> before exec; only cuda:0 is visible.
    cudaSetDevice(0);
#endif
    // Peer device transport: cuda_ipc (default) or nccl (opt-in); no-op for shm.
    slot->init_cuda_ipc(rank);
    slot->init_nccl(rank);

    const char* pay_path = std::getenv("UMA_MP_PAYLOAD_SHM");
    const char* pay_bytes_e = std::getenv("UMA_MP_PAYLOAD_BYTES");
    if (!pay_path || !*pay_path || !pay_bytes_e) {
      throw std::runtime_error("worker: UMA_MP_PAYLOAD_SHM/BYTES required");
    }
    auto* payload =
        uma::PayloadShm::attach_file(pay_path, static_cast<size_t>(std::stoull(pay_bytes_e)));
    bool payload_pinned = false;
#if defined(UMA_ENGINE_USE_CUDA)
    // Match parent pin so H2D from shm is not pageable (V3).
    const cudaError_t reg = cudaHostRegister(
        payload->hdr, payload->bytes, cudaHostRegisterPortable);
    if (reg != cudaSuccess) {
      std::cerr << "worker: cudaHostRegister failed: " << cudaGetErrorString(reg)
                << "\n"
                << std::flush;
    } else {
      payload_pinned = true;
    }
#endif
    std::cerr << "uma_libtorch_mp_worker rank=" << rank
              << " payload_shm=" << pay_path << "\n"
              << std::flush;

    torch::Device dev(torch::kCUDA, 0);
    // Prefer n-specific export (partition offsets baked at trace time).
    std::string path;
    if (const char* natoms_e = std::getenv("UMA_MP_NATOMS")) {
      const std::string nspec = artifact_dir + "/model_mp_w" + std::to_string(world) +
                                "_n" + natoms_e + "_r" + std::to_string(rank) + ".pt";
      if (::access(nspec.c_str(), R_OK) == 0) path = nspec;
    }
    if (path.empty()) {
      path = artifact_dir + "/model_mp_w" + std::to_string(world) + "_r" +
             std::to_string(rank) + ".pt";
    }
    std::cerr << "uma_libtorch_mp_worker rank=" << rank << " loading " << path << "\n"
              << std::flush;
    auto module = torch::jit::load(path, dev);
    module.eval();
    module.to(dev);

    auto metadata = uma::load_artifact_metadata(artifact_dir + "/metadata.json");

    const bool verbose = [] {
      const char* e = std::getenv("UMA_MP_VERBOSE");
      return e && e[0] == '1' && e[1] == '\0';
    }();
    // W11: CUDA Graph of fwd+bwd (requires UMA_EDGE_PAD=1 for fixed shapes).
    // NCCL-inside-TS makes this stretch; capture failure → eager forever.
    const bool want_cuda_graph = [] {
      const char* e = std::getenv("UMA_CUDA_GRAPH");
      const char* p = std::getenv("UMA_EDGE_PAD");
      return e && std::string(e) == "1" && p && std::string(p) == "1";
    }();
    const int graph_warmup = [] {
      const char* e = std::getenv("UMA_CUDA_GRAPH_WARMUP");
      if (!e || !*e) return 3;
      const int v = std::atoi(e);
      return v > 0 ? v : 3;
    }();
#if defined(UMA_ENGINE_USE_CUDA)
    at::cuda::CUDAGraph cuda_graph;
    // Capture/replay require a non-default stream (PyTorch CUDAGraph rule).
    at::cuda::CUDAStream graph_stream =
        want_cuda_graph ? at::cuda::getStreamFromPool(/*isHighPriority=*/true)
                        : at::cuda::getDefaultCUDAStream();
    bool graph_captured = false;
    bool graph_failed = false;
    int graph_step = 0;
    int64_t graph_e_cap = -1;
    torch::Tensor g_pos, g_z, g_cell, g_eidx, g_coff, g_pbc, g_ch, g_sp;
    torch::Tensor g_normed, g_forces;
#endif
    if (want_cuda_graph) {
      std::cerr << "uma_libtorch_mp_worker rank=" << rank
                << " W11 CUDA_GRAPH enabled (warmup=" << graph_warmup
                << " stream=pool_nondefault)\n"
                << std::flush;
    }
    std::cerr << "uma_libtorch_mp_worker rank=" << rank << " module loaded, sending ready\n"
              << std::flush;
    uint8_t ready = 1;
    write_all(STDOUT_FILENO, &ready, 1);

    while (true) {
      int32_t cmd = 0;
      read_all(STDIN_FILENO, &cmd, sizeof(cmd));
      if (cmd == 0) {
        std::cerr << "uma_libtorch_mp_worker rank=" << rank << " shutdown\n" << std::flush;
        slot->release_nccl();
        slot->release_cuda_ipc();
#if defined(UMA_ENGINE_USE_CUDA)
        if (payload_pinned) {
          cudaHostUnregister(payload->hdr);
          payload_pinned = false;
        }
#endif
        payload->destroy();
        break;
      }
      // cmd=1: predict via payload shm. Pipe carries only gen (wake).
      int32_t gen = 0;
      read_all(STDIN_FILENO, &gen, sizeof(gen));

      // W18: instrument the block between the wake and t_fwd0. W13 showed
      // ~33.7 ms/step (35% of the step) inside ms_wait_workers that no counter
      // covers; ms_fwd/ms_bwd only bracket the model call. Optimizing an
      // unmeasured region is guesswork, so these land before any change.
      using w18_clock = std::chrono::steady_clock;
      const auto t_w18_h2d0 = w18_clock::now();
      double ms_h2d = 0.0, ms_wshard = 0.0, ms_pad = 0.0, ms_prep = 0.0;
      double ms_bar_pre_bwd = 0.0, ms_post = 0.0;

      // Parent does not rewrite payload until all ranks return ok — safe to read
      // after gen matches without holding the mutex through H2D.
      pthread_mutex_lock(&payload->hdr->mu);
      if (payload->hdr->gen != gen) {
        pthread_mutex_unlock(&payload->hdr->mu);
        throw std::runtime_error("worker: payload gen mismatch");
      }
      const int32_t n = payload->hdr->n;
      const int32_t n_edges_full = payload->hdr->n_edges_full;
      const int64_t charge = payload->hdr->charge;
      const int64_t spin = payload->hdr->spin;
      double cell[9];
      int32_t pbc[3];
      std::memcpy(cell, payload->hdr->cell, sizeof(cell));
      std::memcpy(pbc, payload->hdr->pbc, sizeof(pbc));
      pthread_mutex_unlock(&payload->hdr->mu);

      // V3: non-blocking H2D from pinned payload, one sync before forward.
      auto pos_t =
          torch::from_blob(payload->pos_ptr(), {n, 3}, torch::kFloat64)
              .to(dev, /*non_blocking=*/true)
              .contiguous();
      auto z_t = torch::from_blob(payload->z_ptr(), {n}, torch::kLong)
                     .to(dev, /*non_blocking=*/true)
                     .contiguous();
      auto cell_t = torch::from_blob(cell, {3, 3}, torch::kFloat64)
                        .to(dev, /*non_blocking=*/true)
                        .contiguous();
      // W6: H2D full graph, then FairChem partition on GPU (center ∈ node_partition).
      torch::Tensor eidx_t;
      torch::Tensor coff_t;
      int32_t nedges = 0;
      if (n_edges_full > 0) {
        auto eidx_full =
            torch::from_blob(payload->eidx_full_ptr(), {2, n_edges_full}, torch::kLong)
                .to(dev, /*non_blocking=*/true)
                .contiguous();
        auto coff_full =
            torch::from_blob(payload->coff_full_ptr(), {n_edges_full, 3}, torch::kInt32)
                .to(dev, /*non_blocking=*/true)
                .contiguous();
        ms_h2d = std::chrono::duration<double, std::milli>(
                     w18_clock::now() - t_w18_h2d0).count();
        const auto t_w18_shard0 = w18_clock::now();
        auto shard = uma::graph_shard::shard_edges(eidx_full, coff_full, n, world, rank);
        eidx_t = shard.edge_index;
        // W5: int32→FP64 cast on device before TorchScript.
        coff_t = shard.cell_offsets.to(torch::kFloat64).contiguous();
        nedges = static_cast<int32_t>(eidx_t.size(1));
        ms_wshard = std::chrono::duration<double, std::milli>(
                       w18_clock::now() - t_w18_shard0).count();
      } else {
        eidx_t = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kLong).device(dev));
        coff_t = torch::empty({0, 3},
                              torch::TensorOptions().dtype(torch::kFloat64).device(dev));
      }
      // W10: optional fixed-shape edge pad (high-water; unlocks W11 CUDA graph).
      const auto t_w18_pad0 = w18_clock::now();
      {
        const char* pad_e = std::getenv("UMA_EDGE_PAD");
        if (pad_e && std::string(pad_e) == "1") {
          static thread_local int64_t edge_pad_cap = 0;
          int64_t e_now = eidx_t.size(1);
          if (const char* pe = std::getenv("UMA_EDGE_PAD_E")) {
            const int64_t forced = std::strtoll(pe, nullptr, 10);
            if (forced > edge_pad_cap) edge_pad_cap = forced;
          }
          if (e_now > edge_pad_cap) edge_pad_cap = e_now;
          if (edge_pad_cap < 1) edge_pad_cap = 1;
          auto nodes = uma::graph_shard::node_partition(n, world, rank).to(dev);
          const int64_t pad_atom =
              nodes.numel() > 0 ? nodes[0].item<int64_t>() : int64_t{0};
          uma::graph_shard::pad_edges_to_capacity(eidx_t, coff_t, edge_pad_cap,
                                                  pad_atom);
          nedges = static_cast<int32_t>(eidx_t.size(1));
        }
      }
      ms_pad = std::chrono::duration<double, std::milli>(
                   w18_clock::now() - t_w18_pad0).count();
      // Tier0/W4: same-stream H2D is ordered before forward; no host sync here.

      const auto t_w18_prep0 = w18_clock::now();
      pos_t = pos_t.set_requires_grad(true);
      auto pbc_t = torch::tensor({pbc[0] != 0, pbc[1] != 0, pbc[2] != 0},
                                 torch::TensorOptions().dtype(torch::kBool).device(dev));
      auto ch = torch::tensor(charge, torch::TensorOptions().dtype(torch::kLong).device(dev));
      auto sp = torch::tensor(spin, torch::TensorOptions().dtype(torch::kLong).device(dev));
      ms_prep = std::chrono::duration<double, std::milli>(
                    w18_clock::now() - t_w18_prep0).count();

      using clock = std::chrono::steady_clock;
      double escale = 1.0 / static_cast<double>(world);
      if (const char* es = std::getenv("UMA_GRAD_ENERGY_SCALE")) {
        escale = std::strtod(es, nullptr);
        if (!(escale > 0.0)) escale = 1.0 / static_cast<double>(world);
      }
      const bool skip_pre_bwd = [] {
        const char* skip_bar = std::getenv("UMA_SKIP_PRE_BWD_BARRIER");
        return skip_bar && std::string(skip_bar) == "1";
      }();

      // Shared eager body: fwd → denorm → (optional barrier) → bwd.
      // Used for non-graph path and graph warmup/capture.
      auto run_eager_fwd_bwd =
          [&](torch::Tensor& pos_use, torch::Tensor& z_use, torch::Tensor& cell_use,
              torch::Tensor& pbc_use, torch::Tensor& eidx_use, torch::Tensor& coff_use,
              torch::Tensor& ch_use, torch::Tensor& sp_use, bool skip_barrier)
              -> std::tuple<torch::Tensor, torch::Tensor, double, double> {
        const auto t_fwd0 = clock::now();
        if (verbose) {
          std::cerr << "uma_libtorch_mp_worker rank=" << rank << " edges=" << nedges
                    << " starting module.forward\n"
                    << std::flush;
        }
        std::vector<torch::jit::IValue> args = {pos_use, z_use, cell_use, pbc_use,
                                                eidx_use, coff_use, ch_use, sp_use};
        torch::Tensor normed;
        {
          torch::autograd::AutoGradMode guard(true);
          normed = module.forward(args).toTensor().to(torch::kFloat64);
        }
        const double ms_fwd_loc =
            std::chrono::duration<double, std::milli>(clock::now() - t_fwd0).count();
        auto energy_loc =
            uma::denorm_energy(normed, metadata.normalizer_mean, metadata.normalizer_rmsd);
        if (metadata.element_references.defined()) {
          auto refs = metadata.element_references.to(dev, torch::kFloat64);
          auto batch =
              torch::zeros({n}, torch::TensorOptions().dtype(torch::kLong).device(dev));
          energy_loc = uma::undo_element_references(energy_loc, z_use, batch, refs);
        }
        auto e_for_grad = energy_loc.reshape({-1}).sum() * escale;
        // W18: time the barrier separately. It sits between the fwd and bwd
        // timers, so its cost was previously invisible in both -- a prime
        // suspect for the 33.7 ms unaccounted block. W12 showed removing it
        // does NOT help (the wait reappears in the backward NCCL collective),
        // so measure it rather than delete it.
        if (!skip_barrier) {
          const auto t_bar0 = w18_clock::now();
          uma::PeerContext::instance().slot().barrier(rank);
          ms_bar_pre_bwd = std::chrono::duration<double, std::milli>(
                               w18_clock::now() - t_bar0).count();
        }
        const auto t_bwd0 = clock::now();
        auto grads =
            torch::autograd::grad({e_for_grad}, {pos_use}, {}, false, false, false);
        auto forces_loc = (-grads[0]).to(torch::kFloat64).contiguous();
        const double ms_bwd_loc =
            std::chrono::duration<double, std::milli>(clock::now() - t_bwd0).count();
        return {energy_loc, forces_loc, ms_fwd_loc, ms_bwd_loc};
      };

      torch::Tensor energy;
      torch::Tensor forces;
      double ms_fwd = 0.0;
      double ms_bwd = 0.0;
      const char* path_tag = "eager";

#if defined(UMA_ENGINE_USE_CUDA)
      const bool try_graph =
          want_cuda_graph && !graph_failed && eidx_t.size(1) > 0;
      if (try_graph) {
        const int64_t e_cap = eidx_t.size(1);
        // Rebuild static buffers if N / E capacity changed (invalidates capture).
        if (!g_pos.defined() || g_pos.size(0) != n || graph_e_cap != e_cap) {
          if (graph_captured) {
            cuda_graph.reset();
            graph_captured = false;
            graph_step = 0;
            std::cerr << "uma_libtorch_mp_worker rank=" << rank
                      << " W11 graph reset (shape change e_cap=" << e_cap << ")\n"
                      << std::flush;
          }
          {
            torch::NoGradGuard ng;
            g_pos = torch::empty(
                {n, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            g_z = torch::empty({n},
                              torch::TensorOptions().dtype(torch::kLong).device(dev));
            g_cell = torch::empty(
                {3, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            g_eidx = torch::empty(
                {2, e_cap}, torch::TensorOptions().dtype(torch::kLong).device(dev));
            g_coff = torch::empty(
                {e_cap, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            g_pbc = torch::empty({3},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
            g_ch = torch::empty({},
                               torch::TensorOptions().dtype(torch::kLong).device(dev));
            g_sp = torch::empty({},
                               torch::TensorOptions().dtype(torch::kLong).device(dev));
          }
          g_pos.set_requires_grad(true);
          graph_e_cap = e_cap;
        }
        {
          torch::NoGradGuard ng;
          g_pos.copy_(pos_t);
          g_z.copy_(z_t);
          g_cell.copy_(cell_t);
          g_eidx.copy_(eidx_t);
          g_coff.copy_(coff_t);
          g_pbc.copy_(pbc_t);
          g_ch.copy_(ch);
          g_sp.copy_(sp);
        }

        if (!graph_captured) {
          // Warmup then capture (skip host barrier — NCCL in TS orders ranks).
          if (graph_step < graph_warmup) {
            std::tie(energy, forces, ms_fwd, ms_bwd) = run_eager_fwd_bwd(
                g_pos, g_z, g_cell, g_pbc, g_eidx, g_coff, g_ch, g_sp, /*skip=*/true);
            ++graph_step;
            path_tag = "graph_warmup";
          } else {
            uma::PeerContext::instance().slot().barrier(rank);
            bool capture_begun = false;
            try {
              // Must not capture on the default stream.
              at::cuda::CUDAStreamGuard guard(graph_stream);
              cuda_graph.capture_begin(
                  /*pool=*/{0, 0}, cudaStreamCaptureModeThreadLocal);
              capture_begun = true;
              std::tie(energy, forces, ms_fwd, ms_bwd) = run_eager_fwd_bwd(
                  g_pos, g_z, g_cell, g_pbc, g_eidx, g_coff, g_ch, g_sp,
                  /*skip=*/true);
              g_normed = energy;  // keep live handles updated by replay
              g_forces = forces;
              cuda_graph.capture_end();
              capture_begun = false;
              graph_captured = true;
              path_tag = "graph_capture";
              std::cerr << "uma_libtorch_mp_worker rank=" << rank
                        << " W11 CUDA graph captured e_cap=" << e_cap << "\n"
                        << std::flush;
            } catch (const std::exception& ex) {
              // Mid-capture abort leaves the stream capturing; fully unwind
              // before eager (reset alone can leave sticky capturing state).
              graph_failed = true;
              graph_captured = false;
              if (capture_begun) {
                try {
                  cuda_graph.reset();
                } catch (...) {
                }
              }
              {
                cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
                if (cudaStreamIsCapturing(graph_stream.stream(), &cap_st) ==
                        cudaSuccess &&
                    cap_st != cudaStreamCaptureStatusNone) {
                  cudaGraph_t orphan = nullptr;
                  cudaStreamEndCapture(graph_stream.stream(), &orphan);
                  if (orphan) cudaGraphDestroy(orphan);
                }
                cudaGetLastError();  // clear sticky async error
                cudaStreamSynchronize(graph_stream.stream());
              }
              std::cerr << "uma_libtorch_mp_worker rank=" << rank
                        << " W11 CUDA graph CAPTURE FAILED: " << ex.what()
                        << " — falling back to eager (graph disabled)\n"
                        << std::flush;
              std::tie(energy, forces, ms_fwd, ms_bwd) = run_eager_fwd_bwd(
                  g_pos, g_z, g_cell, g_pbc, g_eidx, g_coff, g_ch, g_sp,
                  skip_pre_bwd);
              path_tag = "graph_fail_eager";
            }
            uma::PeerContext::instance().slot().barrier(rank);
          }
        } else {
          const auto t0 = clock::now();
          {
            at::cuda::CUDAStreamGuard guard(graph_stream);
            cuda_graph.replay();
          }
          energy = g_normed;
          forces = g_forces;
          ms_fwd =
              std::chrono::duration<double, std::milli>(clock::now() - t0).count();
          ms_bwd = 0.0;  // folded into replay
          path_tag = "graph_replay";
        }
      } else
#endif
      {
        std::tie(energy, forces, ms_fwd, ms_bwd) = run_eager_fwd_bwd(
            pos_t, z_t, cell_t, pbc_t, eidx_t, coff_t, ch, sp, skip_pre_bwd);
        path_tag = "eager";
      }

      const auto t_w18_post0 = w18_clock::now();
      const char* skip_fred = std::getenv("UMA_SKIP_FORCE_GP_REDUCE");
      double ms_fred = 0.0;
      if (skip_fred && std::string(skip_fred) == "1") {
        uma::PeerContext::instance().slot().barrier(rank);
      } else {
        const auto t_fred0 = clock::now();
        forces = uma::PeerContext::instance().slot().all_reduce(
            rank, forces.contiguous());
        ms_fred =
            std::chrono::duration<double, std::milli>(clock::now() - t_fred0).count();
      }
      // W14: scoped stream sync + async D2H into pinned payload (no full
      // cudaDeviceSynchronize; avoid torch CPU clone when shm is registered).
#if defined(UMA_ENGINE_USE_CUDA)
      auto stream = at::cuda::getCurrentCUDAStream();
      bool forces_d2h_async = false;
      if (rank == 0) {
        forces = forces.contiguous();
        const size_t nbytes =
            static_cast<size_t>(n) * 3u * sizeof(double);
        if (payload_pinned && forces.is_cuda() &&
            forces.scalar_type() == torch::kFloat64) {
          const cudaError_t st = cudaMemcpyAsync(
              payload->forces_ptr(), forces.data_ptr<double>(), nbytes,
              cudaMemcpyDeviceToHost, stream.stream());
          if (st != cudaSuccess) {
            throw std::runtime_error(
                std::string("W14 cudaMemcpyAsync forces D2H: ") +
                cudaGetErrorString(st));
          }
          forces_d2h_async = true;
        }
      }
      {
        const cudaError_t st = cudaStreamSynchronize(stream.stream());
        if (st != cudaSuccess) {
          throw std::runtime_error(
              std::string("W14 cudaStreamSynchronize: ") +
              cudaGetErrorString(st));
        }
      }
#else
      const bool forces_d2h_async = false;
#endif
      ms_post = std::chrono::duration<double, std::milli>(
                    w18_clock::now() - t_w18_post0).count();
      // W18: ms_accounted should track the parent's ms_wait_workers to within
      // a couple of ms. A residual gap means there is still hidden cost.
      const double ms_accounted = ms_h2d + ms_wshard + ms_pad + ms_prep +
                                  ms_fwd + ms_bar_pre_bwd + ms_bwd + ms_post;
      std::cerr << "PERF_TICK rank=" << rank << " world=" << world
                << " nedges=" << nedges << " path=" << path_tag
                << " ms_fwd=" << ms_fwd
                << " ms_bwd=" << ms_bwd << " ms_force_ar=" << ms_fred
                << " ms_h2d=" << ms_h2d
                << " ms_wshard=" << ms_wshard
                << " ms_pad=" << ms_pad
                << " ms_prep=" << ms_prep
                << " ms_bar_pre_bwd=" << ms_bar_pre_bwd
                << " ms_post=" << ms_post
                << " ms_accounted=" << ms_accounted
                << " ms_compute≈" << (ms_fwd + ms_bwd + ms_fred)
#if defined(UMA_ENGINE_USE_CUDA)
                << " d2h=" << (forces_d2h_async ? "async_pin" : "sync_cpu")
#endif
                << "\n"
                << std::flush;

      // Rank0 publishes E+F into payload shm; pipe returns ok only (pipe-tax cut).
      const double e = energy.reshape({-1})[0].item<double>();
      if (rank == 0) {
        pthread_mutex_lock(&payload->hdr->mu);
        payload->hdr->energy = e;
        if (!forces_d2h_async) {
          auto f_cpu = forces.to(torch::kCPU).contiguous();
          std::memcpy(payload->forces_ptr(), f_cpu.data_ptr<double>(),
                      static_cast<size_t>(n) * 3 * sizeof(double));
        }
        payload->hdr->result_gen = gen;
        pthread_mutex_unlock(&payload->hdr->mu);
      } else {
        // Ensure non-rank0 finished before parent reads (parent waits on all oks).
        (void)e;
      }
      uint8_t ok = 1;
      write_all(STDOUT_FILENO, &ok, 1);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "uma_libtorch_mp_worker rank=" << argv[1] << " ERROR: " << ex.what()
              << "\n";
    uint8_t ok = 0;
    try {
      write_all(STDOUT_FILENO, &ok, 1);
      std::string msg = ex.what();
      int32_t mlen = static_cast<int32_t>(msg.size());
      write_all(STDOUT_FILENO, &mlen, sizeof(mlen));
      write_all(STDOUT_FILENO, msg.data(), msg.size());
    } catch (...) {
    }
    return 1;
  }
}
