// Persistent process-per-rank LibTorch MP worker (exec'd; CUDA-safe).
// argv: uma_libtorch_mp_worker <rank> <world> <artifact_dir> <shm_path> <shm_bytes>
// stdin/stdout = parent pipes

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(UMA_ENGINE_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <c10/core/AutogradState.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>

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
        payload->destroy();
        break;
      }
      // cmd=1: predict via payload shm. Pipe carries only gen (wake).
      int32_t gen = 0;
      read_all(STDIN_FILENO, &gen, sizeof(gen));

      // Parent does not rewrite payload until all ranks return ok — safe to read
      // after gen matches without holding the mutex through H2D.
      pthread_mutex_lock(&payload->hdr->mu);
      if (payload->hdr->gen != gen) {
        pthread_mutex_unlock(&payload->hdr->mu);
        throw std::runtime_error("worker: payload gen mismatch");
      }
      const int32_t n = payload->hdr->n;
      const int32_t nedges = payload->hdr->nedges[rank];
      const int64_t charge = payload->hdr->charge;
      const int64_t spin = payload->hdr->spin;
      double cell[9];
      int32_t pbc[3];
      std::memcpy(cell, payload->hdr->cell, sizeof(cell));
      std::memcpy(pbc, payload->hdr->pbc, sizeof(pbc));
      pthread_mutex_unlock(&payload->hdr->mu);

      auto pos_t =
          torch::from_blob(payload->pos_ptr(), {n, 3}, torch::kFloat64).to(dev).contiguous();
      auto z_t = torch::from_blob(payload->z_ptr(), {n}, torch::kLong).to(dev).contiguous();
      auto cell_t = torch::from_blob(cell, {3, 3}, torch::kFloat64).to(dev).contiguous();
      auto eidx_t =
          (nedges > 0)
              ? torch::from_blob(payload->eidx_ptr(rank), {2, nedges}, torch::kLong)
                    .to(dev)
                    .contiguous()
              : torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kLong).device(dev));
      auto coff_t =
          (nedges > 0)
              ? torch::from_blob(payload->coff_ptr(rank), {nedges, 3}, torch::kFloat64)
                    .to(dev)
                    .contiguous()
              : torch::empty({0, 3},
                             torch::TensorOptions().dtype(torch::kFloat64).device(dev));

      pos_t = pos_t.set_requires_grad(true);
      auto pbc_t = torch::tensor({pbc[0] != 0, pbc[1] != 0, pbc[2] != 0},
                                 torch::TensorOptions().dtype(torch::kBool).device(dev));
      auto ch = torch::tensor(charge, torch::TensorOptions().dtype(torch::kLong).device(dev));
      auto sp = torch::tensor(spin, torch::TensorOptions().dtype(torch::kLong).device(dev));

      using clock = std::chrono::steady_clock;
      const auto t_fwd0 = clock::now();
      if (verbose) {
        std::cerr << "uma_libtorch_mp_worker rank=" << rank << " edges=" << nedges
                  << " starting module.forward\n"
                  << std::flush;
      }
      std::vector<torch::jit::IValue> args = {pos_t, z_t, cell_t, pbc_t, eidx_t, coff_t, ch, sp};
      torch::Tensor normed;
      {
        torch::autograd::AutoGradMode guard(true);
        normed = module.forward(args).toTensor().to(torch::kFloat64);
      }
      // No explicit sync after forward — peer barrier below drains the device.
      const double ms_fwd =
          std::chrono::duration<double, std::milli>(clock::now() - t_fwd0).count();
      auto energy =
          uma::denorm_energy(normed, metadata.normalizer_mean, metadata.normalizer_rmsd);
      if (metadata.element_references.defined()) {
        auto refs = metadata.element_references.to(dev, torch::kFloat64);
        auto batch =
            torch::zeros({n}, torch::TensorOptions().dtype(torch::kLong).device(dev));
        energy = uma::undo_element_references(energy, z_t, batch, refs);
      }
      // TS graph already contains uma_peer::all_reduce_sum on the energy logit.
      double escale = 1.0 / static_cast<double>(world);
      if (const char* es = std::getenv("UMA_GRAD_ENERGY_SCALE")) {
        escale = std::strtod(es, nullptr);
        if (!(escale > 0.0)) escale = 1.0 / static_cast<double>(world);
      }
      auto e_for_grad = energy.reshape({-1}).sum() * escale;
      uma::PeerContext::instance().slot().barrier(rank);
      const auto t_bwd0 = clock::now();
      auto grads = torch::autograd::grad({e_for_grad}, {pos_t}, {}, false, false, false);
      auto forces = (-grads[0]).to(torch::kFloat64).contiguous();
      const double ms_bwd =
          std::chrono::duration<double, std::milli>(clock::now() - t_bwd0).count();
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
#if defined(UMA_ENGINE_USE_CUDA)
      // One sync before D2H / result publish.
      cudaDeviceSynchronize();
#endif
      if (rank == 0) {
        std::cerr << "PERF_TICK rank=0 world=" << world << " ms_fwd=" << ms_fwd
                  << " ms_bwd=" << ms_bwd << " ms_force_ar=" << ms_fred
                  << " ms_compute≈" << (ms_fwd + ms_bwd + ms_fred) << "\n"
                  << std::flush;
      }

      // Rank0 publishes E+F into payload shm; pipe returns ok only (pipe-tax cut).
      const double e = energy.reshape({-1})[0].item<double>();
      if (rank == 0) {
        auto f_cpu = forces.to(torch::kCPU).contiguous();
        pthread_mutex_lock(&payload->hdr->mu);
        payload->hdr->energy = e;
        std::memcpy(payload->forces_ptr(), f_cpu.data_ptr<double>(),
                    static_cast<size_t>(n) * 3 * sizeof(double));
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
