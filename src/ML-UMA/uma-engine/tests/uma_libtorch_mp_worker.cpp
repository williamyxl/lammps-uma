// Persistent process-per-rank LibTorch MP worker (exec'd; CUDA-safe).
// argv: uma_libtorch_mp_worker <rank> <world> <artifact_dir> <shm_path> <shm_bytes>
// stdin/stdout = parent pipes

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
    torch::Device dev(torch::kCUDA, 0);
    const std::string path = artifact_dir + "/model_mp_w" + std::to_string(world) +
                             "_r" + std::to_string(rank) + ".pt";
    auto module = torch::jit::load(path, dev);
    module.eval();
    module.to(dev);

    auto metadata = uma::load_artifact_metadata(artifact_dir + "/metadata.json");

    std::cerr << "uma_libtorch_mp_worker rank=" << rank << " module loaded, sending ready\n"
              << std::flush;
    uint8_t ready = 1;
    write_all(STDOUT_FILENO, &ready, 1);

    while (true) {
      int32_t cmd = 0;
      read_all(STDIN_FILENO, &cmd, sizeof(cmd));
      if (cmd == 0) {
        std::cerr << "uma_libtorch_mp_worker rank=" << rank << " shutdown\n" << std::flush;
        break;
      }
      int32_t n = 0;
      read_all(STDIN_FILENO, &n, sizeof(n));
      std::cerr << "uma_libtorch_mp_worker rank=" << rank << " predict n=" << n << "\n"
                << std::flush;
      std::vector<double> pos(static_cast<size_t>(n) * 3);
      std::vector<int64_t> z(static_cast<size_t>(n));
      double cell[9];
      int32_t pbc[3];
      int64_t charge = 0, spin = 0;
      int32_t nedges = 0;
      read_all(STDIN_FILENO, pos.data(), pos.size() * sizeof(double));
      read_all(STDIN_FILENO, z.data(), z.size() * sizeof(int64_t));
      read_all(STDIN_FILENO, cell, sizeof(cell));
      read_all(STDIN_FILENO, pbc, sizeof(pbc));
      read_all(STDIN_FILENO, &charge, sizeof(charge));
      read_all(STDIN_FILENO, &spin, sizeof(spin));
      read_all(STDIN_FILENO, &nedges, sizeof(nedges));
      std::vector<int64_t> eidx(static_cast<size_t>(nedges) * 2);
      std::vector<double> coff(static_cast<size_t>(nedges) * 3);
      if (nedges > 0) {
        read_all(STDIN_FILENO, eidx.data(), eidx.size() * sizeof(int64_t));
        read_all(STDIN_FILENO, coff.data(), coff.size() * sizeof(double));
      }

      auto pos_t = torch::from_blob(pos.data(), {n, 3}, torch::kFloat64).clone().to(dev);
      pos_t = pos_t.set_requires_grad(true);
      auto z_t = torch::from_blob(z.data(), {n}, torch::kLong).clone().to(dev);
      auto cell_t = torch::from_blob(cell, {3, 3}, torch::kFloat64).clone().to(dev);
      auto pbc_t = torch::tensor({pbc[0] != 0, pbc[1] != 0, pbc[2] != 0},
                                 torch::TensorOptions().dtype(torch::kBool).device(dev));
      auto eidx_t =
          torch::from_blob(eidx.data(), {2, nedges}, torch::kLong).clone().to(dev);
      auto coff_t =
          torch::from_blob(coff.data(), {nedges, 3}, torch::kFloat64).clone().to(dev);
      auto ch = torch::tensor(charge, torch::TensorOptions().dtype(torch::kLong).device(dev));
      auto sp = torch::tensor(spin, torch::TensorOptions().dtype(torch::kLong).device(dev));

      std::cerr << "uma_libtorch_mp_worker rank=" << rank
                << " edges=" << nedges << " starting module.forward\n"
                << std::flush;
      std::vector<torch::jit::IValue> args = {pos_t, z_t, cell_t, pbc_t, eidx_t, coff_t, ch, sp};
      torch::Tensor normed;
      {
        torch::autograd::AutoGradMode guard(true);
        normed = module.forward(args).toTensor().to(torch::kFloat64);
      }
      std::cerr << "uma_libtorch_mp_worker rank=" << rank << " forward done"
                << " normed.shape=" << normed.sizes()
                << " requires_grad=" << normed.requires_grad()
                << " grad_fn=" << (normed.grad_fn() ? "yes" : "no")
                << " normed0=" << normed.reshape({-1})[0].item<double>() << "\n"
                << std::flush;
      auto energy =
          uma::denorm_energy(normed, metadata.normalizer_mean, metadata.normalizer_rmsd);
      std::cerr << "uma_libtorch_mp_worker rank=" << rank
                << " after_denorm=" << energy.reshape({-1})[0].item<double>()
                << " mean=" << metadata.normalizer_mean
                << " rmsd=" << metadata.normalizer_rmsd << "\n"
                << std::flush;
      if (metadata.element_references.defined()) {
        auto refs = metadata.element_references.to(dev, torch::kFloat64);
        auto batch =
            torch::zeros({n}, torch::TensorOptions().dtype(torch::kLong).device(dev));
        energy = uma::undo_element_references(energy, z_t, batch, refs);
        std::cerr << "uma_libtorch_mp_worker rank=" << rank
                  << " after_refs=" << energy.reshape({-1})[0].item<double>() << "\n"
                  << std::flush;
      }
      // TS graph already contains uma_peer::all_reduce_sum on the energy logit.
      // Do NOT all_reduce again (that would 2x a full-system energy). Keep the
      // tensor on the Autograd graph for forces.
      auto e_for_grad = energy.reshape({-1}).sum();
      // Sync ranks before backward (collectives inside gather bwd).
      uma::PeerContext::instance().slot().barrier(rank);
      std::cerr << "uma_libtorch_mp_worker rank=" << rank
                << " e_for_grad=" << e_for_grad.item<double>()
                << " requires_grad=" << e_for_grad.requires_grad()
                << " grad_fn=" << (e_for_grad.grad_fn() ? "yes" : "no")
                << " starting autograd::grad\n"
                << std::flush;
      auto grads = torch::autograd::grad({e_for_grad}, {pos_t}, {}, false, false, false);
      auto forces = (-grads[0]).to(torch::kFloat64).contiguous();
      // Gather sum_grad bwd already all_reduces embedding grads; do not
      // all_reduce forces here (would 2x). Barrier so both ranks finish bwd.
      uma::PeerContext::instance().slot().barrier(rank);

      double e = e_for_grad.item<double>();
      auto f_cpu = forces.to(torch::kCPU).contiguous();
      uint8_t ok = 1;
      write_all(STDOUT_FILENO, &ok, 1);
      write_all(STDOUT_FILENO, &e, sizeof(e));
      write_all(STDOUT_FILENO, f_cpu.data_ptr<double>(),
                static_cast<size_t>(n) * 3 * sizeof(double));
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
