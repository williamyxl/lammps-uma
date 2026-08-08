#include "uma/kokkos_peer.h"

#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  using namespace uma;

  if (!torch::cuda::is_available() || torch::cuda::device_count() < 2) {
    std::cout << "kokkos_peer_device_smoke SKIP (need >=2 CUDA devices)\n";
    return 0;
  }

  kokkos_peer::enable_peer_access(2);
  kokkos_peer::fence_all(2);

  auto a = torch::arange(4, torch::TensorOptions().dtype(torch::kFloat64).device(torch::Device(torch::kCUDA, 0)))
               .unsqueeze(1);
  auto b = torch::arange(4, 8, torch::TensorOptions().dtype(torch::kFloat64).device(torch::Device(torch::kCUDA, 1)))
               .unsqueeze(1);

  auto full = kokkos_peer::all_gather_nodes({a, b}, /*n_atoms=*/8, torch::Device(torch::kCUDA, 0));
  auto expect = torch::arange(8, torch::TensorOptions().dtype(torch::kFloat64).device(torch::Device(torch::kCUDA, 0)))
                    .unsqueeze(1);
  if (!torch::allclose(full, expect)) {
    std::cerr << "peer all_gather_nodes mismatch\n" << full << "\n";
    return 1;
  }

  auto copied = kokkos_peer::peer_copy(a, torch::Device(torch::kCUDA, 1));
  if (!copied.is_cuda() || copied.get_device() != 1) {
    std::cerr << "peer_copy device mismatch\n";
    return 1;
  }
  if (!torch::allclose(copied.cpu(), a.cpu())) {
    std::cerr << "peer_copy value mismatch\n";
    return 1;
  }

  kokkos_peer::PeerGatherSlot slot(2);
  std::vector<torch::Tensor> got(2);
  std::thread t0([&] {
    got[0] = slot.all_gather_concat(0, a, 8);
  });
  std::thread t1([&] {
    got[1] = slot.all_gather_concat(1, b, 8);
  });
  t0.join();
  t1.join();
  if (!torch::allclose(got[0].cpu(), expect.cpu()) ||
      !torch::allclose(got[1].cpu(), expect.cpu())) {
    std::cerr << "PeerGatherSlot mismatch\n";
    return 1;
  }

  std::cout << "kokkos_peer_device_smoke OK\n";
  return 0;
}
