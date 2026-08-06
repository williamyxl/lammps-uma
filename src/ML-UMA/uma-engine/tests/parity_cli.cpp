// Minimal CLI: load artifact, predict structure (FP32 or FP64 host positions).
#include "uma/predictor.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

static bool load_structure(const std::string& path, std::vector<double>& pos,
                           std::vector<int>& z, std::vector<double>& cell) {
  // Format: N, then N lines "Z x y z", then 9 cell values (row-major).
  std::ifstream in(path);
  if (!in) return false;
  int n = 0;
  in >> n;
  pos.resize(static_cast<size_t>(n) * 3);
  z.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    in >> z[i] >> pos[3 * i] >> pos[3 * i + 1] >> pos[3 * i + 2];
  }
  cell.resize(9);
  for (int i = 0; i < 9; ++i) in >> cell[i];
  return static_cast<bool>(in) || in.eof();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <artifact_dir> [structure.txt]\n";
    return 1;
  }
  const std::string artifact = argv[1];
  torch::Device device =
      torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  auto pred = uma::Predictor::from_artifact(artifact, device);
  const bool f64 = pred.compute_dtype() == torch::kFloat64;
  std::printf("compute_dtype=%s device=%s\n", f64 ? "float64" : "float32",
              device.is_cuda() ? "cuda" : "cpu");

  std::vector<double> pos;
  std::vector<int> z;
  std::vector<double> cell;
  if (argc >= 3) {
    if (!load_structure(argv[2], pos, z, cell)) {
      std::cerr << "Failed to read structure " << argv[2] << "\n";
      return 1;
    }
  } else {
    const double a = 5.64;
    const int nrep = 2;
    const double L = a * nrep;
    cell = {L, 0, 0, 0, L, 0, 0, 0, L};
    for (int ix = 0; ix < nrep; ++ix)
      for (int iy = 0; iy < nrep; ++iy)
        for (int iz = 0; iz < nrep; ++iz) {
          auto add = [&](int Z, double ox, double oy, double oz) {
            z.push_back(Z);
            pos.push_back((ix + ox) * a);
            pos.push_back((iy + oy) * a);
            pos.push_back((iz + oz) * a);
          };
          add(11, 0.0, 0.0, 0.0);
          add(17, 0.5, 0.0, 0.0);
          add(17, 0.0, 0.5, 0.0);
          add(11, 0.5, 0.5, 0.0);
          add(17, 0.0, 0.0, 0.5);
          add(11, 0.5, 0.0, 0.5);
          add(11, 0.0, 0.5, 0.5);
          add(17, 0.5, 0.5, 0.5);
        }
  }

  const int n = static_cast<int>(z.size());
  std::vector<double> forces(static_cast<size_t>(n) * 3);
  int pbc[3] = {1, 1, 1};
  auto out = pred.predict_host(n, pos.data(), z.data(), cell.data(), pbc, forces.data());
  double fmax = 0.0;
  for (double f : forces) fmax = std::max(fmax, std::abs(f));
  std::printf("n=%d energy=%.12f eV  fmax=%.10e\n", n, out.energy, fmax);
  return 0;
}
