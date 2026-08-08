// Minimal CLI: load artifact, predict structure (FP32 or FP64 host positions).
// Optional --devices N selects traced (1) vs FairChem eager GP (>1).
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

static void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " <artifact_dir> [structure.txt] [--devices N] [--write-forces FILE]\n"
            << "  devices=1 (default): TorchScript traced Predictor\n"
            << "  devices>1: C++ LibTorch MP (model_mp_wN_r*.pt); "
            << "Python only if UMA_PYTHON_GP_WORKER=1\n"
            << "  Env: UMA_CHECKPOINT, UMA_PYTHON, UMA_GP_WORKER\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }

  std::string artifact;
  std::string structure;
  std::string forces_out;
  int num_devices = 1;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--devices") {
      if (i + 1 >= argc) {
        usage(argv[0]);
        return 1;
      }
      num_devices = std::stoi(argv[++i]);
    } else if (a == "--write-forces") {
      if (i + 1 >= argc) {
        usage(argv[0]);
        return 1;
      }
      forces_out = argv[++i];
    } else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    } else if (artifact.empty()) {
      artifact = a;
    } else if (structure.empty()) {
      structure = a;
    } else {
      std::cerr << "Unexpected argument: " << a << "\n";
      usage(argv[0]);
      return 1;
    }
  }

  if (artifact.empty()) {
    usage(argv[0]);
    return 1;
  }
  if (num_devices < 1) {
    std::cerr << "--devices must be >= 1\n";
    return 1;
  }

  torch::Device device =
      torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  auto pred = uma::Predictor::from_artifact(artifact, device, num_devices);
  const bool f64 = pred.compute_dtype() == torch::kFloat64;
  std::printf("compute_dtype=%s device=%s devices=%d gp=%s\n",
              f64 ? "float64" : "float32", device.is_cuda() ? "cuda" : "cpu",
              pred.num_devices(), pred.uses_graph_parallel() ? "yes" : "no");

  std::vector<double> pos;
  std::vector<int> z;
  std::vector<double> cell;
  if (!structure.empty()) {
    if (!load_structure(structure, pos, z, cell)) {
      std::cerr << "Failed to read structure " << structure << "\n";
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
  double fmax_atom = 0.0;
  for (int i = 0; i < n; ++i) {
    const double fx = forces[static_cast<size_t>(3 * i)];
    const double fy = forces[static_cast<size_t>(3 * i + 1)];
    const double fz = forces[static_cast<size_t>(3 * i + 2)];
    fmax = std::max(fmax, std::max(std::abs(fx), std::max(std::abs(fy), std::abs(fz))));
    fmax_atom = std::max(fmax_atom, std::sqrt(fx * fx + fy * fy + fz * fz));
  }
  std::printf("n=%d energy=%.12f eV  fmax=%.10e  fmax_atom=%.10e\n", n, out.energy,
              fmax, fmax_atom);
  if (!forces_out.empty()) {
    std::ofstream fo(forces_out);
    if (!fo) {
      std::cerr << "Failed to write forces to " << forces_out << "\n";
      return 1;
    }
    fo.write(reinterpret_cast<const char*>(forces.data()),
             static_cast<std::streamsize>(forces.size() * sizeof(double)));
    std::printf("wrote_forces=%s bytes=%zu\n", forces_out.c_str(),
                forces.size() * sizeof(double));
  }
  return 0;
}
