#include "uma/graph_parallel.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "uma/libtorch_mp.h"

namespace uma {
namespace {

bool file_exists(const std::string& path) {
  std::ifstream in(path);
  return static_cast<bool>(in);
}

std::string json_get_string(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return {};
  pos = json.find('"', pos + needle.size());
  if (pos == std::string::npos) return {};
  ++pos;
  // Handle escaped quotes minimally (paths rarely need them).
  auto end = pos;
  while (end < json.size() && json[end] != '"') {
    if (json[end] == '\\' && end + 1 < json.size()) end += 2;
    else ++end;
  }
  return json.substr(pos, end - pos);
}

bool json_get_bool(const std::string& json, const std::string& key, bool def) {
  const std::string needle = "\"" + key + "\":";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return def;
  pos += needle.size();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (json.compare(pos, 4, "true") == 0) return true;
  if (json.compare(pos, 5, "false") == 0) return false;
  return def;
}

double json_get_number(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  auto pos = json.find(needle);
  if (pos == std::string::npos) {
    throw std::runtime_error("JSON missing key: " + key);
  }
  pos += needle.size();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  return std::stod(json.substr(pos));
}

std::string dtype_name(torch::ScalarType dtype) {
  if (dtype == torch::kFloat64) return "float64";
  if (dtype == torch::kFloat32) return "float32";
  throw std::runtime_error("GraphParallelRuntime: unsupported compute dtype");
}

std::string find_python() {
  if (const char* e = std::getenv("UMA_PYTHON")) {
    if (*e) return e;
  }
  if (const char* e = std::getenv("PYTHON")) {
    if (*e) return e;
  }
  return "python3";
}

}  // namespace

std::string resolve_gp_checkpoint(const std::string& /*artifact_dir*/,
                                  const ArtifactMetadata& metadata) {
  if (!metadata.checkpoint_path.empty() && file_exists(metadata.checkpoint_path)) {
    return metadata.checkpoint_path;
  }
  if (const char* e = std::getenv("UMA_CHECKPOINT")) {
    if (*e && file_exists(e)) return e;
  }
  static const char* kDefault =
      "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt";
  if (file_exists(kDefault)) return kDefault;
  throw std::runtime_error(
      "GraphParallelRuntime: no UMA checkpoint found (set metadata "
      "checkpoint_path or UMA_CHECKPOINT)");
}

bool prefer_python_gp_worker() {
  // Product default is C++ LibTorch MP. Python workers are opt-in only.
  if (const char* e = std::getenv("UMA_PYTHON_GP_WORKER")) {
    if (e[0] == '1' && e[1] == '\0') return true;
  }
  if (const char* e = std::getenv("UMA_ALLOW_RAY_GP")) {
    if (e[0] == '1' && e[1] == '\0') return true;
  }
  return false;
}

bool prefer_native_kokkos_gp_worker() {
  // Used only when Python worker is selected: prefer uma_native_gp_worker.py
  // over Ray unless UMA_ALLOW_RAY_GP=1.
  if (const char* e = std::getenv("UMA_ALLOW_RAY_GP")) {
    if (e[0] == '1' && e[1] == '\0') return false;
  }
  return true;
}

std::string resolve_gp_worker_script() {
  if (const char* e = std::getenv("UMA_GP_WORKER")) {
    if (*e && file_exists(e)) return e;
  }

  const bool native = prefer_native_kokkos_gp_worker();
  const char* native_name = "uma_native_gp_worker.py";
  const char* ray_name = "uma_gp_worker.py";
  const char* primary = native ? native_name : ray_name;
  const char* secondary = native ? ray_name : native_name;

#ifdef UMA_ENGINE_PYTHON_DIR
  {
    std::string p = std::string(UMA_ENGINE_PYTHON_DIR) + "/" + primary;
    if (file_exists(p)) return p;
  }
#endif
  const char* rel_roots[] = {
      "src/ML-UMA/uma-engine/python/",
      "uma-engine/python/",
      "python/",
  };
  for (const char* root : rel_roots) {
    std::string p = std::string(root) + primary;
    if (file_exists(p)) return p;
  }
  // Last resort: the other worker (still no silent Ray if forbid is set).
#ifdef UMA_ENGINE_PYTHON_DIR
  {
    std::string p = std::string(UMA_ENGINE_PYTHON_DIR) + "/" + secondary;
    if (file_exists(p)) return p;
  }
#endif
  for (const char* root : rel_roots) {
    std::string p = std::string(root) + secondary;
    if (file_exists(p)) return p;
  }
  throw std::runtime_error(
      "GraphParallelRuntime: GP worker script not found (set UMA_GP_WORKER)");
}

std::unique_ptr<GraphParallelRuntime> GraphParallelRuntime::create(
    const std::string& artifact_dir, const ArtifactMetadata& metadata,
    int num_devices, torch::ScalarType compute_dtype,
    bool activation_checkpointing) {
  // Single-GPU eager path is allowed ONLY for activation checkpointing (which
  // cannot be traced): run the eager FairChem model (workers=1, no Ray) via the
  // Python worker. Otherwise num_devices>1 is required for GP.
  if (num_devices <= 1 && !activation_checkpointing) {
    throw std::runtime_error(
        "GraphParallelRuntime::create requires num_devices > 1 "
        "(or num_devices==1 with activation_checkpointing)");
  }

  // --- Product path: C++ LibTorch MP (Kokkos peer + vesin) ---
  // Force the Python worker when checkpointing is requested (traced MP can't
  // checkpoint); otherwise honor the normal selection.
  const bool force_python = activation_checkpointing || prefer_python_gp_worker();
  if (!force_python) {
    if (LibtorchMpRuntime::artifacts_present(artifact_dir, num_devices)) {
      auto cpp = LibtorchMpRuntime::try_create(artifact_dir, metadata, num_devices,
                                               compute_dtype);
      if (!cpp) {
        throw std::runtime_error(
            "GraphParallelRuntime: C++ LibTorch MP artifacts present but "
            "LibtorchMpRuntime::try_create failed");
      }
      auto rt = std::unique_ptr<GraphParallelRuntime>(new GraphParallelRuntime(
          num_devices, /*checkpoint=*/artifact_dir, cpp->backend()));
      rt->cpp_mp_ = std::move(cpp);
      return rt;
    }
    throw std::runtime_error(
        "GraphParallelRuntime: C++ LibTorch MP artifacts missing under " +
        artifact_dir +
        " (need model_mp_w" + std::to_string(num_devices) +
        "_r*.pt from python/export_mp_artifact.py). "
        "Opt into legacy Python GP with UMA_PYTHON_GP_WORKER=1. "
        "See agent_stamps/cpp_libtorch/BLOCKERS.md.");
  }

  const std::string checkpoint = resolve_gp_checkpoint(artifact_dir, metadata);
  // Checkpointing needs an eager worker (traced models can't checkpoint).
  //   single-GPU: uma_gp_worker.py (workers=1 -> no Ray)
  //   multi-GPU : uma_native_gp_worker.py (Ray-free, host-staged GP collectives)
  if (activation_checkpointing) {
#ifdef UMA_ENGINE_PYTHON_DIR
    const char* wname =
        (num_devices > 1) ? "uma_native_gp_worker.py" : "uma_gp_worker.py";
    std::string cand = std::string(UMA_ENGINE_PYTHON_DIR) + "/" + wname;
    if (file_exists(cand)) setenv("UMA_GP_WORKER", cand.c_str(), /*overwrite=*/1);
#endif
  }
  const std::string script = resolve_gp_worker_script();
  const std::string python = find_python();
  const std::string task =
      metadata.task_name.empty() ? "omat" : metadata.task_name;

  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (pipe(to_child) != 0 || pipe(from_child) != 0) {
    throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
  }

  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
  }
  if (pid == 0) {
    // Child: stdin = to_child[0], stdout = from_child[1]
    dup2(to_child[0], STDIN_FILENO);
    dup2(from_child[1], STDOUT_FILENO);
    close(to_child[0]);
    close(to_child[1]);
    close(from_child[0]);
    close(from_child[1]);
    // Keep stderr for diagnostics.
    execlp(python.c_str(), python.c_str(), "-u", script.c_str(),
           static_cast<char*>(nullptr));
    // execlp failed
    std::cerr << "GraphParallelRuntime: failed to exec " << python << " " << script
              << ": " << std::strerror(errno) << "\n";
    _exit(127);
  }

  close(to_child[0]);
  close(from_child[1]);

  const std::string default_backend =
      (script.find("uma_native_gp_worker") != std::string::npos)
          ? "kokkos_peer_thread_gp"
          : "fairchem_eager_python";
  auto rt = std::unique_ptr<GraphParallelRuntime>(
      new GraphParallelRuntime(num_devices, checkpoint, default_backend));
  rt->child_pid_ = pid;
  rt->to_child_fd_ = to_child[1];
  rt->from_child_fd_ = from_child[0];

  try {
    // Ready banner.
    const std::string ready = rt->read_line();
    if (!json_get_bool(ready, "ok", false) || !json_get_bool(ready, "ready", false)) {
      throw std::runtime_error("GP worker ready handshake failed: " + ready);
    }

    std::ostringstream init;
    init << "{\"cmd\":\"init\",\"checkpoint\":\"" << checkpoint
         << "\",\"workers\":" << num_devices << ",\"dtype\":\""
         << dtype_name(compute_dtype) << "\",\"task\":\"" << task
         << "\",\"activation_checkpointing\":"
         << (activation_checkpointing ? "true" : "false") << "}";
    rt->write_line(init.str());
    const std::string resp = rt->read_line();
    if (!json_get_bool(resp, "ok", false)) {
      throw std::runtime_error("GP worker init failed: " + resp);
    }
    const std::string backend = json_get_string(resp, "backend");
    if (!backend.empty()) rt->backend_ = backend;

    const char* forbid = std::getenv("UMA_FORBID_RAY_GP");
    // workers==1 uses NO Ray (Ray only kicks in for workers>1), so the
    // single-GPU eager-checkpointing path is exempt from the Ray ban.
    if (forbid != nullptr && forbid[0] == '1' && forbid[1] == '\0' &&
        num_devices > 1) {
      if (rt->backend_ == "fairchem_eager_python" ||
          script.find("uma_gp_worker.py") != std::string::npos) {
        throw std::runtime_error(
            "GraphParallelRuntime: Ray GP forbidden (UMA_FORBID_RAY_GP=1); "
            "use uma_native_gp_worker.py / UMA_NATIVE_KOKKOS_GP=1");
      }
    }

    std::cerr << "uma GraphParallelRuntime: backend=" << rt->backend_
              << " workers=" << num_devices << " dtype=" << dtype_name(compute_dtype)
              << " script=" << script << " checkpoint=" << checkpoint << "\n";
  } catch (...) {
    rt->shutdown_worker();
    throw;
  }
  return rt;
}

GraphParallelRuntime::GraphParallelRuntime(int num_devices, std::string checkpoint,
                                           std::string backend)
    : num_devices_(num_devices),
      checkpoint_(std::move(checkpoint)),
      backend_(std::move(backend)) {}

GraphParallelRuntime::~GraphParallelRuntime() {
  cpp_mp_.reset();
  shutdown_worker();
}

void GraphParallelRuntime::shutdown_worker() {
  if (cpp_mp_) return;
  if (child_pid_ > 0 && to_child_fd_ >= 0) {
    try {
      write_line("{\"cmd\":\"shutdown\"}");
      (void)read_line();
    } catch (...) {
      // fall through to kill
    }
  }
  if (to_child_fd_ >= 0) {
    close(to_child_fd_);
    to_child_fd_ = -1;
  }
  if (from_child_fd_ >= 0) {
    close(from_child_fd_);
    from_child_fd_ = -1;
  }
  if (child_pid_ > 0) {
    int status = 0;
    // Soft wait then SIGTERM.
    for (int i = 0; i < 50; ++i) {
      const pid_t r = waitpid(child_pid_, &status, WNOHANG);
      if (r == child_pid_) {
        child_pid_ = -1;
        return;
      }
      usleep(100000);  // 100 ms
    }
    kill(child_pid_, SIGTERM);
    waitpid(child_pid_, &status, 0);
    child_pid_ = -1;
  }
}

void GraphParallelRuntime::write_exact(const void* src, size_t n) {
  const char* p = static_cast<const char*>(src);
  size_t off = 0;
  while (off < n) {
    const ssize_t w = ::write(to_child_fd_, p + off, n - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("GP write failed: ") + std::strerror(errno));
    }
    if (w == 0) throw std::runtime_error("GP write: unexpected EOF");
    off += static_cast<size_t>(w);
  }
}

void GraphParallelRuntime::read_exact(void* dst, size_t n) {
  char* p = static_cast<char*>(dst);
  size_t off = 0;
  while (off < n) {
    const ssize_t r = ::read(from_child_fd_, p + off, n - off);
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("GP read failed: ") + std::strerror(errno));
    }
    if (r == 0) throw std::runtime_error("GP read: unexpected EOF");
    off += static_cast<size_t>(r);
  }
}

void GraphParallelRuntime::write_line(const std::string& line) {
  write_exact(line.data(), line.size());
  write_exact("\n", 1);
}

std::string GraphParallelRuntime::read_line() {
  std::string out;
  char c = 0;
  while (true) {
    read_exact(&c, 1);
    if (c == '\n') break;
    out.push_back(c);
  }
  return out;
}

Prediction GraphParallelRuntime::predict_host(int n, const double* pos_xyz,
                                              const int* atomic_numbers,
                                              const double* cell_3x3,
                                              const int* pbc_3,
                                              double* forces_out_optional) {
  if (n < 1) throw std::runtime_error("GraphParallelRuntime::predict_host n < 1");
  if (cpp_mp_) {
    return cpp_mp_->predict_host(n, pos_xyz, atomic_numbers, cell_3x3, pbc_3,
                                 forces_out_optional);
  }

  std::ostringstream cmd;
  cmd << "{\"cmd\":\"predict\",\"n\":" << n << ",\"charge\":0,\"spin\":0}";
  write_line(cmd.str());
  std::cerr << "uma GraphParallelRuntime: predict n=" << n << " sent\n" << std::flush;

  // pos f64
  write_exact(pos_xyz, static_cast<size_t>(n) * 3 * sizeof(double));
  // Z i32
  std::vector<int32_t> z32(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) z32[static_cast<size_t>(i)] = atomic_numbers[i];
  write_exact(z32.data(), z32.size() * sizeof(int32_t));
  // cell f64 row-major 3x3
  write_exact(cell_3x3, 9 * sizeof(double));
  // pbc i32
  int32_t pbc32[3] = {pbc_3[0] ? 1 : 0, pbc_3[1] ? 1 : 0, pbc_3[2] ? 1 : 0};
  write_exact(pbc32, 3 * sizeof(int32_t));
  std::cerr << "uma GraphParallelRuntime: predict payload written, waiting...\n"
            << std::flush;

  const std::string resp = read_line();
  std::cerr << "uma GraphParallelRuntime: predict response bytes=" << resp.size()
            << "\n"
            << std::flush;
  if (!json_get_bool(resp, "ok", false)) {
    throw std::runtime_error("GP predict failed: " + resp);
  }
  const double energy = json_get_number(resp, "energy");

  std::vector<double> forces(static_cast<size_t>(n) * 3);
  read_exact(forces.data(), forces.size() * sizeof(double));
  if (forces_out_optional) {
    std::memcpy(forces_out_optional, forces.data(),
                forces.size() * sizeof(double));
  }

  Prediction out;
  out.energy = energy;
  out.forces = torch::from_blob(forces.data(), {n, 3}, torch::kFloat64).clone();
  return out;
}

Prediction GraphParallelRuntime::predict(const torch::Tensor& pos,
                                         const torch::Tensor& atomic_numbers,
                                         const torch::Tensor& cell,
                                         const torch::Tensor& pbc, int64_t charge,
                                         int64_t spin) {
  if (cpp_mp_) {
    return cpp_mp_->predict(pos, atomic_numbers, cell, pbc, charge, spin);
  }
  auto pos_cpu = pos.to(torch::kCPU, torch::kFloat64).contiguous();
  auto z_cpu = atomic_numbers.to(torch::kCPU, torch::kLong).contiguous();
  auto cell_in = cell.to(torch::kCPU, torch::kFloat64).contiguous();
  if (cell_in.dim() == 3) cell_in = cell_in.squeeze(0);
  auto pbc_in = pbc.to(torch::kCPU, torch::kBool).contiguous();
  if (pbc_in.dim() == 2) pbc_in = pbc_in.squeeze(0);

  const int64_t n = pos_cpu.size(0);
  std::vector<double> pos_host(static_cast<size_t>(n) * 3);
  std::vector<int> z_host(static_cast<size_t>(n));
  std::memcpy(pos_host.data(), pos_cpu.data_ptr<double>(),
              sizeof(double) * static_cast<size_t>(n) * 3);
  auto z_acc = z_cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < n; ++i) z_host[static_cast<size_t>(i)] = static_cast<int>(z_acc[i]);

  double cell_host[9];
  auto cell_acc = cell_in.accessor<double, 2>();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) cell_host[3 * i + j] = cell_acc[i][j];

  int pbc_host[3] = {pbc_in[0].item<bool>() ? 1 : 0, pbc_in[1].item<bool>() ? 1 : 0,
                     pbc_in[2].item<bool>() ? 1 : 0};

  return predict_host(static_cast<int>(n), pos_host.data(), z_host.data(), cell_host,
                      pbc_host, nullptr);
}

}  // namespace uma
