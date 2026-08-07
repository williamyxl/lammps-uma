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

std::string resolve_gp_worker_script() {
  if (const char* e = std::getenv("UMA_GP_WORKER")) {
    if (*e && file_exists(e)) return e;
  }
#ifdef UMA_ENGINE_PYTHON_DIR
  {
    std::string p = std::string(UMA_ENGINE_PYTHON_DIR) + "/uma_gp_worker.py";
    if (file_exists(p)) return p;
  }
#endif
  // Fallback: walk from CWD / common relative layouts.
  const char* candidates[] = {
      "src/ML-UMA/uma-engine/python/uma_gp_worker.py",
      "uma-engine/python/uma_gp_worker.py",
      "python/uma_gp_worker.py",
  };
  for (const char* c : candidates) {
    if (file_exists(c)) return c;
  }
  throw std::runtime_error(
      "GraphParallelRuntime: uma_gp_worker.py not found (set UMA_GP_WORKER)");
}

std::unique_ptr<GraphParallelRuntime> GraphParallelRuntime::create(
    const std::string& artifact_dir, const ArtifactMetadata& metadata,
    int num_devices, torch::ScalarType compute_dtype) {
  if (num_devices <= 1) {
    throw std::runtime_error(
        "GraphParallelRuntime::create requires num_devices > 1");
  }

  const std::string checkpoint = resolve_gp_checkpoint(artifact_dir, metadata);
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

  auto rt = std::unique_ptr<GraphParallelRuntime>(
      new GraphParallelRuntime(num_devices, checkpoint, "fairchem_eager_python"));
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
         << dtype_name(compute_dtype) << "\",\"task\":\"" << task << "\"}";
    rt->write_line(init.str());
    const std::string resp = rt->read_line();
    if (!json_get_bool(resp, "ok", false)) {
      throw std::runtime_error("GP worker init failed: " + resp);
    }
    const std::string backend = json_get_string(resp, "backend");
    if (!backend.empty()) rt->backend_ = backend;

    std::cerr << "uma GraphParallelRuntime: backend=" << rt->backend_
              << " workers=" << num_devices << " dtype=" << dtype_name(compute_dtype)
              << " checkpoint=" << checkpoint << "\n";
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

GraphParallelRuntime::~GraphParallelRuntime() { shutdown_worker(); }

void GraphParallelRuntime::shutdown_worker() {
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
                                         const torch::Tensor& pbc, int64_t /*charge*/,
                                         int64_t /*spin*/) {
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
