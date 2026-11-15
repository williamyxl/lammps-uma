// Step 17 (S5). Report only. Launch cost and FP64 bandwidth of silu_affine
// against the torch elementwise formula, at the chunk lengths used by the
// stock model: 65536 edges, that length times edge_channels (128), and that
// length times the 9x9 Wigner block. The kernel waits before it returns, so
// each time includes that wait. The first calls are discarded.

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

at::Tensor silu_affine(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("uma_sycl::silu_affine", "")
                       .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  return op.call(x, w, b);
}

at::Tensor torch_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  return torch::silu(x * w + b);
}

void sync() { torch::xpu::synchronize(0); }

double median_ms(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Row {
  const char* name;
  int64_t n;
  int reps;
};

}  // namespace

int main() {
  std::cerr << "S5_ENTER\n" << std::flush;
  if (!torch::xpu::is_available()) {
    std::cerr << "S5_FAIL no xpu\n";
    return 1;
  }
  // Chunk length from the exporter (EDGE_AC_CHUNK / activation_checkpoint_chunk_size).
  // Widths from the uma-s-1p2 config recorded in the plan: edge_channels=128, lmax=2.
  const Row rows[] = {
      {"launch", 1, 50},
      {"edges", 65536, 30},
      {"edges_x_channels", static_cast<int64_t>(65536) * 128, 20},
      {"edges_x_wigner", static_cast<int64_t>(65536) * 9 * 9, 20},
  };
  int fails = 0;
  std::cerr << "S5_HDR name n bytes median_sycl_ms median_torch_ms sycl_GBs torch_GBs rel\n";
  for (const Row& row : rows) {
    auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kXPU);
    auto x = torch::randn({row.n}, opt);
    auto w = torch::randn({row.n}, opt);
    auto b = torch::randn({row.n}, opt);
    for (int i = 0; i < 3; ++i) {
      auto y = silu_affine(x, w, b);
      auto yt = torch_silu(x, w, b);
      (void)y;
      (void)yt;
    }
    sync();
    auto y = silu_affine(x, w, b);
    auto yt = torch_silu(x, w, b);
    const double scale = std::max(1.0, yt.abs().max().item<double>());
    const double rel = (y - yt).abs().max().item<double>() / scale;
    if (rel > 1e-12) ++fails;

    std::vector<double> sycl_ms;
    std::vector<double> torch_ms;
    sycl_ms.reserve(row.reps);
    torch_ms.reserve(row.reps);
    for (int i = 0; i < row.reps; ++i) {
      sync();
      auto t0 = std::chrono::steady_clock::now();
      auto ys = silu_affine(x, w, b);
      sync();
      auto t1 = std::chrono::steady_clock::now();
      (void)ys;
      sycl_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    for (int i = 0; i < row.reps; ++i) {
      sync();
      auto t0 = std::chrono::steady_clock::now();
      auto yt2 = torch_silu(x, w, b);
      sync();
      auto t1 = std::chrono::steady_clock::now();
      (void)yt2;
      torch_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const double ms_s = median_ms(sycl_ms);
    const double ms_t = median_ms(torch_ms);
    // x, w, b read and y written.
    const double bytes = 4.0 * static_cast<double>(row.n) * 8.0;
    const double gb_s = bytes / (ms_s * 1e6);
    const double gb_t = bytes / (ms_t * 1e6);
    std::cerr << "S5_ROW " << row.name << " " << row.n << " " << static_cast<int64_t>(bytes)
              << " " << ms_s << " " << ms_t << " " << gb_s << " " << gb_t << " " << rel << "\n";
  }
  if (fails != 0) {
    std::cerr << "S5_FAIL rows=" << fails << "\n";
    return 1;
  }
  std::cerr << "S5_REPORT\n";
  return 0;
}
