// Step 18 (N2). Microbenchmark only. Segmented edge->node sum versus index_add_
// at a padded chunk of 65536 edges. Not wired into the model.

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

namespace {

void sync() { torch::xpu::synchronize(0); }

double median_ms(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Shape {
  const char* name;
  int64_t n_nodes;
  int64_t edges;
  int64_t feat;
};

torch::Tensor segreduce_torch(const torch::Tensor& src, const torch::Tensor& offsets) {
  return torch::segment_reduce(src, "sum", /*lengths=*/{}, /*indices=*/{}, offsets,
                              /*axis=*/0, /*unsafe=*/false, /*initial=*/{});
}

torch::Tensor index_add_ref(const torch::Tensor& src, const torch::Tensor& centers, int64_t n_nodes) {
  auto out = torch::zeros({n_nodes, src.size(1)}, src.options());
  out.index_add_(0, centers, src);
  return out;
}

torch::Tensor segreduce_sycl(const torch::Tensor& src, const torch::Tensor& centers,
                             const torch::Tensor& offsets) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("uma_n2::segreduce", "")
                       .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  return op.call(src, centers, offsets);
}

double max_abs(const torch::Tensor& a, const torch::Tensor& b) {
  return (a - b).abs().max().item<double>();
}

struct Times {
  double fwd = 0;
  double both = 0;
};

Times time_fn(const std::function<torch::Tensor()>& fn, const torch::Tensor& src, int reps) {
  for (int i = 0; i < 2; ++i) {
    auto y = fn();
    torch::autograd::variable_list outs{y};
    torch::autograd::variable_list ins{src};
    auto gy = torch::ones_like(y);
    torch::autograd::variable_list gouts{gy};
    torch::autograd::grad(outs, ins, gouts);
  }
  sync();
  std::vector<double> fwd;
  std::vector<double> both;
  for (int i = 0; i < reps; ++i) {
    sync();
    auto t0 = std::chrono::steady_clock::now();
    auto y = fn();
    sync();
    auto t1 = std::chrono::steady_clock::now();
    auto gy = torch::ones_like(y);
    torch::autograd::variable_list outs{y};
    torch::autograd::variable_list ins{src};
    torch::autograd::variable_list gouts{gy};
    torch::autograd::grad(outs, ins, gouts);
    sync();
    auto t2 = std::chrono::steady_clock::now();
    fwd.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    both.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
  }
  return {median_ms(fwd), median_ms(both)};
}

}  // namespace

int main() {
  std::cerr << "N2_ENTER\n" << std::flush;
  if (!torch::xpu::is_available()) {
    std::cerr << "N2_FAIL no xpu\n";
    return 1;
  }
  // Padded chunk is 65536 edges. NaCl 14^3 / 12 ranks is 1830 nodes.
  // ZIF-8 3x3x3 / 12 ranks is 621 nodes. Features: message path L=9, C=256
  // (source and target concatenated, sphere_channels=128); edge-degree path C=128.
  const Shape shapes[] = {
      {"nacl_message", 1830, 65536, 9 * 256},
      {"nacl_degree", 1830, 65536, 9 * 128},
      {"zif_message", 621, 65536, 9 * 256},
  };
  int fails = 0;
  std::cerr << "N2_HDR name nodes edges feat index_fwd index_both seg_fwd seg_both "
               "sycl_fwd sycl_both abs_seg_vs_sycl abs_index_vs_sycl\n";
  for (const Shape& sh : shapes) {
    std::vector<int64_t> centers_h(static_cast<size_t>(sh.edges));
    std::vector<int64_t> offsets_h(static_cast<size_t>(sh.n_nodes + 1));
    const int64_t base = sh.edges / sh.n_nodes;
    const int64_t extra = sh.edges % sh.n_nodes;
    int64_t cursor = 0;
    offsets_h[0] = 0;
    for (int64_t node = 0; node < sh.n_nodes; ++node) {
      const int64_t len = base + (node < extra ? 1 : 0);
      for (int64_t k = 0; k < len; ++k) centers_h[static_cast<size_t>(cursor + k)] = node;
      cursor += len;
      offsets_h[static_cast<size_t>(node + 1)] = cursor;
    }
    auto centers = torch::from_blob(centers_h.data(), {sh.edges}, torch::kLong).clone().to(torch::kXPU);
    auto offsets =
        torch::from_blob(offsets_h.data(), {sh.n_nodes + 1}, torch::kLong).clone().to(torch::kXPU);
    auto src = torch::randn({sh.edges, sh.feat},
                            torch::TensorOptions().dtype(torch::kFloat64).device(torch::kXPU));
    src.requires_grad_(true);

    torch::Tensor y_index;
    torch::Tensor y_seg;
    torch::Tensor y_sycl;
    try {
      y_index = index_add_ref(src, centers, sh.n_nodes);
      y_seg = segreduce_torch(src, offsets);
      y_sycl = segreduce_sycl(src, centers, offsets);
    } catch (const c10::Error& err) {
      std::cerr << "N2_FAIL " << sh.name << " " << err.what_without_backtrace() << "\n";
      return 1;
    }
    const double abs_seg = max_abs(y_seg, y_sycl);
    const double abs_index = max_abs(y_index, y_sycl);
    // Sequential segment sum is the deterministic value. index_add_ uses atomics.
    if (abs_seg > 1e-8) ++fails;

    auto t_index = time_fn([&] { return index_add_ref(src, centers, sh.n_nodes); }, src, 8);
    auto t_seg = time_fn([&] { return segreduce_torch(src, offsets); }, src, 8);
    auto t_sycl = time_fn([&] { return segreduce_sycl(src, centers, offsets); }, src, 8);
    std::cerr << "N2_ROW " << sh.name << " " << sh.n_nodes << " " << sh.edges << " " << sh.feat
              << " " << t_index.fwd << " " << t_index.both << " " << t_seg.fwd << " " << t_seg.both
              << " " << t_sycl.fwd << " " << t_sycl.both << " " << abs_seg << " " << abs_index
              << "\n";
  }
  if (fails != 0) {
    std::cerr << "N2_FAIL rows=" << fails << "\n";
    return 1;
  }
  std::cerr << "N2_REPORT\n";
  return 0;
}
