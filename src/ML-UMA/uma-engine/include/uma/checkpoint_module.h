#pragma once

// C++ activation checkpointing for a traced TorchScript UMA module.
//
// The traced module cannot use torch.utils.checkpoint (it does not trace).
// This custom autograd Function reproduces gradient checkpointing at the C++
// level: forward runs the module WITHOUT retaining the autograd graph (no
// intermediate activations kept), and backward RECOMPUTES the module forward
// with grad enabled to produce the vjp w.r.t. positions. Trades ~1 extra
// forward for ~3x less activation memory, enabling much larger N on one tile.
//
// Only `pos` needs a gradient (forces = -dE/dpos); the other inputs are
// captured as constants. Bit-exact vs the non-checkpointed forward.

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

namespace uma {

struct CheckpointModuleFn
    : public torch::autograd::Function<CheckpointModuleFn> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::jit::script::Module* module,
                               torch::Tensor pos, torch::Tensor z,
                               torch::Tensor cell, torch::Tensor pbc,
                               torch::Tensor eidx, torch::Tensor coff,
                                torch::Tensor charge, torch::Tensor spin) {
    // A3/G.5 item 4 (audit rev 26 §G.18.6): lifetime contract for the smuggled
    // module pointer. `module` is a NON-OWNING borrow of a torch::jit::Module
    // owned by the Predictor for the whole predict() call. forward() and
    // backward() both run inside that single autograd invocation, so the module
    // is guaranteed live when backward() recomputes through it. It is passed as
    // int64_t only because AutogradContext::saved_data holds IValues, not raw
    // pointers; it is NEVER freed here and MUST NOT be an autograd input.
    TORCH_CHECK(module != nullptr,
                "CheckpointModuleFn: null module pointer into saved_data");
    ctx->saved_data["module"] = reinterpret_cast<int64_t>(module);
    ctx->save_for_backward({pos, z, cell, pbc, eidx, coff, charge, spin});
    torch::NoGradGuard no_grad;  // do not retain the forward graph
    std::vector<torch::jit::IValue> args = {pos,  z,      cell,   pbc,
                                            eidx, coff,   charge, spin};
    return module->forward(args).toTensor();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto* module = reinterpret_cast<torch::jit::script::Module*>(
        ctx->saved_data["module"].toInt());
    TORCH_CHECK(module != nullptr,
                "CheckpointModuleFn::backward: module pointer lost across the "
                "autograd boundary (owner freed before backward)");
    auto saved = ctx->get_saved_variables();
    auto pos = saved[0].detach().set_requires_grad(true);
    auto z = saved[1], cell = saved[2], pbc = saved[3];
    auto eidx = saved[4], coff = saved[5], charge = saved[6], spin = saved[7];
    torch::Tensor normed;
    {
      torch::autograd::AutoGradMode grad_on(true);
      std::vector<torch::jit::IValue> args = {pos,  z,      cell,   pbc,
                                              eidx, coff,   charge, spin};
      normed = module->forward(args).toTensor();
    }
    auto grads = torch::autograd::grad({normed}, {pos}, {grad_outputs[0]},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/true);
    torch::Tensor gpos =
        grads[0].defined() ? grads[0] : torch::zeros_like(pos);
    // Only pos (arg index 1) gets a gradient; module ptr + others are null.
    return {torch::Tensor(), gpos,           torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor()};
  }
};

// Env gate: UMA_CKPT (whole-module checkpoint). A10 (audit rev 29 §G.25 — owner
// directive): activation checkpointing defaults OFF on ALL builds, so an
// unflagged run is fully differentiable (NPT/virial works) at lower capacity.
// The previous XPU "default ON" special case is removed. UMA_CKPT=1 opts back in.
// NOTE: on production artifacts this whole-module branch is dead code (the
// per-chunk ops short-circuit first, predictor.cpp) — the effective AC default is
// controlled by UMA_AC in block_context.cpp (see §G.25.1). This flag is kept
// consistent with the directive for the no-per-chunk-op case.
inline bool checkpoint_enabled() {
  const char* e = std::getenv("UMA_CKPT");
  if (e == nullptr) return false;  // A10: default OFF on every build
  return !(e[0] == '0' && e[1] == '\0');
}

}  // namespace uma
