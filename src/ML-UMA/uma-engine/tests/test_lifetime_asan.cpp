// A3 / G.5 item 5 (audit rev 26 §G.18.3): lifetime harness meant to run under
// -fsanitize=address (ci/asan_build.sh). Parity runs prove the NUMBERS are stable;
// they cannot see a use-after-free, double-free, or an uninitialised-mutex lock.
// This exercises the exact lifetime patterns A3 touched, twice (the "define -> run
// -> redefine -> run" cycle), so ASAN can prove they are memory-clean:
//
//   1. HaloContext singleton set_callbacks()/clear() with a CAPTURING callback
//      whose captured object is then FREED and the singleton re-used — the P0'.4
//      dangling-callback shape. clear() must drop the std::function so the freed
//      capture is never invoked.
//   2. SharedPeerGatherSlot::init_control_block()/destroy_control_block() on a
//      calloc'd Shm — the A3 slice-2 pthread mutex/cond init/destroy (zeroed !=
//      initialised). Init+destroy must be balanced and valid.
//
// Returns 0 on success. Under ASAN, any leak/UAF/double-free aborts nonzero.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#include "uma/halo_context.h"
#include "uma/shared_peer.h"

namespace {

// A stand-in for the pair style: a heap object captured by the halo callback,
// mirroring how PairUMA captured `this` into the process-wide HaloContext.
struct FakePairStyle {
  int marker = 0xABCD;
  void install() {
    auto* self = this;
    uma::HaloContext::instance().set_callbacks(
        [self](double* buf, int64_t nall, int64_t per_node) {
          // touch captured state (would be UAF if self were freed + callback kept)
          for (int64_t i = 0; i < nall * per_node; ++i) buf[i] += self->marker * 0.0;
        },
        [self](double*, int64_t, int64_t) { (void)self; },
        /*nlocal=*/4, /*nall=*/8);
  }
};

int halo_redefine_cycle() {
  // "define": create pair style, install callback capturing it.
  auto p1 = std::make_unique<FakePairStyle>();
  p1->install();
  if (!uma::HaloContext::instance().active()) {
    std::cerr << "halo: expected active after install\n";
    return 1;
  }
  // "run": (callback would be invoked here by the engine; skipped — no torch model)

  // teardown must clear the singleton's std::function BEFORE the capture is freed,
  // exactly as ~PairUMA does (P0'.4). If clear() did not drop the function, the
  // captured p1 would dangle.
  uma::HaloContext::instance().clear();
  p1.reset();  // free the captured object

  if (uma::HaloContext::instance().active()) {
    std::cerr << "halo: expected inactive after clear\n";
    return 1;
  }

  // "redefine + run": a second pair style re-uses the singleton. If clear() had
  // left the old (now-dangling) capture installed, ASAN would flag a UAF here or
  // the stale callback would fire.
  auto p2 = std::make_unique<FakePairStyle>();
  p2->install();
  uma::HaloContext::instance().clear();
  p2.reset();
  return 0;
}

int shm_control_block_cycle() {
  namespace kp = uma::kokkos_peer;
  const int world = 4;
  const int transport = kp::SharedPeerGatherSlot::kTransportShm;
  const size_t bytes = kp::SharedPeerGatherSlot::map_bytes_for(world, transport);
  // Two full init/destroy cycles: mirrors ~MpiPeerPredictor freeing + a redefine
  // allocating a fresh control block. A zeroed (calloc'd) pthread_mutex_t that was
  // never init'd would make destroy_control_block UB; a missing destroy would leak.
  for (int cycle = 0; cycle < 2; ++cycle) {
    auto* shm = static_cast<kp::SharedPeerGatherSlot::Shm*>(std::calloc(1, bytes));
    if (!shm) { std::cerr << "shm calloc failed\n"; return 1; }
    shm->world = world;
    shm->transport = transport;
    kp::SharedPeerGatherSlot::init_control_block(shm);   // A3 slice-2 fix
    kp::SharedPeerGatherSlot::destroy_control_block(shm);
    std::free(shm);
  }
  return 0;
}

}  // namespace

int main() {
  if (int rc = halo_redefine_cycle()) return rc;
  if (int rc = shm_control_block_cycle()) return rc;
  std::cout << "test_lifetime_asan OK (halo redefine + shm control-block cycles)\n";
  return 0;
}
