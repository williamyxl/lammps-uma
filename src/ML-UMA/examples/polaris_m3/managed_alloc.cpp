// Minimal CUDA managed-memory allocator for torch.cuda.CUDAPluggableAllocator.
// Enables GPU-memory oversubscription: allocations spill to host (Unified Memory)
// and page on demand. Build:
//   nvcc -O2 -shared -Xcompiler -fPIC managed_alloc.cpp -o libmanaged_alloc.so
// Optional env:
//   MANAGED_PREFER_DEVICE=1  -> cudaMemAdvise PreferredLocation=GPU (reduce thrash)
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstdio>

extern "C" {

void* managed_malloc(size_t size, int device, void* /*stream*/) {
  void* ptr = nullptr;
  if (size == 0) size = 1;  // cudaMallocManaged(0) returns null; torch expects a ptr
  cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
  if (err != cudaSuccess || ptr == nullptr) {
    fprintf(stderr, "managed_malloc: cudaMallocManaged(%zu) failed: %s\n",
            size, cudaGetErrorString(err));
    return nullptr;
  }
  // Hint: keep hot data on the GPU; the driver still spills under pressure.
  if (const char* e = std::getenv("MANAGED_PREFER_DEVICE")) {
    if (e[0] == '1') {
      cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
      cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
    }
  }
  return ptr;
}

void managed_free(void* ptr, size_t /*size*/, int /*device*/, void* /*stream*/) {
  if (ptr) cudaFree(ptr);
}

}  // extern "C"
