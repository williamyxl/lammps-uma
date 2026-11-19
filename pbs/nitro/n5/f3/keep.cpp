// Keeps the icpx object in the shared library. The static TORCH_LIBRARY
// initializers live in that object.
extern "C" void uma_n5f3_anchor();
extern "C" void uma_n5f3_keep() { uma_n5f3_anchor(); }
