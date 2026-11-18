// Keeps the icpx object in the shared library. The static TORCH_LIBRARY
// initializers live in that object.
extern "C" void uma_n5f1_anchor();
extern "C" void uma_n5f1_keep() { uma_n5f1_anchor(); }
