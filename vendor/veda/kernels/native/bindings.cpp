#include <torch/extension.h>
void veda_attention(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("attention", &veda_attention);}
