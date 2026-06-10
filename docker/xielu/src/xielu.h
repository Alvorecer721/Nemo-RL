#pragma once
#include <torch/extension.h>

at::Tensor xielu_forward(const at::Tensor &x, const at::Tensor &alpha_p, const at::Tensor &alpha_n, double beta, double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> xielu_backward(const at::Tensor &x, const at::Tensor &go, const at::Tensor &alpha_p, const at::Tensor &alpha_n, double beta, double eps);
