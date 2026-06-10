import torch
from torch import Tensor

import _xielu  # noqa: F401


def xielu(
    x: Tensor, alpha_p: Tensor, alpha_n: Tensor, beta: float = 0.5, eps: float = -1e-6
) -> Tensor:
    return torch.ops.xielu.xielu(x, alpha_p, alpha_n, beta, eps)
