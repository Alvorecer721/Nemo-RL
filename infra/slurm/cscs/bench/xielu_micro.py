import torch

torch.manual_seed(0)
dev = "cuda"

# Mirrors the production formulas by hand: the bridge's compiled_xielu
# (megatron/bridge/models/apertus/xielu_activation.py) is @jit_fuser-wrapped so
# its eager form cannot be imported, and vLLM's _xielu_python is a method. If
# either changes, update this copy or the microbenchmark measures a stale op.
def xielu_python(x, alpha_p, alpha_n, beta, eps):
    ap = torch.nn.functional.softplus(alpha_p)
    an = beta + torch.nn.functional.softplus(alpha_n)
    return torch.where(
        x > 0,
        ap * x * x + beta * x,
        (torch.expm1(torch.min(x, eps)) - x) * an + beta * x,
    )

xielu_compiled = torch.compile(xielu_python, fullgraph=True)

import xielu.ops  # registers torch.classes.xielu
kernel = torch.classes.xielu.XIELU()

ap = torch.tensor([0.8], dtype=torch.bfloat16, device=dev)
an = torch.tensor([0.6], dtype=torch.bfloat16, device=dev)
beta_t = torch.tensor(0.5, dtype=torch.bfloat16, device=dev)
eps_t = torch.tensor(-1e-6, dtype=torch.bfloat16, device=dev)

def bench(fn, x, iters=200):
    for _ in range(20):
        fn(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn(x)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us

for label, shape in [("decode  [16,1,21504]", (16, 1, 21504)), ("prefill [4096,1,21504]", (4096, 1, 21504))]:
    x = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
    r_eager = bench(lambda t: xielu_python(t, ap, an, beta_t, eps_t), x)
    r_comp = bench(lambda t: xielu_compiled(t, ap, an, beta_t, eps_t), x)
    r_kern = bench(lambda t: kernel.forward(t, ap, an, 0.5, -1e-6, False), x)
    gb = x.numel() * 2 * 2 / 1e9
    print(f"{label}: eager-python {r_eager:8.1f} us | compiled {r_comp:7.1f} us ({gb/(r_comp/1e6):5.0f} GB/s) | cuda-kernel {r_kern:7.1f} us ({gb/(r_kern/1e6):5.0f} GB/s)")

# correctness cross-check while we're here
x = torch.randn(256, 1, 21504, dtype=torch.bfloat16, device=dev)
d = (xielu_python(x, ap, an, beta_t, eps_t) - kernel.forward(x, ap, an, 0.5, -1e-6, False)).abs().max()
print("max |python - kernel| =", d.item())
