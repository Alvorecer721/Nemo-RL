"""Runtime tensor checks for the Apertus factor-32 research arm.

These checks read the constructed model, including vLLM's cached rotations.
They never modify frequencies or repair a misconfigured model.
"""

import json
import math
import os
from pathlib import Path
import socket
from typing import Any

import torch


def expected_frequencies(factor: float) -> torch.Tensor:
    # Apertus checkpoint: 128 dimensions, theta=4M; Llama3 bands 2048/8192.
    frequency = 4_000_000.0 ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    wavelength = 2 * math.pi / frequency
    blend = ((8192 / wavelength - 1) / 3).clamp(0, 1)
    return (frequency * (blend + (1 - blend) / factor)).float()


def check_frequencies(actual: torch.Tensor, factor: float) -> list[float]:
    actual = actual.detach().cpu().float()
    expected = expected_frequencies(factor)
    if actual.shape != expected.shape or not torch.allclose(
        actual, expected, rtol=2e-6, atol=1e-10
    ):
        raise ValueError(
            f"Actual RoPE frequencies do not match Apertus factor {factor}"
        )
    if factor == 32 and torch.allclose(
        actual, expected_frequencies(8), rtol=2e-6, atol=1e-10
    ):
        raise ValueError("Actual RoPE frequencies still match factor 8")
    return actual.tolist()


def check_cache(actual: torch.Tensor, positions: torch.Tensor, factor: float) -> float:
    phase = positions.cpu().float()[:, None] * expected_frequencies(factor)[None, :]
    expected = torch.cat([phase.cos(), phase.sin()], dim=-1)
    actual = actual.detach().cpu()
    expected = expected.to(actual.dtype).float()
    if actual.shape != expected.shape or not torch.allclose(
        actual.float(), expected, atol=0.009, rtol=0.005
    ):
        raise ValueError(f"Actual RoPE cos/sin cache does not match factor {factor}")
    return float((actual.float() - expected).abs().max())


def write_report(
    role: str, factor: float, modules: list[dict[str, Any]]
) -> dict[str, Any]:
    if not modules:
        raise ValueError(f"No actual RoPE modules found on {role} worker")
    report = {
        "status": "PASS",
        "role": role,
        "factor": factor,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": torch.distributed.get_rank(),
        "modules": modules,
    }
    directory = Path(os.environ["NRL_ROPE_AUDIT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    target = (
        directory / f"{role}-{report['hostname']}-{report['rank']}-{os.getpid()}.json"
    )
    with target.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        f"ROPE_RUNTIME_VERIFIED role={role} rank={report['rank']} factor={factor} modules={len(modules)} artifact={target}",
        flush=True,
    )
    return report


def verify_megatron(model: torch.nn.Module, factor: float) -> dict[str, Any]:
    records = []
    for name, module in model.named_modules():
        if type(module).__name__ == "RotaryEmbedding":
            if module.seq_len_interpolation_factor is not None:
                raise ValueError("Unexpected additional linear position interpolation")
            values = check_frequencies(module.inv_freq, factor)
            records.append(
                {"name": name, "class": type(module).__name__, "inv_freq": values}
            )
    return write_report("megatron", factor, records)


def verify_vllm(worker: Any, factor: float) -> dict[str, Any]:
    config = worker.model_config.hf_config
    parameters = config.rope_parameters
    if parameters["factor"] != factor or parameters["rope_type"] != "llama3":
        raise ValueError("vLLM's resolved HF configuration differs from requested RoPE")
    records = []
    for name, module in worker.model_runner.model.named_modules():
        if type(module).__name__ == "Llama3RotaryEmbedding":
            if module.scaling_factor != factor or module.base != 4_000_000:
                raise ValueError(
                    "vLLM's constructed RoPE parameters differ from requested values"
                )
            values = check_frequencies(module._compute_inv_freq(module.base), factor)
            positions = torch.tensor([1, 2048, 8192, 12287], dtype=torch.long)
            cache = module.cos_sin_cache.index_select(
                0, positions.to(module.cos_sin_cache.device)
            )
            error = check_cache(cache, positions, factor)
            records.append(
                {
                    "name": name,
                    "class": type(module).__name__,
                    "inv_freq": values,
                    "cache_positions": positions.tolist(),
                    "cache_max_abs_error": error,
                }
            )
    return write_report("vllm", factor, records)
