#!/usr/bin/env python3
"""Generate matched DAPO prompts directly from the raw Apertus 70B HF weights."""

import hashlib
import json
import os
import time
from pathlib import Path


def _load_unique_prompts(path: Path, count: int) -> list[str]:
    prompts: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            prompt = record["content"][0][0]
            if prompt not in prompts:
                prompts.append(prompt)
            if len(prompts) == count:
                break
    if len(prompts) != count:
        raise AssertionError(f"Expected {count} unique prompts, found {len(prompts)}")
    return prompts


def main() -> None:
    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

    ensure_vllm_source_compat()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    model = Path(os.environ["APERTUS_DIRECT_MODEL"]).resolve(strict=True)
    train_data = Path(os.environ["APERTUS_DIRECT_TRAIN_DATA"]).resolve(strict=True)
    output = Path(os.environ["APERTUS_DIRECT_OUTPUT"])
    source_head = os.environ["APERTUS_DIRECT_SOURCE_HEAD"]
    prompt_count = int(os.environ.get("APERTUS_DIRECT_PROMPTS", "4"))
    samples_per_prompt = int(os.environ.get("APERTUS_DIRECT_SAMPLES", "2"))
    max_tokens = int(os.environ.get("APERTUS_DIRECT_MAX_TOKENS", "512"))

    assert vllm.__version__ == "0.25.1", vllm.__version__
    assert torch.cuda.device_count() == 4, torch.cuda.device_count()
    prompts = _load_unique_prompts(train_data, prompt_count)
    llm = LLM(
        model=str(model),
        tokenizer=str(model),
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.50,
        enforce_eager=False,
        tensor_parallel_size=4,
        distributed_executor_backend="ray",
        trust_remote_code=True,
    )
    start = time.perf_counter()
    results = llm.generate(
        prompts,
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_tokens,
            n=samples_per_prompt,
            seed=42,
        ),
    )
    elapsed = time.perf_counter() - start

    samples = []
    total_tokens = 0
    for prompt_index, result in enumerate(results):
        prompt_hash = hashlib.sha256(prompts[prompt_index].encode()).hexdigest()
        for sample_index, generation in enumerate(result.outputs):
            total_tokens += len(generation.token_ids)
            samples.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_sha256": prompt_hash,
                    "sample_index": sample_index,
                    "finish_reason": generation.finish_reason,
                    "token_count": len(generation.token_ids),
                    "text": generation.text,
                }
            )
            print(
                f"prompt={prompt_index} sample={sample_index} "
                f"tokens={len(generation.token_ids)} text={generation.text!r}"
            )

    payload = {
        "schema": "nemo-rl.apertus70b-direct-hf-math.v1",
        "source_head": source_head,
        "model": str(model),
        "train_data": str(train_data),
        "prompt_count": prompt_count,
        "samples_per_prompt": samples_per_prompt,
        "max_tokens": max_tokens,
        "elapsed_s": elapsed,
        "generated_tokens": total_tokens,
        "generated_tokens_per_s": total_tokens / elapsed,
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise AssertionError(f"Refusing to replace evidence: {output}")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"evidence={output}")
    print("apertus70b_direct_hf_math=OK")


if __name__ == "__main__":
    main()
