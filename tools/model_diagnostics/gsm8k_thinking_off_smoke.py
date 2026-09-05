# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded inference-only GSM8K check, preserving tokens and stop reasons."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    tokenizer: str
    dataset: str
    sample_count: int = Field(gt=0)
    seed: int
    max_sequence_length: int = Field(gt=0)
    max_new_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int
    tensor_parallel_size: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_num_seqs: int = Field(gt=0)
    max_num_batched_tokens: int = Field(gt=0)
    enforce_eager: bool
    enable_thinking: bool
    system_prompt_file: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_answer(text: str, expected: str) -> dict:
    # Reuse the training scorer; importing it enforces the source/image check.
    from nemo_rl.environments.bracket_math_reward import (
        completed_final_text,
        score_marked_answer,
        thinking_mode_compliant,
    )

    score = score_marked_answer(text, expected, marker="boxed")
    prefix, suffix = "<|inner_prefix|>", "<|inner_suffix|>"
    return {
        "correct": bool(score.outcome),
        "format_valid": bool(score.format),
        "extracted_answer": score.extracted_answer,
        "unclosed_thinking": text.rfind(prefix) > text.rfind(suffix),
        "malformed_thinking": completed_final_text(text) is None,
        "emitted_thinking": not thinking_mode_compliant(text, enable_thinking=False),
    }


def run(cfg: SmokeConfig, output_dir: Path, *, preflight_only: bool) -> None:
    # Import performs the normal source/image fingerprint check, before loading
    # any engine. This diagnostic does not change the NeMo training/controller path.
    import nemo_rl
    from transformers import AutoTokenizer

    if cfg.enable_thinking:
        raise ValueError("This smoke check requires enable_thinking=false")
    output_dir.mkdir(parents=True, exist_ok=False)
    model = Path(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, local_files_only=True)
    dataset_path = Path(cfg.dataset)
    source_lines = dataset_path.read_text().splitlines()
    system_prompt = Path(cfg.system_prompt_file).read_text()
    selected = sorted(
        random.Random(cfg.seed).sample(range(len(source_lines)), cfg.sample_count)
    )
    rows = []
    for source_index in selected:
        source = json.loads(source_lines[source_index])
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": source["question"],
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
        if "Deliberation: disabled" not in prompt or "Deliberation: enabled" in prompt:
            raise ValueError(
                "Checkpoint template did not render the requested off mode"
            )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        budget = cfg.max_sequence_length - len(tokens)
        if budget <= 0:
            raise ValueError(f"Prompt {source_index} exceeds the total context budget")
        if cfg.max_new_tokens is not None:
            budget = min(budget, cfg.max_new_tokens)
        rows.append(
            {
                "source_index": source_index,
                "question": source["question"],
                "expected_answer": source["answer"],
                "prompt": prompt,
                "prompt_token_ids": tokens,
                "max_new_tokens": budget,
            }
        )
    (output_dir / "inputs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    repo = Path(nemo_rl.__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_changes = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
    ).strip()
    if tracked_changes:
        raise ValueError(
            f"Experiment source has uncommitted changes: {tracked_changes}"
        )
    metadata = {
        "config": cfg.model_dump(),
        "source_commit": revision,
        "dataset_sha256": digest(dataset_path),
        "dataset_rows": len(source_lines),
        "checkpoint_metadata_sha256": {
            name: digest(model / name)
            for name in ["config.json", "model.safetensors.index.json"]
        },
        "tokenizer_metadata_sha256": {
            name: digest(Path(cfg.tokenizer) / name)
            for name in [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
            ]
        },
        "system_prompt_sha256": digest(Path(cfg.system_prompt_file)),
        "prompt_tokens_min": min(len(row["prompt_token_ids"]) for row in rows),
        "prompt_tokens_max": max(len(row["prompt_token_ids"]) for row in rows),
        "scope": "fixed held-out GSM8K subset; boxed binary accuracy; one sampled response per question",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "preflight": "PASS",
                "samples": len(rows),
                "prompt_tokens_max": metadata["prompt_tokens_max"],
            }
        ),
        flush=True,
    )
    if preflight_only:
        return

    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

    # Match the certified image's import probe and NeMo's worker startup.
    ensure_vllm_source_compat()
    from vllm import LLM, SamplingParams

    metadata["versions"] = {
        name: importlib.metadata.version(name)
        for name in ["vllm", "transformers", "torch", "openai"]
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    generation_config = json.loads(
        (Path(cfg.tokenizer) / "generation_config.json").read_text()
    )
    stop_ids = generation_config["eos_token_id"]
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
    started = time.monotonic()
    engine = LLM(
        model=cfg.model,
        tokenizer=cfg.tokenizer,
        dtype="bfloat16",
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_model_len=cfg.max_sequence_length,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        enforce_eager=cfg.enforce_eager,
        seed=cfg.seed,
        generation_config="vllm",
    )
    loaded = time.monotonic()
    sampling = [
        SamplingParams(
            n=1,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            seed=cfg.seed + row["source_index"],
            max_tokens=row["max_new_tokens"],
            stop_token_ids=stop_ids,
            skip_special_tokens=False,
        )
        for row in rows
    ]
    outputs = engine.generate(
        [{"prompt_token_ids": row["prompt_token_ids"]} for row in rows],
        sampling,
        use_tqdm=True,
    )
    generated = time.monotonic()
    if len(outputs) != len(rows):
        raise ValueError("Not all evaluation requests returned")
    results = []
    with (output_dir / "responses.jsonl").open("w") as stream:
        for row, request in zip(rows, outputs):
            if list(request.prompt_token_ids) != row["prompt_token_ids"]:
                raise ValueError("Engine returned a mismatched request prompt")
            response = request.outputs[0]
            token_ids = list(response.token_ids)
            if len(token_ids) > row["max_new_tokens"]:
                raise ValueError("Engine exceeded the per-request context budget")
            result = {
                **row,
                "response": response.text,
                "response_token_ids": token_ids,
                "response_tokens": len(token_ids),
                "finish_reason": response.finish_reason,
                "stop_reason": response.stop_reason,
                **score_answer(response.text, row["expected_answer"]),
            }
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    count = len(results)
    summary = {
        "status": "COMPLETE",
        "scope": metadata["scope"],
        "samples": count,
        "correct": sum(row["correct"] for row in results),
        "accuracy": sum(row["correct"] for row in results) / count,
        "format_rate": sum(row["format_valid"] for row in results) / count,
        "cap_hits": sum(row["finish_reason"] == "length" for row in results),
        "unexpected_thinking": sum(row["emitted_thinking"] for row in results),
        "unclosed_thinking": sum(row["unclosed_thinking"] for row in results),
        "mean_response_tokens": sum(row["response_tokens"] for row in results) / count,
        "max_response_tokens": max(row["response_tokens"] for row in results),
        "engine_load_seconds": loaded - started,
        "generation_seconds": generated - loaded,
        "generation_tokens_per_second": sum(row["response_tokens"] for row in results)
        / (generated - loaded),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.config is None or args.output_dir is None:
        parser.error("--config and --output-dir are required for a run")
    run(
        SmokeConfig.model_validate_json(args.config.read_text()),
        args.output_dir,
        preflight_only=args.preflight_only,
    )
