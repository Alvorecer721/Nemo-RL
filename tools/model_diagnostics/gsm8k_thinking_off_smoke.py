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
import re
import subprocess
import time
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    dataset: str
    sample_count: int = Field(gt=0)
    seed: int
    max_sequence_length: int = Field(gt=0)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int
    tensor_parallel_size: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_num_seqs: int = Field(gt=0)
    max_num_batched_tokens: int = Field(gt=0)
    enforce_eager: bool
    enable_thinking: bool
    answer_instruction: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_answer(text: str, expected: str) -> dict:
    # Never extract an answer from an unfinished deliberation block. Completed
    # deliberation is excluded from the final-answer search, even in off mode.
    parts = re.split(r"(<\|inner_prefix\|>|<\|inner_suffix\|>)", text)
    in_thinking = False
    malformed = False
    visible = []
    for part in parts:
        if part == "<|inner_prefix|>":
            malformed |= in_thinking
            in_thinking = True
        elif part == "<|inner_suffix|>":
            malformed |= not in_thinking
            in_thinking = False
        elif not in_thinking:
            visible.append(part)
    final_text = "".join(visible)
    # Preserve special tokens in artifacts; only known terminal tokens are
    # removed for checking that the requested answer ends the response.
    final_text = re.sub(
        r"(?:(?:<\|assistant_end\|>|</s>)\s*)+$", "", final_text
    ).strip()
    pattern = r"\[\[\[\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\]\]\]\s*$"
    match = re.search(pattern, final_text)
    extracted = match.group(1).replace(",", "") if match else None
    completed = not in_thinking and not malformed
    correct = (
        completed
        and extracted is not None
        and Decimal(extracted) == Decimal(expected.replace(",", "").strip())
    )
    return {
        "correct": bool(correct),
        "format_valid": bool(match and completed),
        "extracted_answer": extracted,
        "unclosed_thinking": in_thinking,
        "malformed_thinking": malformed,
        "emitted_thinking": "<|inner_prefix|>" in text,
    }


def self_check() -> None:
    cases = [
        ("The answer is [[[18]]]", "18", True),
        ("[[[1,234.0]]]", "1234", True),
        ("[[[-12]]]", "-12", True),
        ("[[[18]]]<|assistant_end|>", "18", True),
        ("[[[18]]]</s>", "18", True),
        ("18", "18", False),
        ("[[[1,2]]]", "12", False),
        ("[[[17]]]", "18", False),
        ("[[[18]]] trailing answer", "18", False),
        ("<|inner_prefix|>[[[18]]]", "18", False),
        ("<|inner_prefix|>[[[18]]]<|inner_suffix|>[[[17]]]", "18", False),
        ("<|inner_prefix|>work<|inner_suffix|>[[[18]]]", "18", True),
    ]
    for text, answer, expected in cases:
        assert score_answer(text, answer)["correct"] == expected, text
    print(f"scorer_fixtures_passed={len(cases)}", flush=True)


def run(cfg: SmokeConfig, output_dir: Path, *, preflight_only: bool) -> None:
    # Import performs the normal source/image fingerprint check, before loading
    # any engine. This diagnostic does not change the NeMo training/controller path.
    import nemo_rl
    from transformers import AutoTokenizer

    self_check()
    if cfg.enable_thinking:
        raise ValueError("This smoke check requires enable_thinking=false")
    output_dir.mkdir(parents=True, exist_ok=False)
    model = Path(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, local_files_only=True)
    dataset_path = Path(cfg.dataset)
    source_lines = dataset_path.read_text().splitlines()
    selected = sorted(
        random.Random(cfg.seed).sample(range(len(source_lines)), cfg.sample_count)
    )
    rows = []
    for source_index in selected:
        source = json.loads(source_lines[source_index])
        messages = [
            {
                "role": "user",
                "content": source["question"] + "\n\n" + cfg.answer_instruction,
            }
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
            for name in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
                "model.safetensors.index.json",
                "MANIFEST.json",
            ]
        },
        "prompt_tokens_min": min(len(row["prompt_token_ids"]) for row in rows),
        "prompt_tokens_max": max(len(row["prompt_token_ids"]) for row in rows),
        "scope": "inference-only smoke subset; no RL or parameter-gradient check",
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

    from vllm import LLM, SamplingParams

    metadata["versions"] = {
        name: importlib.metadata.version(name)
        for name in ["vllm", "transformers", "torch", "transformer_engine"]
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    generation_config = json.loads((model / "generation_config.json").read_text())
    stop_ids = generation_config["eos_token_id"]
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
    started = time.monotonic()
    engine = LLM(
        model=cfg.model,
        tokenizer=cfg.model,
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
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.config is None or args.output_dir is None:
            parser.error("--config and --output-dir are required for a run")
        run(
            SmokeConfig.model_validate_json(args.config.read_text()),
            args.output_dir,
            preflight_only=args.preflight_only,
        )
