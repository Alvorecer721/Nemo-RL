"""Fixed-manifest, thinking-enabled math evaluation; generate before grading."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a precise math solver.\n"
    "Solve the problem step by step, then give your final answer inside \\boxed{}."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_token_budget(prompt_tokens: int, generated: int, total: int) -> None:
    if prompt_tokens + generated > total:
        raise ValueError(
            f"Prompt would reduce the generated-token budget: {prompt_tokens}+{generated}>{total}"
        )


def requests_for_shard(
    rows: list[dict], shard: int, shards: int, seed: int
) -> list[dict]:
    if not 0 <= shard < shards:
        raise ValueError("Invalid shard")
    seen = set()
    output = []
    ordinal = 0
    for row in rows:
        key = (row["dataset"], row["id"])
        if key in seen:
            raise ValueError(f"Duplicate problem: {key}")
        seen.add(key)
        for repeat in range(row["repeats"]):
            if ordinal % shards == shard:
                output.append(
                    {
                        **row,
                        "repeat": repeat,
                        "request_id": f"{key[0]}/{key[1]}/{repeat}",
                        "sampling_seed": seed + ordinal,
                        "shard": shard,
                    }
                )
            ordinal += 1
    return output


def generate(args: argparse.Namespace) -> None:
    import nemo_rl  # noqa: F401 -- enforce the source/image fingerprint
    from transformers import AutoTokenizer

    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

    source = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if head != args.expected_head:
        raise ValueError(f"Source changed: {head} != {args.expected_head}")
    if subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--ignore-submodules=all",
        ],
        cwd=source,
        text=True,
    ).strip():
        raise ValueError("Dirty experiment source")
    rows = requests_for_shard(
        [json.loads(s) for s in args.manifest.read_text().splitlines()],
        args.shard,
        args.shards,
        args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    for row in rows:
        row["prompt"] = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if "Deliberation: enabled" not in row["prompt"]:
            raise ValueError("Thinking instruction was not rendered")
        row["prompt_token_ids"] = tokenizer.encode(
            row["prompt"], add_special_tokens=False
        )
        validate_token_budget(
            len(row["prompt_token_ids"]), args.max_new_tokens, args.max_model_len
        )
    (args.output / "inputs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    metadata = {
        "source_head": head,
        "manifest_sha256": digest(args.manifest),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tokenizer_files": {
            n: digest(Path(args.tokenizer) / n)
            for n in [
                "config.json",
                "tokenizer.json",
                "chat_template.jinja",
                "generation_config.json",
            ]
        },
        "model_index_sha256": digest(Path(args.model) / "model.safetensors.index.json"),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "thinking": True,
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.75,
        "max_num_seqs": 128,
        "shard": args.shard,
        "shards": args.shards,
        "seed": args.seed,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "requests": len(rows),
        "versions": {
            p: importlib.metadata.version(p) for p in ["vllm", "torch", "transformers"]
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    ensure_vllm_source_compat()
    from vllm import LLM, SamplingParams

    start = time.monotonic()
    engine = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.75,
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        enforce_eager=False,
        seed=args.seed,
        generation_config="vllm",
    )
    loaded = time.monotonic()
    with (args.output / "responses.jsonl").open("w") as stream:
        # Bound lost work on a failed job while keeping the engine saturated.
        for offset in range(0, len(rows), 256):
            batch = rows[offset : offset + 256]
            params = [
                SamplingParams(
                    n=1,
                    temperature=r["temperature"],
                    top_p=r["top_p"],
                    top_k=-1,
                    seed=r["sampling_seed"],
                    max_tokens=args.max_new_tokens,
                    stop_token_ids=[2, 68, 72],
                    skip_special_tokens=False,
                )
                for r in batch
            ]
            outputs = engine.generate(
                [{"prompt_token_ids": r["prompt_token_ids"]} for r in batch],
                params,
                use_tqdm=True,
            )
            if len(outputs) != len(batch):
                raise ValueError("Incomplete engine output")
            for row, request in zip(batch, outputs):
                if list(request.prompt_token_ids) != row["prompt_token_ids"]:
                    raise ValueError("Engine reordered prompts")
                response = request.outputs[0]
                if len(response.token_ids) > args.max_new_tokens:
                    raise ValueError("Engine exceeded generated-token cap")
                stream.write(
                    json.dumps(
                        {
                            **row,
                            "response": response.text,
                            "response_token_ids": list(response.token_ids),
                            "response_tokens": len(response.token_ids),
                            "finish_reason": response.finish_reason,
                            "stop_reason": response.stop_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
    (args.output / "generation_complete.json").write_text(
        json.dumps(
            {
                "requests": len(rows),
                "load_seconds": loaded - start,
                "generation_seconds": time.monotonic() - loaded,
                "status": "COMPLETE",
            },
            indent=2,
        )
        + "\n"
    )


def score_response(text: str, answer: str, dataset: str) -> dict:
    from nemo_rl.environments.bracket_math_reward import (
        _last_boxed_content,
        completed_final_text,
        score_marked_answer,
    )

    final = completed_final_text(text)
    boxed = _last_boxed_content(final) if final is not None else None
    if dataset == "math500":
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig

        correct = boxed is not None and verify(
            parse(
                "\\boxed{" + answer + "}", extraction_config=[LatexExtractionConfig()]
            ),
            parse(
                "\\boxed{" + boxed + "}", extraction_config=[LatexExtractionConfig()]
            ),
        )
    else:
        correct = bool(score_marked_answer(text, answer, marker="boxed").outcome)
    return {
        "correct": bool(correct),
        "format_valid": boxed is not None,
        "extracted_answer": boxed,
        "malformed_thinking": final is None,
        "emitted_thinking": "<|inner_prefix|>" in text,
    }


def grade(args: argparse.Namespace) -> None:
    # Fail on broken scorer dependencies before touching the result file.
    assert score_response(
        "<|inner_prefix|>\\boxed{2}<|inner_suffix|>\\boxed{1}", "1", "aime24"
    )["correct"]
    assert not score_response("<|inner_prefix|>\\boxed{1}", "1", "aime24")["correct"]
    assert score_response(r"\boxed{\frac{1}{2}}", "0.5", "math500")["correct"]
    assert score_response(
        r"\boxed{(3, \frac{\pi}{2})}", r"\left(3, \frac{\pi}{2}\right)", "math500"
    )["correct"]
    assert not score_response(r"\boxed{2}", "1", "math500")["correct"]
    assert (args.output / "generation_complete.json").exists()
    rows = []
    with (args.output / "graded.jsonl").open("w") as stream:
        for line in (args.output / "responses.jsonl").read_text().splitlines():
            row = json.loads(line)
            result = {
                **row,
                **score_response(row["response"], row["answer"], row["dataset"]),
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            rows.append(result)
    complete = json.loads((args.output / "generation_complete.json").read_text())
    if len(rows) != complete["requests"]:
        raise ValueError("Response count differs from generation completion")
    summaries = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        summaries[dataset] = summarize(subset)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
    return {
        "samples": len(rows),
        "problems": len(groups),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "format_rate": sum(r["format_valid"] for r in rows) / len(rows),
        "thinking_rate": sum(r["emitted_thinking"] for r in rows) / len(rows),
        "malformed_thinking": sum(r["malformed_thinking"] for r in rows),
        "cap_hits": sum(r["finish_reason"] == "length" for r in rows),
        "mean_tokens": sum(r["response_tokens"] for r in rows) / len(rows),
        "max_tokens": max(r["response_tokens"] for r in rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "grade"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--expected-head")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=12288)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parsed = parser.parse_args()
    (generate if parsed.mode == "generate" else grade)(parsed)
