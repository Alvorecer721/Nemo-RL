# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the standard Apertus DeepScaler corpus for Math-with-Judge.

Run this only inside a Slurm compute allocation.  The standard Apertus 1.5 GRPO
probe trains on DeepScaler, whereas the Math-with-Judge agent needs each source
problem represented as a NeMo-Gym Responses API row with its verifier target.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SOURCE = "agentica-org/DeepScaleR-Preview-Dataset"
SPLIT = "train"
AGENT_REF = {
    "type": "responses_api_agents",
    "name": "math_with_judge_simple_agent",
}
POC_ROWS = 160  # 40 steps × 4 prompts per step.


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def validate_revision(revision: str) -> str:
    """Accept a source ref before resolution, or an immutable commit SHA."""
    if not revision or any(char.isspace() for char in revision):
        raise ValueError("revision must be a non-empty Hugging Face revision")
    return revision


def get_prompt_template() -> str:
    repo_dir = Path(__file__).resolve().parents[3]
    template = (repo_dir / "examples/prompts/cot.txt").read_text(encoding="utf-8")
    try:
        template.format("sanity-check problem")
    except (IndexError, KeyError) as error:
        raise ValueError(
            "The Apertus CoT prompt must contain one positional placeholder"
        ) from error
    return template


def convert_record(
    record: Mapping[str, Any], prompt_template: str, source_index: int
) -> dict[str, Any]:
    """Convert one DeepScaler record without retaining source-only fields."""
    problem = record.get("problem")
    answer = record.get("answer")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f"DeepScaler row {source_index}: missing problem")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"DeepScaler row {source_index}: missing answer")

    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt_template.format(problem)}]
        },
        "question": problem,
        "expected_answer": answer,
        "agent_ref": dict(AGENT_REF),
        "dataset_name": SOURCE,
        "dataset_split": SPLIT,
        "source_index": source_index,
        "source_record_sha256": hashlib.sha256(json_bytes(dict(record))).hexdigest(),
    }


def _copy_run_inputs(output_dir: Path) -> dict[str, str]:
    """Store job-local launch inputs because untracked files are not in W&B code snapshots."""
    copied = {}
    for env_name, filename in (
        ("RECIPE", "recipe.yaml"),
        ("MATH_WITH_JUDGE_LAUNCHER", "launcher.slurm"),
        ("MATH_WITH_JUDGE_PREPARATION_SCRIPT", "prepare_deepscaler_math_with_judge.py"),
        ("CONTAINER_ENV", "container.toml"),
    ):
        source = os.environ.get(env_name)
        if not source:
            continue
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"{env_name} does not name a file: {source_path}")
        target = output_dir / filename
        shutil.copy2(source_path, target)
        copied[env_name] = filename
    return copied


def _git_revision(repo_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def prepare(
    records: Iterable[Mapping[str, Any]],
    output_dir: Path,
    source_revision: str,
    *,
    prompt_template: str,
) -> dict[str, Any]:
    """Write the exact 40-step source-order slice and a non-overwriting manifest."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    train_path = temporary_dir / "train.jsonl"
    source_hasher = hashlib.sha256()
    payload_hasher = hashlib.sha256()
    source_rows = 0

    try:
        with train_path.open("wb") as output_file:
            for source_index, record in enumerate(records):
                if source_rows == POC_ROWS:
                    break
                source_record = dict(record)
                source_hasher.update(json_bytes(source_record))
                source_hasher.update(b"\n")
                row_bytes = json_bytes(
                    convert_record(source_record, prompt_template, source_index)
                ) + b"\n"
                output_file.write(row_bytes)
                payload_hasher.update(row_bytes)
                source_rows += 1

        if source_rows < POC_ROWS:
            raise ValueError(
                f"DeepScaler has {source_rows} rows; need at least {POC_ROWS} for 40 steps"
            )

        copied_inputs = _copy_run_inputs(temporary_dir)
        repo_dir = Path(__file__).resolve().parents[3]
        manifest = {
            "dataset": SOURCE,
            "split": SPLIT,
            "source_revision": source_revision,
            "source_rows": source_rows,
            "source_records_sha256": source_hasher.hexdigest(),
            "prepared_file": "train.jsonl",
            "prepared_rows": source_rows,
            "prepared_sha256": payload_hasher.hexdigest(),
            "selection": (
                "first 160 source-order rows; this is the exact non-shuffled slice "
                "consumed by 40 steps"
            ),
            "prompt_template": "examples/prompts/cot.txt",
            "agent_ref": AGENT_REF,
            "reproducibility": {
                "git_revision": _git_revision(repo_dir),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_node_list": os.environ.get("SLURM_NODELIST"),
                "container_config": os.environ.get("CONTAINER_ENV"),
                "copied_inputs": copied_inputs,
            },
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return manifest


def resolve_dataset_revision(requested_revision: str) -> str:
    """Resolve a symbolic ref once and persist the immutable dataset commit in the manifest."""
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(
        repo_id=SOURCE, revision=validate_revision(requested_revision)
    ).sha
    if not isinstance(resolved_revision, str) or len(resolved_revision) != 40:
        raise ValueError(f"Hugging Face did not return a commit SHA: {resolved_revision!r}")
    return resolved_revision.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=os.environ.get("MATH_WITH_JUDGE_DATASET_REVISION", "main"),
        help="DeepScaler commit SHA or source ref; the resolved SHA is recorded in manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get("MATH_WITH_JUDGE_DATA_DIR"),
        required=os.environ.get("MATH_WITH_JUDGE_DATA_DIR") is None,
        help="A new job-scoped directory; an existing directory is never overwritten",
    )
    args = parser.parse_args()

    resolved_revision = resolve_dataset_revision(args.revision)
    from datasets import load_dataset

    records = load_dataset(SOURCE, split=SPLIT, revision=resolved_revision)
    manifest = prepare(
        records,
        args.output_dir,
        resolved_revision,
        prompt_template=get_prompt_template(),
    )
    print(
        f"Prepared {manifest['prepared_rows']} DeepScaler rows at {args.output_dir} "
        f"from revision {resolved_revision}",
        flush=True,
    )


if __name__ == "__main__":
    main()
