# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MBPP function-coding tasks for workspace_swe, inside a Slurm container.

This is an adaptation of MBPP, not repository repair or the official MBPP metric.
The first assertion is a public interface example; remaining distinct assertions
are withheld from the prompt and graded using workspace_swe's pytest reward.
Reference solutions are never executed or included in generated task files.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


SOURCE = "google-research-datasets/mbpp"
SPLIT_IDS = {"train": range(601, 975), "validation": range(511, 601)}
SYSTEM_PROMPT = (
    "You are implementing a Python function in a workspace. Take one action per turn: "
    '<cmd>shell command</cmd>, <write path="solution.py">complete Python code</write>, '
    "or <submit/> to run the withheld tests. Implement the requested interface in solution.py."
)


def validate_revision(revision: str) -> str:
    """Require an immutable Hugging Face commit, rather than a branch or tag."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full 40-character Hugging Face commit SHA")
    return revision.lower()


def convert_record(record: dict, split: str, revision: str, max_steps: int = 8) -> dict:
    revision = validate_revision(revision)
    task_id = record["task_id"]
    if split not in SPLIT_IDS or type(task_id) is not int or task_id not in SPLIT_IDS[split]:
        raise ValueError(f"Task {task_id!r} does not belong to the official {split!r} split")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    description = record["text"]
    assertions = record["test_list"]
    setup = record.get("test_setup_code", "")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Task {task_id}: missing description")
    if not isinstance(assertions, list) or len(assertions) < 2:
        raise ValueError(f"Task {task_id}: need a public assertion and withheld assertions")
    if not isinstance(setup, str):
        raise ValueError(f"Task {task_id}: invalid test_setup_code")
    ast.parse(setup)
    assertion_keys = []
    for assertion in assertions:
        if not isinstance(assertion, str):
            raise ValueError(f"Task {task_id}: assertions must be strings")
        tree = ast.parse(assertion)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            raise ValueError(f"Task {task_id}: expected one assert statement per test")
        assertion_keys.append(ast.dump(tree))
    withheld = []
    seen = {assertion_keys[0]}
    for assertion, key in zip(assertions[1:], assertion_keys[1:]):
        if key not in seen:
            withheld.append(assertion)
            seen.add(key)
    if not withheld:
        raise ValueError(f"Task {task_id}: no tests distinct from the public example")

    # Each assertion is a real pytest test, so successful collection cannot be
    # confused with passed tests. Load a fresh solution namespace for every case.
    grader = "import runpy\nfrom pathlib import Path\n\n"
    for index, assertion in enumerate(withheld):
        grader += (
            f"def test_withheld_{index}():\n"
            '    namespace = runpy.run_path(str(Path(__file__).with_name("solution.py")))\n'
            f"    exec({setup!r}, namespace)\n"
            f"    exec({assertion!r}, namespace)\n\n"
        )
    public_example = "\n".join(part for part in (setup.rstrip(), assertions[0]) if part)
    task = (
        f"MBPP function-coding task {task_id}:\n{description.strip()}\n\n"
        "Implement solution.py. The following public example specifies the required "
        "names and calling convention; additional assertions are withheld for grading.\n"
        f"```python\n{public_example}\n```\n"
        "Write your implementation, then submit. Do not modify the grader."
    )
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Implement the task provided by the workspace."},
            ]
        },
        "agent_ref": {"type": "responses_api_agents", "name": "workspace_swe_gymnasium_agent"},
        "task": task,
        "files": {"solution.py": "# Implement the requested Python interface here.\n"},
        "hidden_tests": {"test_mbpp.py": grader},
        "test_cmd": "python -m pytest -q test_mbpp.py",
        "max_steps": max_steps,
        "dataset_name": SOURCE,
        "dataset_split": split,
        "source_revision": revision,
        "task_id": task_id,
    }


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def prepare(splits: dict, output_dir: Path, revision: str, max_steps: int = 8) -> dict:
    revision = validate_revision(revision)
    prepared = {}
    manifest = {
        "dataset": SOURCE,
        "configuration": "full",
        "revision": revision,
        "adaptation": "function coding; first assertion public; distinct remaining assertions withheld",
        "license": "CC-BY-4.0",
        "attribution": "MBPP, Austin et al. (2021), Program Synthesis with Large Language Models",
        "paper_url": "https://arxiv.org/abs/2108.07732",
        "source_url": f"https://huggingface.co/datasets/{SOURCE}/tree/{revision}",
        "max_steps": max_steps,
        "splits": {},
    }
    for split, expected_ids in SPLIT_IDS.items():
        records = sorted((dict(row) for row in splits[split]), key=lambda row: row["task_id"])
        ids = [row["task_id"] for row in records]
        if ids != list(expected_ids):
            raise ValueError(f"{split}: expected the complete official ID range, without duplicates")
        rows = [convert_record(row, split, revision, max_steps) for row in records]
        payload = b"".join(json_bytes(row) + b"\n" for row in rows)
        filename = f"mbpp_{split}.jsonl"
        prepared[filename] = payload
        manifest["splits"][split] = {
            "file": filename,
            "rows": len(rows),
            "task_ids": ids,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_records_sha256": hashlib.sha256(json_bytes(records)).hexdigest(),
        }
    # Validate all records before creating outputs; never overwrite an earlier dataset.
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, payload in prepared.items():
        (output_dir / filename).write_bytes(payload)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Full Hugging Face dataset commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    revision = validate_revision(args.revision)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    from datasets import load_dataset

    splits = {
        split: load_dataset(SOURCE, "full", split=split, revision=revision)
        for split in SPLIT_IDS
    }
    manifest = prepare(splits, args.output_dir, revision, args.max_steps)
    print(f"Prepared {sum(item['rows'] for item in manifest['splits'].values())} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
