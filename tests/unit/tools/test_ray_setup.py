# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failing per-node preflight must stop the generated Ray startup script."""

import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("role", ["head", "worker"])
@pytest.mark.parametrize("setup_status", [0, 42])
def test_setup_failure_stops_ray_startup(tmp_path, role, setup_status):
    source = (Path(__file__).resolve().parents[3] / "ray.sub").read_text()
    blocks = re.findall(
        r'  if \[\[ -n "\$SETUP_COMMAND_FILE".*?\n  fi', source, re.DOTALL
    )
    assert len(blocks) == 2
    block = blocks[["head", "worker"].index(role)]
    setup = tmp_path / "setup.sh"
    setup.write_text(f"exit {setup_status}\n")
    environment = {
        **os.environ,
        "SETUP_COMMAND_FILE": str(setup),
        "LOG_DIR": str(tmp_path),
    }
    # Apply the same expanding heredoc as ray.sub, then execute its actual guard
    # without errexit, matching the head and worker startup shells.
    rendered = subprocess.run(
        ["bash"],
        input=f"cat <<EOF\n{block}\nEOF\n",
        text=True,
        env=environment,
        capture_output=True,
        check=True,
    ).stdout
    result = subprocess.run(
        ["bash"],
        input=rendered + '\nprintf reached > "$LOG_DIR/reached"\n',
        text=True,
        env=environment,
        capture_output=True,
    )
    assert result.returncode == setup_status, result.stderr
    assert (tmp_path / "reached").exists() == (setup_status == 0)
    assert (tmp_path / "ENDED").exists() == (setup_status != 0)
