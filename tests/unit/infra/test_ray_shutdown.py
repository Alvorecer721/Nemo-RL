# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise launcher cleanup with real child processes, without a Ray cluster."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run_shutdown(tmp_path, child, grace=1, step="12345.6"):
    source = (ROOT / "ray.sub").read_text()
    start = source.index("wait_for_ray_cluster() {")
    function = source[start : source.index("\n}\n", start) + 3]
    cancel = tmp_path / "scancel"
    cancel.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CANCEL_LOG"\n')
    cancel.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CANCEL_LOG=str(tmp_path / "cancel.log"),
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            function
            + f'\n{child} &\npid=$!\nwait_for_ray_cluster "$pid" "{step}" {grace}\n',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result, tmp_path / "cancel.log"


def test_completed_child_preserves_status(tmp_path):
    result, cancel = run_shutdown(tmp_path, "(exit 7)")
    assert result.returncode == 7
    assert not cancel.exists()


def test_stuck_child_is_bounded_and_only_its_step_is_cancelled(tmp_path):
    result, cancel = run_shutdown(tmp_path, "sleep 600")
    assert result.returncode == 124
    assert cancel.read_text().strip() == "--signal=KILL 12345.6"
    assert "cleanup deadline" in result.stderr


def test_success_does_not_cancel_step(tmp_path):
    result, cancel = run_shutdown(tmp_path, "(exit 0)")
    assert result.returncode == 0
    assert not cancel.exists()


def test_invalid_identity_never_cancels_entire_job(tmp_path):
    result, cancel = run_shutdown(tmp_path, "sleep 600", step="12345")
    assert result.returncode == 124
    assert not cancel.exists()
    assert "Invalid Ray step identity" in result.stderr


def test_child_ignoring_term_is_killed(tmp_path):
    result, cancel = run_shutdown(tmp_path, "bash -c 'trap \"\" TERM; exec sleep 600'")
    assert result.returncode == 124
    assert cancel.exists()


def shell_function(name):
    source = (ROOT / "ray.sub").read_text()
    assert f"\n{name}() {{" in source, f"ray.sub does not define {name}"
    start = source.index(f"\n{name}() {{") + 1
    return source[start : source.index("\n}\n", start) + 3]


def alive(pid_file):
    status = Path(f"/proc/{int(pid_file.read_text())}/status")
    return status.exists() and "\nState:\tZ" not in status.read_text()


def daemon(pid_file, scope=None, ignore_term=False):
    """A detached process, as Ray daemons and vLLM engine cores are, that records its pid."""
    marker = f"NRL_TASK_SCOPE={scope} " if scope else ""
    trap = 'trap "" TERM; ' if ignore_term else ""
    return f"{marker}setsid bash -c '{trap}echo $$ > {pid_file}; exec sleep 600' < /dev/null > /dev/null 2>&1 &\n"


def wait_for(*pid_files):
    return "".join(f"until [ -s {p} ]; do sleep 0.1; done\n" for p in pid_files)


def test_watcher_runs_on_the_hosts_of_every_node_in_the_job(tmp_path):
    srun = tmp_path / "srun"
    srun.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$SRUN_LOG"\n')
    srun.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        SRUN_LOG=str(tmp_path / "srun.log"),
        SLURM_JOB_ID="777",
        SLURM_JOB_NUM_NODES="2",
    )
    script = "".join(
        shell_function(name)
        for name in (
            "abort_stuck_image_mounts",
            "watch_image_mounts",
            "start_image_mount_watcher",
        )
    )
    subprocess.run(
        ["bash", "-c", script + "start_image_mount_watcher 777.4\nwait\n"],
        env=env,
        check=True,
        timeout=20,
    )
    call = (tmp_path / "srun.log").read_text()
    assert call.startswith("--overlap --jobid=777 --nodes=2 ")
    assert "watch_image_mounts 777 4 " in call
    assert "--environment" not in call and "--container" not in call


def test_launcher_does_not_return_before_the_watcher_has_finished(tmp_path):
    """A cancelled step ends srun at once; leaving then would kill the watcher with the job."""
    srun = tmp_path / "srun"
    srun.write_text('#!/bin/sh\nsleep 2\necho finished > "$WATCHER_DONE"\n')
    srun.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        WATCHER_DONE=str(tmp_path / "watcher.done"),
        SLURM_JOB_ID="777",
        SLURM_JOB_NUM_NODES="2",
    )
    script = "".join(
        shell_function(name)
        for name in (
            "abort_stuck_image_mounts",
            "watch_image_mounts",
            "start_image_mount_watcher",
            "wait_for_ray_cluster",
            "finish_ray_cluster",
        )
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            script
            + '(exit 7) &\npid=$!\nfinish_ray_cluster "$pid" 777.4 1\nrc=$?\n'
            + '[ -e "$WATCHER_DONE" ] && echo watcher-finished-first\nexit $rc\n',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7, result.stderr
    assert result.stdout.strip() == "watcher-finished-first"


def watch(tmp_path, pids, seconds=3):
    """Run the watcher against a fake step cgroup holding ``pids`` and one connection with waiters."""
    task = tmp_path / "cgroup/slurmstepd.scope/job_777/step_4/user/task_1"
    task.mkdir(parents=True)
    (task / "cgroup.procs").write_text("".join(f"{pid}\n" for pid in pids))
    connection = tmp_path / "connections/60"
    connection.mkdir(parents=True)
    (connection / "waiting").write_text("18\n")
    (connection / "abort").write_text("")
    script = (
        shell_function("abort_stuck_image_mounts")
        + shell_function("watch_image_mounts")
        + f"watch_image_mounts 777 4 {seconds} {tmp_path}/cgroup {tmp_path}/connections {tmp_path}/fuse\n"
    )
    result = subprocess.run(
        ["bash", "-c", script], check=True, timeout=30, capture_output=True, text=True
    )
    return (connection / "abort").read_text().strip(), result.stdout


def holding(path):
    """A process that keeps ``path`` open, as the image's mount daemon keeps /dev/fuse open."""
    path.touch()
    return subprocess.Popen(["sleep", "600"], stdin=open(path))


def test_watcher_aborts_the_mount_once_its_daemon_is_all_that_is_left(tmp_path):
    daemon_process = holding(tmp_path / "fuse")
    try:
        aborted, log = watch(tmp_path, [daemon_process.pid])
        assert aborted == "1"
        assert "released a stuck image mount" in log
    finally:
        daemon_process.kill()


def test_watcher_leaves_the_mount_alone_while_the_task_still_runs(tmp_path):
    daemon_process = holding(tmp_path / "fuse")
    worker = subprocess.Popen(["sleep", "600"])
    try:
        assert watch(tmp_path, [daemon_process.pid, worker.pid]) == ("", "")
    finally:
        daemon_process.kill()
        worker.kill()


def test_watcher_returns_as_soon_as_the_step_is_gone(tmp_path):
    script = (
        shell_function("abort_stuck_image_mounts")
        + shell_function("watch_image_mounts")
        + f"watch_image_mounts 777 4 600 {tmp_path} {tmp_path} {tmp_path}/fuse\n"
    )
    subprocess.run(["bash", "-c", script], check=True, timeout=10)


def test_only_owned_connections_with_waiters_are_aborted(tmp_path):
    for name, waiting in (("60", "272"), ("61", "0")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "waiting").write_text(waiting + "\n")
        (tmp_path / name / "abort").write_text("")
    subprocess.run(
        [
            "bash",
            "-c",
            shell_function("abort_stuck_image_mounts")
            + f"\nabort_stuck_image_mounts {tmp_path}\n",
        ],
        check=True,
        timeout=20,
    )
    assert (tmp_path / "60" / "abort").read_text().strip() == "1"
    assert (tmp_path / "61" / "abort").read_text() == ""


def test_reap_kills_the_tasks_detached_processes_and_nothing_else(tmp_path):
    ours, stubborn, foreign = (
        tmp_path / "ours",
        tmp_path / "stubborn",
        tmp_path / "foreign",
    )
    script = (
        shell_function("reap_task_processes")
        + "export NRL_TASK_SCOPE=job.0.1\n"
        + daemon(ours, "job.0.1")
        + daemon(stubborn, "job.0.1", ignore_term=True)
        + daemon(foreign, "job.0.2")
        + wait_for(ours, stubborn, foreign)
        + 'reap_task_processes "$NRL_TASK_SCOPE" 1\necho survived\n'
    )
    try:
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=40
        )
        assert result.stdout.strip() == "survived", result.stderr
        assert not alive(ours)
        assert not alive(stubborn)
        assert alive(foreign)
    finally:
        for pid_file in (ours, stubborn, foreign):
            if pid_file.exists() and alive(pid_file):
                os.kill(int(pid_file.read_text()), 9)


def run_dispatch(tmp_path, procid, node, head_rc=0, worker_rc=0):
    leftover = tmp_path / f"leftover-{procid}"
    for name, rc in (("head.sh", head_rc), ("worker.sh", worker_rc)):
        (tmp_path / name).write_text(
            daemon(leftover) + wait_for(leftover) + f"exit {rc}\n"
        )
    script = (
        shell_function("reap_task_processes")
        + shell_function("write_cluster_dispatch")
        + f"LOG_DIR={tmp_path} head_node=head _task_digits=1\n"
        + f"HEAD_COMMAND_FILE={tmp_path}/head.sh WORKER_COMMAND_FILE={tmp_path}/worker.sh\n"
        + f"CLUSTER_DISPATCH_FILE={tmp_path}/dispatch.sh\nwrite_cluster_dispatch\n"
    )
    subprocess.run(["bash", "-c", script], check=True, timeout=20)
    env = dict(
        os.environ,
        SLURM_JOB_ID="777",
        SLURM_STEP_ID="0",
        SLURM_PROCID=str(procid),
        SLURMD_NODENAME=node,
    )
    try:
        result = subprocess.run(
            ["bash", str(tmp_path / "dispatch.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result, alive(leftover)
    finally:
        if leftover.exists() and alive(leftover):
            os.kill(int(leftover.read_text()), 9)


def test_failed_head_leaves_nothing_behind_and_does_not_make_slurm_kill_the_step(
    tmp_path,
):
    result, leftover_alive = run_dispatch(tmp_path, procid=0, node="head", head_rc=3)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "RAY_HEAD_EXIT_CODE").read_text().strip() == "3"
    assert (tmp_path / "RAY_HEAD_FINISHED").exists()
    assert not leftover_alive


def test_worker_leaves_nothing_behind_and_keeps_its_status(tmp_path):
    result, leftover_alive = run_dispatch(tmp_path, procid=1, node="w1", worker_rc=5)
    assert result.returncode == 5, result.stderr
    assert not leftover_alive
