"""Validate Ray and materialize the exact Megatron worker environment."""

import os
from pathlib import Path

import ray

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.venvs import create_local_venv_on_each_node


MEGATRON_WORKER_FQN = (
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
)


def main() -> None:
    expected_nodes = int(os.environ["EXPECTED_NODES"])
    expected_gpus = int(os.environ["EXPECTED_UNITS"])
    init_ray()
    alive_nodes = [node for node in ray.nodes() if node["Alive"]]
    gpu_count = int(ray.cluster_resources().get("GPU", 0))
    assert len(alive_nodes) == expected_nodes, alive_nodes
    assert gpu_count == expected_gpus, ray.cluster_resources()
    print("ray_nodes=" + str(len(alive_nodes)))
    print("ray_gpus=" + str(gpu_count))

    if os.environ.get("CHECK_MEGATRON_VENV") == "1":
        expected_project_root = str(Path(os.environ["EXPECTED_PROJECT_ROOT"]).resolve())
        actor_command = get_actor_python_env(MEGATRON_WORKER_FQN)
        expected_directory = "--directory " + expected_project_root
        assert expected_directory in actor_command, actor_command
        print("megatron_actor_command=" + actor_command)
        worker_python = create_local_venv_on_each_node(
            actor_command,
            MEGATRON_WORKER_FQN,
        )
        assert Path(worker_python).is_file(), worker_python
        print("megatron_worker_venv=" + worker_python)
    ray.shutdown()


if __name__ == "__main__":
    main()
