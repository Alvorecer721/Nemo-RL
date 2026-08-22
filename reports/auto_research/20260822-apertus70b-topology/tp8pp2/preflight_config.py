"""Resolve and validate the committed TP8/PP2 experiment configuration."""

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.dpo import MasterConfig
from nemo_rl.utils.config import load_config


def main() -> None:
    config_path = Path(sys.argv[1]).resolve()
    raw_config = OmegaConf.to_container(load_config(config_path), resolve=True)
    config = MasterConfig(**raw_config)

    megatron = config.policy["megatron_cfg"]
    tp = megatron["tensor_model_parallel_size"]
    pp = megatron["pipeline_model_parallel_size"]
    cp = megatron["context_parallel_size"]
    world_size = config.cluster["num_nodes"] * config.cluster["gpus_per_node"]
    assert (tp, pp, cp, world_size) == (8, 2, 1, 16)
    assert world_size // (tp * pp * cp) == 1
    assert config.policy["train_global_batch_size"] == 32
    assert config.policy["train_micro_batch_size"] == 1
    assert config.policy["max_total_sequence_length"] == 2048
    assert config.data["max_input_seq_length"] == 2048
    assert config.dpo.max_num_steps == 10
    assert megatron["optimizer"]["use_distributed_optimizer"] is True
    assert megatron["optimizer"]["optimizer_cpu_offload"] is False

    model_config_path = Path(config.policy["model_name"]) / "config.json"
    model_config = json.loads(model_config_path.read_text())
    divisibility = {
        "hidden_size": (model_config["hidden_size"], tp),
        "num_attention_heads": (model_config["num_attention_heads"], tp),
        "num_key_value_heads": (model_config["num_key_value_heads"], tp),
        "intermediate_size": (model_config["intermediate_size"], tp),
        "vocab_size": (model_config["vocab_size"], tp),
        "num_hidden_layers": (model_config["num_hidden_layers"], pp),
        "sequence_length": (config.policy["max_total_sequence_length"], tp),
    }
    for name, (value, divisor) in divisibility.items():
        assert value % divisor == 0, (name, value, divisor)

    print("config_path=" + str(config_path))
    print(f"tp={tp} pp={pp} dp=1 cp={cp} world_size={world_size}")
    print("distributed_optimizer=true optimizer_cpu_offload=false")
    print("model_divisibility=OK")


if __name__ == "__main__":
    main()
