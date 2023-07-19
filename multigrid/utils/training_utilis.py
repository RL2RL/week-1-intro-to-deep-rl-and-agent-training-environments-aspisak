import os
import random
from pathlib import Path
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms import AlgorithmConfig
from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls
from typing import Callable


def get_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    """
    Recursively search for checkpoints within the given directory.

    If more than one is found, returns the most recently modified checkpoint directory.

    Parameters
    ----------
    search_dir : Path or str
        The directory to search for checkpoints within
    """
    try:
        checkpoints = Path(search_dir).expanduser().glob("**/*.is_checkpoint")
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent
    except:
        pass


def get_policy_mapping_fn(checkpoint_dir: Path | str | None, num_agents: int) -> Callable:
    """
    Create policy mapping function from saved policies in checkpoint directory.
    Maps agent i to the (i % num_policies)-th policy.

    Parameters
    ----------
    checkpoint_dir : Path or str
        The checkpoint directory to load policies from
    num_agents : int
        The number of agents in the environment
    """
    try:
        policies = sorted([path for path in (checkpoint_dir / "policies").iterdir() if path.is_dir()])

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return policies[agent_id % len(policies)].name

        print("Loading policies from:", checkpoint_dir)
        for agent_id in range(num_agents):
            print("Agent ID:", agent_id, "Policy ID:", policy_mapping_fn(agent_id))

        return policy_mapping_fn

    except:
        return lambda agent_id, *args, **kwargs: f"policy_{agent_id}"


def can_use_gpu() -> bool:
    """
    Return whether or not GPU training is available.
    """
    try:
        _, tf, _ = try_import_tf()
        return tf.test.is_gpu_available()
    except:
        pass

    try:
        torch, _ = try_import_torch()
        return torch.cuda.is_available()
    except:
        pass

    return False


def policy_mapping_fn(agent_id: int, *args, **kwargs) -> str:
    """
    Map an environment agent ID to an RLlib policy ID.
    """
    return f"policy_{agent_id}"


def model_config(framework: str = "torch", lstm: bool = False, custom_model_config: dict = {}):
    """
    Return a model configuration dictionary for RLlib.
    """
    if framework == "torch":
        if lstm:
            model = TorchLSTMModel
        else:
            model = TorchModel
    else:
        if lstm:
            raise NotImplementedError
        else:
            model = TFModel

    return {
        "custom_model": model,
        "custom_model_config": custom_model_config,
        "conv_filters": [
            [16, [3, 3], 1],
            [16, [1, 1], 1],
            [32, [3, 3], 1],
            [32, [1, 1], 1],
            [64, [3, 3], 1],
            [64, [1, 1], 1],
        ],
        "fcnet_hiddens": [64, 64],
        "post_fcnet_hiddens": [],
        "lstm_cell_size": 64,
        "max_seq_len": 64,
    }


def algorithm_config(
    algo: str = "PPO",
    env: str = "MultiGrid-Empty-8x8-v0",
    env_config: dict = {},
    num_agents: int = 2,
    framework: str = "torch",
    lstm: bool = False,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = None,
    **kwargs,
) -> AlgorithmConfig:
    """
    Return the RL algorithm configuration dictionary.
    """
    env_config = {**env_config, "agents": num_agents}
    return (
        get_trainable_cls(algo)
        .get_default_config()
        .environment(env=env, env_config=env_config)
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus if can_use_gpu() else 0)
        .multi_agent(
            policies={f"policy_{i}" for i in range(num_agents)},
            policy_mapping_fn=get_policy_mapping_fn(None, num_agents),
        )
        .training(
            model=model_config(framework=framework, lstm=lstm),
            lr=(lr or NotProvided),
            vf_loss_coeff=0.5,
            entropy_coeff=0.001,
        )
        .debugging(seed=random.randint(0, int(1e6)))
    )
