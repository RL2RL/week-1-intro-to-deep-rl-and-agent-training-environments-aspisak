from __future__ import annotations

import argparse
import json
import random
import numpy as np
import git
import ray

from multigrid.utils.training_utilis import (
    algorithm_config,
    get_checkpoint_dir,
)

from pprint import pprint
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.tune import CLIReporter
from ray.air.integrations.mlflow import MLflowLoggerCallback

import pathlib

SCRIPT_PATH = str(pathlib.Path(__file__).parent.absolute().parent.absolute())


# Set up Ray CLIReporter
# Limit the number of rows
reporter = CLIReporter(max_progress_rows=10)


# Configurable Training Function
def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    user_name: str,
    checkpoint_freq: int = 20,
    load_dir: str | None = None,
    local_mode: bool = False,
    experiment_name: str = "testing_experiment",
    mlflow_tracking_uri: str = "submission/mlflow/",
):
    """
    Train an RLlib algorithm.

    Parameters
    ----------
    algo : str
        Name of the RLlib-registered algorithm to use.
    config : AlgorithmConfig
        Algorithm-specific configuration parameters.
    stop_conditions : dict
        Conditions to stop the training loop.
    save_dir : str
        Directory to save training checkpoints and results.
    user_name : str
        Experimenter's name.
    checkpoint_freq : int, optional
        Frequency of saving checkpoints, by default 20.
    load_dir : str, optional
        Directory to load pre-trained models from, by default None.
    local_mode : bool, optional
        Set to True to run Ray in local mode for debugging, by default False.
    experiment_name : str, optional
        Name of the experiment, by default "testing_experiment".
    mlflow_tracking_uri : str, optional
        Directory to save MLFlow metrics and artifacts, by default "submission/mlflow".
    """

    # Initialize Ray.
    ray.init(num_cpus=(config.num_rollout_workers + 1), local_mode=local_mode)

    # Execute training
    tune.run(
        algo,
        stop=stop_conditions,
        config=config,
        local_dir=save_dir,
        verbose=1,
        restore=get_checkpoint_dir(load_dir),
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        progress_reporter=reporter,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=experiment_name,
                tags={
                    "user_name": user_name,
                    "git_commit_hash": git.Repo(SCRIPT_PATH).head.commit,
                },
                save_artifact=True,
            )
        ],
    )

    # Shutdown Ray after training is complete
    ray.shutdown()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        help="The name of the RLlib-registered algorithm to use.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["torch", "tf", "tf2"],
        default="torch",
        help="Deep learning framework to use.",
    )
    parser.add_argument("--lstm", action="store_true", help="Use LSTM model.")
    parser.add_argument(
        "--env",
        type=str,
        default="MultiGrid-CompetativeRedBlueDoor-v0",
        help="MultiGrid environment to use.",
    )
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')",
    )
    parser.add_argument("--num-agents", type=int, default=1, help="Number of agents in environment.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the random seed of each worker. This makes experiments reproducible",
    )
    parser.add_argument("--num-workers", type=int, default=6, help="Number of rollout workers.")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to train on.")
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1e7,
        help="Total number of timesteps to train.",
    )
    parser.add_argument("--lr", type=float, help="Learning rate for training.")
    parser.add_argument(
        "--load-dir",
        type=str,
        help="Checkpoint directory for loading pre-trained policies.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="submission/ray_results/",
        help="Directory for saving checkpoints, results, and trained policies.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20,
        help="The frequency for saving Checkpoints in training iterations.",
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default="<Your Namet>",
        help="Experinemnter Name.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="testing_experiment",
        help="Experinemnt Name.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="submission/mlflow/",
        help="Directory for saving mlflow metrics and artifacts",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="<my_experinemnt>",
        help="The name of your experinemnt",
    )
    parser.add_argument(
        "--local-mode",
        type=bool,
        default=False,
        help="Boolean value to set to use local mode for debugging",
    )
    parser.add_argument(
        "--our-agent-ids",
        nargs="+",
        type=int,
        default=[0],
        help="List of agent ids to train",
    )
    parser.add_argument(
        "--policies-to-train",
        nargs="+",
        type=str,
        default=["policy_0"],
        help="List of agent ids to train",
    )

    args = parser.parse_args()
    config = algorithm_config(**vars(args))
    config.seed = args.seed  # NOTE You can use tune.randint(0, 10000) if needed
    stop_conditions = {"timesteps_total": args.num_timesteps}

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.framework == "torch":
        import torch

        torch.manual_seed(args.seed)

    print()
    print(f"Running with following CLI options: {args}")
    print("\n", "-" * 64, "\n", "Training with following configuration:", "\n", "-" * 64)
    print()
    pprint(config.to_dict())

    # Execute training
    train(
        algo=args.algo,
        config=config,
        stop_conditions=stop_conditions,
        save_dir=args.save_dir,
        user_name=args.user_name,
        checkpoint_freq=args.checkpoint_freq,
        load_dir=args.load_dir,
        local_mode=args.local_mode,
        experiment_name=args.experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
