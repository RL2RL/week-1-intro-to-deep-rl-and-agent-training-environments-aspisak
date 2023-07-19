import argparse
import json
import numpy as np

from ray.rllib.algorithms import Algorithm
from multigrid.utils.training_utilis import (
    algorithm_config,
    get_checkpoint_dir,
    policy_mapping_fn,
)


def visualize(algorithm: Algorithm, num_episodes: int = 100) -> list[np.ndarray]:
    """
    Visualize trajectories from trained agents.

    Parameters
    ----------
    algorithm : Algorithm
        The algorithm instance, from which the policy is derived.
    num_episodes : int, optional
        The number of episodes to visualize, by default 100.

    Returns
    -------
    list[np.ndarray]
        A list of frames depicting the agents' trajectories.
    """
    # An empty list to store frames
    frames = []

    # Create an environment instance using the environment creator from the algorithm
    env = algorithm.env_creator(algorithm.config.env_config)

    # Iterate over the defined number of episodes
    for episode in range(num_episodes):
        print("\n", "-" * 32, "\n", "Episode", episode, "\n", "-" * 32)

        # Initialize a dictionary to store episode rewards for each agent
        episode_rewards = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        # Initialize termination and truncation flags for the episode
        terminations, truncations = {"__all__": False}, {"__all__": False}
        # Reset the environment to get initial observations and info
        observations, infos = env.reset()
        # Initialize the states for each agent
        states = {
            agent_id: algorithm.get_policy(policy_mapping_fn(agent_id)).get_initial_state()
            for agent_id in env.get_agent_ids()
        }

        # Continue the episode until a termination or truncation condition is met
        while not terminations["__all__"] and not truncations["__all__"]:
            # Append the current environment frame to the list of frames
            frames.append(env.get_frame())

            actions = {}
            for agent_id in env.get_agent_ids():
                # Single-agent
                actions[agent_id] = algorithm.compute_single_action(
                    observations[agent_id],
                    states[agent_id],
                    policy_id=policy_mapping_fn(agent_id),
                )

            # Perform a step in the environment using the actions
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in rewards:
                episode_rewards[agent_id] += rewards[agent_id]

        # Append the final frame of the episode
        frames.append(env.get_frame())
        print("Rewards:", episode_rewards)

    env.close()
    return frames


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
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to visualize.")
    parser.add_argument(
        "--load-dir",
        type=str,
        help="Checkpoint directory for loading pre-trained policies.",
    )
    parser.add_argument("--gif", type=str, help="Store output as GIF at given path.")
    parser.add_argument(
        "--our-agent-ids",
        nargs="+",
        type=int,
        default=[0],
        help="List of agent ids to evaluate",
    )

    args = parser.parse_args()
    args.env_config.update(render_mode="human")
    config = algorithm_config(
        **vars(args),
        num_workers=0,
        num_gpus=0,
    )
    # Build the algorithm
    algorithm = config.build()
    # Get the checkpoint directory
    checkpoint = get_checkpoint_dir(args.load_dir)
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        algorithm.restore(checkpoint)

    # Visualize the trajectories and get the frames
    frames = visualize(algorithm, num_episodes=args.num_episodes)
    if args.gif:
        import imageio

        filename = args.gif if args.gif.endswith(".gif") else f"{args.gif}.gif"
        print(f"Saving GIF to {filename}")
        # Write the frames to a GIF file
        imageio.mimsave(filename, frames)
