#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env
import argparse

from multigrid.envs import *
from multigrid.core.actions import Action
from multigrid.base import MultiGridEnv
from multigrid.wrappers import (
    SingleAgentWrapper,
)


class ManualControl:
    """
    Class to manual control an agent within a given environment.
    It allows the user to interact with the environment using keyboard input.
    """

    def __init__(self, env: Env, seed=None, agents=2) -> None:
        """
        Initialize ManualControl.

        Parameters
        ----------
        env : Env
            The environment in which to operate.
        seed : int, optional
            The seed used for random number generator.
        agents : int, optional
            The number of agents to operate in the environment. Default is 2.
        """
        self.env = env
        self.seed = seed
        self.closed = False
        self.total_episodic_rewards = 0
        self.agents = agents

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        # Event loop for the pygame display.
        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Action):
        """Perform a step in the environment with the given action.

        The step updates the environment based on the action provided.
        If the step leads to a termination or truncation, the environment is reset.

        Parameters
        ----------
        action : Action
            The action to be performed.
        """

        _, reward, terminated, truncated, _ = self.env.step(action)
        # Adjust reward, termination and truncation based on number of agents.
        reward = reward if self.agents < 2 else reward[0]
        terminated = terminated if self.agents < 2 else terminated[0]
        truncated = truncated if self.agents < 2 else truncated[0]

        self.total_episodic_rewards += reward
        print(
            f"step={self.env.step_count}, reward={reward:.2f}, total episodic reward={self.total_episodic_rewards: .2f} "
        )

        if terminated:
            print(f"terminated! total episodic reward={self.total_episodic_rewards: .2f} ")
            self.total_episodic_rewards = 0
            self.reset(self.seed)
        elif truncated:
            print(f"truncated! total episodic reward={self.total_episodic_rewards: .2f} ")
            self.total_episodic_rewards = 0
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        """Reset the environment with an optional new seed.

        Parameters
        ----------
        seed : int, optional
            The seed for the environment's random number generator.
        """
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        """
        Handle key events.

        Certain keys are mapped to specific actions in the environment.
        Escape and Backspace keys have special functionality, closing the environment
        and resetting the environment respectively.

        Parameters
        ----------
        event : pygame.event.Event
            The event to handle.
        """
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Action.left,
            "right": Action.right,
            "up": Action.forward,
            "space": Action.toggle,
            "pageup": Action.pickup,
            "pagedown": Action.drop,
            "tab": Action.pickup,
            "left shift": Action.drop,
            "enter": Action.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            if self.agents < 2:
                self.step(action)
            else:
                actions = {0: action}
                for i in range(1, self.agents):
                    actions[i] = Action.done
                self.step(actions)
        else:
            print(key)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MultiGrid-CompetativeRedBlueDoor-v0",  #  MultiGrid-LockedHallway-2Rooms-v0
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument("--tile-size", type=int, help="size at which to render tiles", default=32)
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--single-agent",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--our-agent-ids",
        nargs="+",
        type=int,
        default=[0],
        help="List of agent ids to evaluate",
    )

    args = parser.parse_args()

    # Instantiate environment with provided parameters
    env: MultiGridEnv = gym.make(
        args.env_id,
        render_mode="human",
        agents=args.agents,
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    if args.single_agent:
        print("Convert to single agent")
        env = SingleAgentWrapper(env)

    manual_control = ManualControl(env, seed=args.seed, agents=args.agents)
    manual_control.start()
