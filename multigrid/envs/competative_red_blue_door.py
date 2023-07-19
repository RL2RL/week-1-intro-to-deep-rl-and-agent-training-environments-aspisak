from __future__ import annotations

from multigrid.base import MultiGridEnv
from multigrid.core import Action, Grid, MissionSpace
from multigrid.core.constants import Color
from multigrid.core.world_object import Door, Key, Ball
from multigrid.core.agent import Agent



class CompetativeRedBlueDoorEnvV2(MultiGridEnv):
    """
    .. image:: https://i.imgur.com/usbavAh.gif
        :width: 400

    ***********
    Description
    ***********

    This environment is a room with one red and one blue door facing
    opposite directions. Agents must open the red door and then open the blue door,
    in that order.

    The standard setting is cooperative, where all agents receive the reward
    upon completion of the task.

    *************
    Mission Space
    *************

    "open the red door then the blue door"

    *****************
    Observation Space
    *****************

    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.

    Each agent observation is a dictionary with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment,
        where the object at each grid cell is encoded as a vector:
        (:class:`.Type`, :class:`.Color`, :class:`.State`)
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : Mission
        Task string corresponding to the current environment configuration

    ************
    Action Space
    ************

    The multi-agent action space is a Dict mapping from agent index to
    corresponding agent action space.

    Agent actions are discrete integer values, given by:

    +-----+--------------+-----------------------------+
    | Num | Name         | Action                      |
    +=====+==============+=============================+
    | 0   | left         | Turn left                   |
    +-----+--------------+-----------------------------+
    | 1   | right        | Turn right                  |
    +-----+--------------+-----------------------------+
    | 2   | forward      | Move forward                |
    +-----+--------------+-----------------------------+
    | 3   | pickup       | Pick up an object           |
    +-----+--------------+-----------------------------+
    | 4   | drop         | Drop an object              |
    +-----+--------------+-----------------------------+
    | 5   | toggle       | Toggle / activate an object |
    +-----+--------------+-----------------------------+
    | 6   | done         | Done completing task        |
    +-----+--------------+-----------------------------+

    *******
    Rewards
    *******

    A reward of ``1 - 0.9 * (step_count / max_steps)`` is given for success,
    and ``0`` for failure.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * Any agent opens the blue door while the red door is open (success)
    * Any agent opens the blue door while the red door is not open (failure)
    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-RedBlueDoors-6x6-v0``
    * ``MultiGrid-RedBlueDoors-8x8-v0``
    """

    def __init__(
        self,
        size: int = 8,
        max_steps: int | None = None,
        joint_reward: bool = True,
        success_termination_mode: str = "any",
        failure_termination_mode: str = "any",
        **kwargs,
    ):
        """
        Parameters
        ----------
        size : int, default=8
            Width and height of the grid
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the reward when the task is completed
        success_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent fails the task
            or after all agents fail the task
        failure_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent fails the task
            or after all agents fail the task
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        self.size = size
        mission_space = MissionSpace.from_string("open the door that match your agent's color")
        super().__init__(
            mission_space=mission_space,
            width=(2 * size),
            height=size,
            max_steps=max_steps or (20 * size**2),
            joint_reward=joint_reward,
            success_termination_mode=success_termination_mode,
            failure_termination_mode=failure_termination_mode,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        LEFT, HALLWAY, RIGHT = range(3)  # columns
        color_sequence = ["red", "blue"]

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        room_top = (width // 4, 0)
        room_size = (width // 2, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(*room_top, *room_size)

        # Add a red door at the bottom of the left wall
        red_door_x = room_top[0]
        red_door_y = height - 2  # self._rand_int(1, height - 1)
        self.red_door = Door(Color.red, is_locked=True)
        self.grid.set(red_door_x, red_door_y, self.red_door)

        # Add a blue door at the top of the right wall
        blue_door_x = room_top[0] + room_size[0] - 1
        blue_door_y = 1  # self._rand_int(1, height - 1)
        self.blue_door = Door(Color.blue, is_locked=True)
        self.grid.set(blue_door_x, blue_door_y, self.blue_door)

        # # Block red door with a ball
        # self.grid.set(red_door_x + 1, red_door_y, Ball(color=self._rand_color()))

        # Place keys in hallway
        for key_color in color_sequence:
            self.place_obj(Key(color=key_color), top=room_top, size=room_size)

        # Place agents in the top-left corner
        # TODO - update to encapsulate muti-agent positioning
        for agent in self.agents:
            if agent.color == "red":
                self.place_agent(agent, top=(red_door_x + 1, red_door_y), size=room_size)
                agent.state.pos = (blue_door_x - 1, blue_door_y)
                agent.state.dir = 2

            else:
                self.place_agent(agent, top=(blue_door_x - 1, blue_door_y), size=room_size)
                agent.state.pos = (red_door_x + 1, red_door_y)
                agent.state.dir = 0

    def step(self, actions):
        """
        :meta private:
        """
        obs, reward, terminated, truncated, info = super().step(actions)

        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            if action == Action.toggle:
                fwd_obj = self.grid.get(*agent.front_pos) # TODO - get opponent agent
                
                for other_agent in self.agents:
                    if (agent.front_pos == other_agent.pos) and other_agent.color != agent.color:
                        fwd_obj = other_agent

                if fwd_obj == self.red_door and self.red_door.is_open:
                    if self.red_door.is_open:
                        # TODO - Set Done Conditions
                        self.on_success(agent, reward, terminated)
                    # else:
                    #     self.on_failure(agent, reward, terminated)
                    #     self.blue_door.is_open = False  # close the door again
                elif isinstance(fwd_obj, Agent):
                    # TODO - Make this clean
                    fwd_obj.terminated = True
                    self.grid.set(*fwd_obj.pos, None)
                    fwd_obj.pos = (999,999)

            
            # TODO - Add Sparse rewards
            elif action == Action.pickup:
                if agent.carrying and (agent.carrying.type == "key") and (agent.carrying.is_available == True):
                    # FIXME - make me elegant 
                    agent.carrying.is_available = False
                    agent.carrying.is_pickedup = True
                    reward[agent_id] += 0.5
                else:
                    # If we are grabbing bad stuff
                    reward[agent_id] -= 0.001  # OG 0.2
            


            # TODO - Add Dense Rewards to encourage agent to learn faster



        return obs, reward, terminated, truncated, info




class CompetativeRedBlueDoorEnv(MultiGridEnv):
    """
    .. image:: https://i.imgur.com/usbavAh.gif
        :width: 400

    ***********
    Description
    ***********

    This environment is a room with one red and one blue door facing
    opposite directions. Agents must open the red door and then open the blue door,
    in that order.

    The standard setting is cooperative, where all agents receive the reward
    upon completion of the task.

    *************
    Mission Space
    *************

    "open the red door then the blue door"

    *****************
    Observation Space
    *****************

    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.

    Each agent observation is a dictionary with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment,
        where the object at each grid cell is encoded as a vector:
        (:class:`.Type`, :class:`.Color`, :class:`.State`)
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : Mission
        Task string corresponding to the current environment configuration

    ************
    Action Space
    ************

    The multi-agent action space is a Dict mapping from agent index to
    corresponding agent action space.

    Agent actions are discrete integer values, given by:

    +-----+--------------+-----------------------------+
    | Num | Name         | Action                      |
    +=====+==============+=============================+
    | 0   | left         | Turn left                   |
    +-----+--------------+-----------------------------+
    | 1   | right        | Turn right                  |
    +-----+--------------+-----------------------------+
    | 2   | forward      | Move forward                |
    +-----+--------------+-----------------------------+
    | 3   | pickup       | Pick up an object           |
    +-----+--------------+-----------------------------+
    | 4   | drop         | Drop an object              |
    +-----+--------------+-----------------------------+
    | 5   | toggle       | Toggle / activate an object |
    +-----+--------------+-----------------------------+
    | 6   | done         | Done completing task        |
    +-----+--------------+-----------------------------+

    *******
    Rewards
    *******

    A reward of ``1 - 0.9 * (step_count / max_steps)`` is given for success,
    and ``0`` for failure.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * Any agent opens the blue door while the red door is open (success)
    * Any agent opens the blue door while the red door is not open (failure)
    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-RedBlueDoors-6x6-v0``
    * ``MultiGrid-RedBlueDoors-8x8-v0``
    """

    def __init__(
        self,
        size: int = 8,
        max_steps: int | None = None,
        joint_reward: bool = True,
        success_termination_mode: str = "any",
        failure_termination_mode: str = "any",
        **kwargs,
    ):
        """
        Parameters
        ----------
        size : int, default=8
            Width and height of the grid
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the reward when the task is completed
        success_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent fails the task
            or after all agents fail the task
        failure_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent fails the task
            or after all agents fail the task
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        self.size = size
        mission_space = MissionSpace.from_string("open the door that match your agent's color")
        super().__init__(
            mission_space=mission_space,
            width=(2 * size),
            height=size,
            max_steps=max_steps or (20 * size**2),
            joint_reward=joint_reward,
            success_termination_mode=success_termination_mode,
            failure_termination_mode=failure_termination_mode,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        LEFT, HALLWAY, RIGHT = range(3)  # columns
        color_sequence = ["red"] # ["red", "blue"]

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        room_top = (width // 4, 0)
        room_size = (width // 2, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(*room_top, *room_size)

        # Add a red door at the bottom of the left wall
        red_door_x = room_top[0]
        red_door_y = height - 2  # self._rand_int(1, height - 1)
        self.red_door = Door(Color.red, is_locked=True)
        self.grid.set(red_door_x, red_door_y, self.red_door)

        # Add a blue door at the top of the right wall
        blue_door_x = room_top[0] + room_size[0] - 1
        blue_door_y = 1  # self._rand_int(1, height - 1)
        self.blue_door = Door(Color.blue, is_locked=True)
        self.grid.set(blue_door_x, blue_door_y, self.blue_door)

        # # Block red door with a ball
        # self.grid.set(red_door_x + 1, red_door_y, Ball(color=self._rand_color()))

        # Place keys in hallway
        for key_color in color_sequence:
            self.place_obj(Key(color=key_color), top=room_top, size=room_size)

        # Place agents in the top-left corner
        # TODO - update to encapsulate muti-agent positioning
        for agent in self.agents:
            if agent.color == "red":
                self.place_agent(agent, top=(red_door_x + 1, red_door_y), size=room_size)
                agent.state.pos = (blue_door_x - 1, blue_door_y)
                agent.state.dir = 2

            else:
                self.place_agent(agent, top=(blue_door_x - 1, blue_door_y), size=room_size)
                agent.state.pos = (red_door_x + 1, red_door_y)
                agent.state.dir = 0

    def step(self, actions):
        """
        :meta private:
        """
        obs, reward, terminated, truncated, info = super().step(actions)

        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            if action == Action.toggle:
                fwd_obj = self.grid.get(*agent.front_pos)
                if fwd_obj == self.red_door and self.red_door.is_open:
                    if self.red_door.is_open:
                        # TODO - Set Done Conditions
                        self.on_success(agent, reward, terminated)
                    # else:
                    #     self.on_failure(agent, reward, terminated)
                    #     self.blue_door.is_open = False  # close the door again
            
            # TODO - Add Sparse rewards
            elif action == Action.pickup:
                if agent.carrying and (agent.carrying.type == "key") and (agent.carrying.is_available == True):
                    # FIXME - make me elegant 
                    agent.carrying.is_available = False
                    agent.carrying.is_pickedup = True
                    reward[agent_id] += 0.5
                else:
                    # If we are grabbing bad stuff
                    reward[agent_id] -= 0.001  # OG 0.2

            # TODO - Add Dense Rewards to encourage agent to learn faster



        return obs, reward, terminated, truncated, info





