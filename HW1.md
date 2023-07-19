

---

# Homework 1: Intro to Deep RL with Single Agent Training Environments

**Due Date:** September 14, 11:30 pm

The goal of this assignment is to gain hands-on experience with the key components of Reinforcement Learning (RL) environments. You will be able to:

- Debug your environment by ensuring the following:
  - The environment has the correct reward scale.
  - The environment's termination conditions meet the learning objective.
  - Your agent utilizes the right observation and action spaces for training.
- Begin training on your local machine or Google Colab.
- Familiarize yourself with Tensorboard and how to use custom metrics.
- Understand the assignment submission process.

The starter code for this assignment can be found [here](https://classroom.github.com/classrooms/123430433-rl2rl-deeprl/assignments/week-1-intro-to-deep-rl-and-agent-training-environments).

## Setup

You have the option of running the code either on Google Colab or on your local machine.

1. **Local Setup:** If you choose to run locally, you will need to install MuJoCo and some Python packages. Refer to [INSTALLATION.md](INSTALLATION.md) for instructions.
2. **Google Colab:** The initial sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heng2j/multigrid/blob/hw1/notebooks/homework1.ipynb)

## Complete the Code

We recommend reading the files in the following order. For some files, you will need to fill in the sections labeled `TODO` or `FIXME`.

- [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py)
- [wrappers.py](multigrid/wrappers.py)

Look for sections marked with `HW1` to understand how your changes will be utilized. You might also find the following files relevant:

- [scripts/manual_control.py](scripts/manual_control.py)
- [scripts/visualize.py](scripts/visualize.py)
- [scripts/train.py](scripts/train.py)

Depending on your chosen setup, refer to [scripts/train.py](scripts/train.py) and [scripts/visualize.py](scripts/visualize.py) (if running locally), or [notebooks/homework1.ipynb](notebooks/homework1.ipynb) (if running on Colab).

## Run the Code

If you're debugging, you might want to use VSCode's debugger. If you're running on Colab, adjust the `#@params` in the `Args` class as per the command-line arguments above.

---
## Task 1 - Get to Know Your Learning Environment

Running the follwoing script to manually control your agent. The keyboard controls can be found in `key_handler()` in the `ManualControl` class

Command for Task 1:
```shell
python scripts/manual_control.py
```

***Notes:***

1. Experiment with other optional arguments to familiarize yourself with the environment.
2. The dense reward will be displayed at each time step after your input actions. The keyboard controls can be found in the `key_handler()` method in the `ManualControl` class.
3. A wrong action can lead to a penalty per step. A sparse reward will be added to the total episodic reward when you perform an associated operation, e.g., picking up the key from the floor.

**Task 1 Description:** Your agent can perform a wrong action by randomly using the pickup action at each time step. There is a penalty when the agent picks up the incorrect object. You should adjust this value in proportion to the total horizon and the ultimate goal-oriented reward. Please fix the reward scale in [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py).

---
## Task 2 - Debug Observations and Observations Space for Training

If you run the following training command, you are expected to see ValueError on mismatching observation and observation space. Make sure to handle this exception and implement the correct observation to avoid it.


Command for Task 2:
```shell
python scripts/train.py
```
***Note:*** 

You might encounter a `ValueError` for mismatching observation and observation space if you run the above command. Make sure to handle this exception and implement the correct observation to avoid it.

Your training batch size should be larger than the horizon so that you're collecting multiple rollouts when evaluating the performance of your trained policy. For example, if the horizon is 1000 and the training batch size is 5000, you'll collect approximately 5 trajectories (or more if any of them terminate early).

---

## Task 3 - Monitor and Track Agent Training with Tensorboard and Save Out Visualization from Evaluation

Monitor and track your runs using Tensorboard with the following command:
```shell
tensorboard --logdir ./ray_result
```

You can filter the plots using the following filters:

```
episode_len_mean|ray/tune/episode_reward_mean|episode_reward_min|entropy|vf|loss|kl|cpu|ram
```


You will see scalar summaries as well as videos of your trained policies in the 'Images' tab.

To visualize a specific checkpoint, use the following command:
```shell
python scripts/visualize.py --env MultiGrid-CompetativeRedBlueDoor-v0  --num-episodes 20  --load-dir ./ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v0_37eb5_00000_0_2023-07-10_11-12-43 --gif ./result.gif
```
If running on Colab, use the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to achieve the same; see the [notebook](notebooks/homework1.ipynb) for more details.

---


## Task 4 - Submit your homework on Github Classroom

You can submit your results and documentations on a Jupyter Notebook or via Google CoLab Notebook. Convert your notebook into HTML format and then push to your Github Classroom Github Page folder.

During each training, Ray Tune will generate the MLFlow artifacts to your local directory. You will need to push your MLFlow artifacts along with your RLlib checkpoints to your submission folder in your repo.