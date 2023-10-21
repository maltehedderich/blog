---
title: 'Tabular Q-Learning'
date: 2023-10-21
categories:
  - Reinforcement Learning
  - Q-Learning
tags:
  - Advanced
draft: true
---

## Introduction

This blog post delves into the topic of Tabular Q-Learning, a specific type of Q-Learning. We will explore other variants, such as Deep Q-Learning, in subsequent posts. Q-Learning is used in various applications like game playing, robot navigation, in economics and trade, and many more. It's particularly useful when the problem model is not known i.e., when the outcomes for actions are not predictable.

Q-Learning is a `reinforcement learning` algorithm, to identify the best action-selection policy using a Q-function. Reinforcement learning is a subset of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some type of reward. Q-Learning is a `model-free` algorithm, meaning it doesn't need a model of the environment to learn.

The Q-function measures the `quality` of an action in a specific state. This quality is calculated by adding the immediate reward to the discounted future reward. The discounted future reward is the highest possible discounted future reward achievable from the next state onwards.

By continuously updating the Q-function, the algorithm enables us to create a table of Q-values for each state-action pair. This table can then guide us to the optimal action-selection policy.

## Prerequisites

To follow along with this blog post, you should have a basic understanding of Reinforcement Learning. If you are new to Reinforcement Learning, I would recommend you to check out this [blog post](https://towardsdatascience.com/reinforcement-learning-101-e24b50e1d292).

Furthermore, if you want to follow along with the code, you should have a basic understanding of Python. We will use the following libraries in this blog post:

- [Gym](https://gym.openai.com/) - OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms.
- [NumPy](https://numpy.org/) - NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## Understanding Q-Learning

### Algorithm

**1. Initialize the Q-table with zeros.**
To initialize the Q-table, we need to know the number of states and actions. Let's consider a simple game of tic-tac-toe.

In this game, the states would be the different possible configurations of the tic-tac-toe board. Since the board is a 3x3 grid and each cell can be in one of three states (empty, X, or O), there are 3^9 = 19,683 possible states.

The actions would be the different possible moves a player can make. In any given state, a player can place their mark in any empty cell. So, the number of actions in any given state can range from 1 (if there's only one empty cell left) to 9 (if the board is empty).

**2. Explore the environment by taking a random action.**
In the beginning, we don't know which action is the best action to take in a given state. So, we take a random action and observe the reward and the next state. We then update the Q-table using the Bellman Equation.

We have to balance `exploration` (taking random actions) and `exploitation` (taking the best action). We do this by using an `exploration rate`. The exploration rate is the probability that the agent will explore the environment by taking a random action. The exploration rate is initialized to 1, meaning that the agent will always explore the environment by taking a random action. The exploration rate is then decayed over time. This means that the agent will explore the environment less and less as it learns more about the environment.

**3. Update the Q-table using the Bellman Equation.**
The Bellman Equation is the core of Q-Learning. It is a recursive equation that calculates the Q-value for a state-action pair. The equation is as follows:

$$Q(S,A) = Q(S,A) + \alpha * [R(S,A) + \gamma * \max Q(S',a) - Q(S,A)]$$

where,

- $Q(S,A)$ is the Q-value for the state-action pair. It is the quality of the action in the given state.
- $\alpha$ is the learning rate. It determines how much the newly acquired information overrides the existing information. This is a value between 0 and 1. A value of 0 means that the Q-values are never updated, while a value of 1 means that the Q-values are instantly updated.
- $R(S,A)$ is the reward for the state-action pair. It is the immediate reward for taking the action in the given state.
- $\gamma$ is the discount factor. It determines how much importance we give to future rewards. This is a value between 0 and 1. A value of 0 means that we only consider the immediate reward, while a value of 1 means that we consider future rewards with equal importance to immediate rewards.
- $S'$ is the next state. It is the state we transition to after taking the action in the given state.

**4. Repeat steps 2 and 3 until the Q-table converges.**
The Q-table converges when the Q-values stop changing. This means that the Q-values have converged to the optimal Q-values. The optimal Q-values are the Q-values that give us the optimal action-selection policy. The optimal action-selection policy is the policy that gives us the maximum reward.

## Implementation

We will implement Q-Learning to play the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) game. The Frozen Lake game is a grid-world game where the goal is to reach the goal state without falling into a hole. The game is played on a 4x4 grid-world. The grid-world has four types of cells:

- S - Start state
- F - Frozen state
- H - Hole state
- G - Goal state

In each state, the agent can take one of four `actions`:

- Left
- Right
- Up
- Down

The agent receives a `reward` of 1 for reaching the goal state and a reward of 0 for all other states.

The amount of states and actions in this game is small enough to be able to use a tabular approach.
Each field in the grid-world is a state. This means for a 4x4 grid-world, there are 16 states.

We begin by importing the required libraries.

```python
import gymnasium as gym
import numpy as np
import random
```

Next, we create the environment.

```python
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
random.seed(0)
```

We set the `is_slippery` parameter to `False` to make the environment deterministic. This means that the agent will always move in the direction it intends to move.

Next, we initialize the Q-table with zeros.

```python
# Define the state and action space sizes
state_space_size = env.observation_space.n # 16
action_space_size = env.action_space.n # 4

# Initialize the Q-table with zeros
q_table = np.zeros((state_space_size, action_space_size)) # 16x4
```

Each row in the Q-table represents a state and each column represents an action. The Q-value for a state-action pair is stored in the corresponding cell.

Next, we define the hyperparameters.

```python
# Hyperparameters
num_episodes = 10000 # Total number of episodes to train the agent
max_steps_per_episode = 100 # Max steps per episode
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1 # Initial exploration rate
max_exploration_rate = 1
min_exploration_rate = 0.01 # Ensures that the agent never stops exploring entirely
exploration_decay_rate = 0.001 # Exponential decay rate for the exploration rate
```

Next, we define the Q-Learning algorithm.

```python
# Q-Learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_threshold = random.uniform(0, 1)
        # If exploration_threshold > exploration_rate, then exploitation
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, _, _ = env.step(action)
        # Update Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state

        if done == True:
            break

    # Update the exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
```

We begin by resetting the environment and initializing the `done` variable to `False`. The `done` variable is used to determine when the episode is over. The episode is over when the agent reaches the goal state or falls into a hole.

Next, we loop through the steps in the episode. In each step, we determine whether to explore or exploit. If the exploration threshold is greater than the exploration rate, then we exploit. This means that we take the action with the highest Q-value for the given state. Otherwise, we explore. This means that we take a random action.

Next, we take the action and observe the reward and the next state. We then update the Q-table using the Bellman Equation.

Finally, we update the exploration rate using an exponential decay function. This ensures that the agent explores less and less as it learns more about the environment.

## Evolution of the Q-Table

Let's take a look at the evolution of the Q-table over time. Remember that each row in the Q-table represents a state and each column represents an action. The Q-value for a state-action pair is stored in the corresponding cell. In our frozen lake game, there are 16 states and 4 actions.

This means that the Q-table has 16 rows and 4 columns. The first row represents the top left cell in the grid-world. The first column represents the action `Left`. The second column represents the action `Down`. The third column represents the action `Right`. The fourth column represents the action `Up`. The states are numbered from 0 to 15, starting from the top left cell and moving left to right and top to bottom.

After 100 episodes, the Q-table looks like this:

| State | Left | Down | Right | Up  |
| ----- | ---- | ---- | ----- | --- |
| 0     | 0.0  | 0.0  | 0.0   | 0.0 |
| ...   | ...  | ...  | ...   | ... |
| 13    | 0.0  | 0.0  | 0.0   | 0.0 |
| 14    | 0.0  | 0.0  | 0.1   | 0.0 |
| 15    | 0.0  | 0.0  | 0.0   | 0.0 |

There is only one non-zero value in the Q-table. This is the Q-value for the state-action pair (14, Right). This means that the agent has learned that the best action to take in state 14 is to move right. This makes sense because state 14 is the goal state.

As our reward for reaching the goal state is 1 and our learning rate 0.1 we can conclude that the agent only reached the goal state once in the first 100 episodes.

After 1000 episodes, the Q-table looks like this:

| State | Left | Down | Right | Up   |
| ----- | ---- | ---- | ----- | ---- |
| 0     | 0.94 | 0.95 | 0.93  | 0.94 |
| 1     | 0.94 | 0.00 | 0.63  | 0.76 |
| 2     | 0.31 | 0.89 | 0.02  | 0.41 |
| 3     | 0.12 | 0.00 | 0.00  | 0.02 |
| 4     | 0.95 | 0.96 | 0.00  | 0.94 |
| 5     | 0.00 | 0.00 | 0.00  | 0.00 |
| 6     | 0.00 | 0.98 | 0.00  | 0.48 |
| 7     | 0.00 | 0.00 | 0.00  | 0.00 |
| 8     | 0.96 | 0.00 | 0.97  | 0.95 |
| 9     | 0.96 | 0.98 | 0.98  | 0.00 |
| 10    | 0.97 | 0.99 | 0.00  | 0.96 |
| 11    | 0.00 | 0.00 | 0.00  | 0.00 |
| 12    | 0.00 | 0.00 | 0.00  | 0.00 |
| 13    | 0.00 | 0.83 | 0.99  | 0.83 |
| 14    | 0.97 | 0.98 | 1.00  | 0.98 |
| 15    | 0.00 | 0.00 | 0.00  | 0.00 |

Here we can make a few observations:

- The Q-values for the states 5, 7, 11, 12, and 15 are all zero. This makes sense because these states are holes or the goal. The update with the Q-function happens after each step but holes and the goal end the episode. This means that the agent never learns anything about these states.

After 10000 episodes, the Q-table looks like this:

| State | Left | Down | Right | Up   |
| ----- | ---- | ---- | ----- | ---- |
| 0     | 0.94 | 0.95 | 0.93  | 0.94 |
| 1     | 0.94 | 0.00 | 0.80  | 0.88 |
| 2     | 0.91 | 0.20 | 0.01  | 0.18 |
| 3     | 0.11 | 0.00 | 0.00  | 0.00 |
| 4     | 0.95 | 0.96 | 0.00  | 0.94 |
| 5     | 0.00 | 0.00 | 0.00  | 0.00 |
| 6     | 0.00 | 0.86 | 0.00  | 0.26 |
| 7     | 0.00 | 0.00 | 0.00  | 0.00 |
| 8     | 0.96 | 0.00 | 0.97  | 0.95 |
| 9     | 0.96 | 0.98 | 0.98  | 0.00 |
| 10    | 0.94 | 0.99 | 0.00  | 0.67 |
| 11    | 0.00 | 0.00 | 0.00  | 0.00 |
| 12    | 0.00 | 0.00 | 0.00  | 0.00 |
| 13    | 0.00 | 0.98 | 0.99  | 0.97 |
| 14    | 0.98 | 0.99 | 1.00  | 0.98 |
| 15    | 0.00 | 0.00 | 0.00  | 0.00 |

Most of the Q-values converged already after 1000 episodes. The Q-values for the states 2, 6, 10, and 13 took longer to converge.

## Conclusion

In this blog post, we explored the topic of Tabular Q-Learning. We learned about the Bellman Equation, which is the core of Q-Learning. We then implemented Q-Learning to play the Frozen Lake game. The tabular approach is only feasible for small environments. In the next blog post, we will explore Deep Q-Learning, which is a variant of Q-Learning that uses a neural network to approximate the Q-function. This allows us to use Q-Learning in large environments.

The code for this blog post can be found [here](https://github.com/maltehedderich/blog/blob/main/notebooks/tabular_q_learning.ipynb).
