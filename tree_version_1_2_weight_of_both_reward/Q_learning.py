from typing import Tuple, Any, DefaultDict, List

import gym
import numpy as np
from collections import defaultdict
import time

from matplotlib import pyplot as plt

from Tree_env_1 import TreeEnv

MAX_EPISODE_LENGTH = 250
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 1

BINS = 20
NUM_STATES = BINS ** 4


def policy(env: gym.Env, Q: DefaultDict[Tuple[Any, int], float], state, exploration_rate: float) -> int:
    if np.random.uniform(0, 1) < exploration_rate:
        return env.action_space.sample()
    q_values = [Q[(state, action)] for action in range(env.action_space.n)]
    return np.argmax(q_values).item()


def q_learning(env, num_episodes: int, exploration_rate=1.0, exploration_rate_decay=0.9999, min_exploration_rate=0.05) -> \
Tuple[List[float], DefaultDict[Tuple[Any, int], float]]:
    Q = defaultdict(lambda: np.random.uniform(1, -1))

    rewards = []

    print(f"Performing Q-learning with {NUM_STATES:d} states")
    for episode in range(num_episodes):
        rewards.append(0)
        obs = env.reset()

        state = tuple(obs)

        for t in range(MAX_EPISODE_LENGTH):
            action = policy(env, Q, state, exploration_rate)
            # env.render()

            obs, reward, done, _ = env.step(action)

            next_state = tuple(obs)
            optimal_next_action = policy(env, Q, next_state,
                                         exploration_rate)  # need to set no exporation to get optimal action

            # TODO: Implement Q-Learning Update
            Q[(state, action)] = Q[(state, action)] + LEARNING_RATE * (
                        reward + DISCOUNT_FACTOR * Q[(next_state, optimal_next_action)] - Q[(state, action)])

            state = next_state

            rewards[-1] += reward

            if done:
                break

        exploration_rate = max(exploration_rate_decay * exploration_rate, min_exploration_rate)
        if episode % (num_episodes / 100) == 0:
            print(f"Mean Reward: {np.mean(rewards[-int(num_episodes / 100):])}")

    return rewards, Q


if __name__ == "__main__":
    env = TreeEnv()

    rewards, Q = q_learning(env, 10000)

    _, ax = plt.subplots()
    ax.step([i for i in range(1, len(rewards) + 1)], rewards, linewidth=0.5)
    ax.grid()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    plt.title('Version 3 & Q-Learning')
    plt.show()

    print(f'Mean reward: {np.mean(rewards)}')
    print(f'Standard deviation: {np.std(rewards)}')
    print(f'Max reward: {np.max(rewards)}')
    print(f'Min reward: {np.min(rewards)}')

    # simulate agent after training
    obs = env.reset()

    state = tuple(obs)
    current_total_reward=0
    for _ in range(1000):
        # env.render(current_total_reward)
        action = policy(env, Q, state, exploration_rate=0.0)
        obs, get_reward, done, _ = env.step(action)
        current_total_reward+=get_reward
        if done:
            print(f"with action from q_learning get reward {current_total_reward}")
            break

        state = tuple(obs)
        time.sleep(0.5)

    env.close()
