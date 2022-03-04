from typing import Tuple, Any, DefaultDict, List
# import dill as pickle

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

# CART_POSITION = np.linspace(-4.8, 4.8, BINS)
# CART_VELOCITY = np.linspace(-1, 1, BINS)
# POLE_ANGLE = np.linspace(-0.418, 0.418, BINS)
# POLE_ANGULAR_VELOCITY = np.linspace(-3, 3, BINS)



def policy(env: gym.Env, Q: DefaultDict[Tuple[Any, int], float], state, exploration_rate: float) -> int:
    if np.random.uniform(0, 1) < exploration_rate:
        return env.action_space.sample()
    q_values = [Q[(state, action)] for action in range(env.action_space.n)]
    return np.argmax(q_values).item()


def q_learning(env, num_episodes: int, exploration_rate=0.9, exploration_rate_decay=0.9999, min_exploration_rate=0.05) -> \
Tuple[List[float], DefaultDict[Tuple[Any, int], float]]:
    Q = defaultdict(lambda: np.random.uniform(1, -1))

    rewards = []
    print(f"Performing Q-learning with {NUM_STATES:d} states")
    for episode in range(num_episodes):
        rewards.append(0)
        obs = env.reset()

        state = tuple(map(tuple, obs))

        for t in range(MAX_EPISODE_LENGTH):
            action = policy(env, Q, state, exploration_rate)
            # env.render()

            obs, reward, done, _ = env.step(action)

            next_state = tuple(map(tuple, obs))
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

def evaluation(env, Q):
    obs = env.reset()

    state = tuple(obs)
    current_total_reward = 0
    for _ in range(1000):
        # env.render(current_total_reward)
        action = policy(env, Q, state, exploration_rate=0.0)
        obs, get_reward, done, _ = env.step(action)
        current_total_reward += get_reward
        if done:
            print(f"with action from q_learning get reward {current_total_reward}")
            break
        print(f'state: {state}, action: {action}')

        state = tuple(obs)
        # time.sleep(2)

if __name__ == "__main__":
    # env = gym.make('CartPole-v0')
    env = TreeEnv()

    time_start = time.time()
    rewards, Q = q_learning(env, 100000)
    time_end = time.time()
    print("time cost", time_end-time_start,'s')


    # save model
    # with open('Q_learning_model.pkl', 'wb') as pkl_handle:
    #     pickle.dump(Q, pkl_handle)

    _, ax = plt.subplots()
    ax.step([i for i in range(1, len(rewards) + 1)], rewards, linewidth=0.5)
    ax.grid()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    plt.title('Version 5 & Q-Learning')
    plt.show()

    print(f'Mean reward: {np.mean(rewards)}')
    print(f'Standard deviation: {np.std(rewards)}')
    print(f'Max reward: {np.max(rewards)}')
    print(f'Min reward: {np.min(rewards)}')

    # read model from file
    # with open('Q_learning_model.pkl', 'rb') as pkl_handle:
    #     Q = pickle.load(pkl_handle)

    # evaluation
    evaluation(env, Q)

    env.close()
