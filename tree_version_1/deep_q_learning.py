import random
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from Tree_env_1 import TreeEnv

MAX_EPISODE_LENGTH = 1000
DISCOUNT_FACTOR = 1.0
BATCH_SIZE = 64

"""Implement DQN"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

queue_len = 5
replay_buffer = deque(maxlen=queue_len)  # buffer for replaying

class Q_Net(nn.Module):
    def __init__(self, env):
        super(Q_Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(np.prod(env.observation_space.shape), 100),
            nn.Linear(env.observation_space.n, 100),
            nn.ReLU(),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            nn.Linear(100, env.action_space.n)
        )
    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

def make_Q(env) -> nn.Module:
    """
    Creates Q-Function from env.

    Use `env.observation_space.shape` to get the shape of the input data.
    Use `env.action_space.n` to get number of possible actions for this environment.
    """
    # TODO: Extend Q to make it deeper
    # Q = nn.Linear(np.prod(env.observation_space.shape),
    #               env.action_space.n, bias=False)
    # Q.weight.data.uniform_(-0.0001, 0.0001)
    
    Q = Q_Net(env)
    Q.linear_relu_stack[0].weight.data.uniform_(-0.0001, 0.0001)
    
    # use GPU
    # Q.to(device)
    
    return Q


def policy(Q, env, state, exploration_rate):
    if np.random.uniform(0, 1) < exploration_rate:
        return env.action_space.sample()
    
    # use GPU
#     state = torch.from_numpy(state).float()
#     state = state.to(device)
#     q = Q(state).detach().cpu()
#     q_values = q.numpy()
    
    q_values = Q(torch.from_numpy(state).float()).detach().numpy()
    return np.argmax(q_values)


def vfa_update(Q, optimizer, states, actions, rewards, dones, next_states):
    optimizer.zero_grad()
    states = torch.from_numpy(np.array(states)).float()
    actions = torch.from_numpy(np.array(actions)).unsqueeze(-1)
    rewards = torch.from_numpy(np.array(rewards)).float()
    dones = torch.from_numpy(np.array(dones)).float()
    next_states = torch.from_numpy(np.array(next_states)).float()

    # use GPU
#     states = states.to(device)
#     actions = actions.to(device)
#     rewards = rewards.to(device)
#     dones = dones.to(device)
#     next_states = next_states.to(device)
    
    """
    value function approximation update
    """
    q_values = torch.gather(Q(states), dim=-1, index=actions).squeeze()
    target_q_values = rewards + (1 - dones) * DISCOUNT_FACTOR * Q(next_states).max(dim=-1)[0].detach()
    loss = F.mse_loss(q_values, target_q_values)

    loss.backward()
    optimizer.step()
    
    return loss.item()


def q_learning(env, num_episodes, exploration_rate=0.9, exploration_rate_decay=0.999, min_exploration_rate=0.05, Q=None):
    # TODO: Update q-learning and add a replay-buffer
    if Q is None:
        Q = make_Q(env)

    optimizer = optim.Adam(Q.parameters(), lr=5e-4)
    rewards = []
    vfa_update_data = []
    for episode in range(num_episodes):
        rewards.append(0)
        obs = env.reset()
        state = obs

        for t in range(MAX_EPISODE_LENGTH):
#             if episode % 100 == 0:
#                 env.render()
            action = policy(Q, env, state, exploration_rate)

            obs, reward, done, _ = env.step(action)

            next_state = obs
            vfa_update_data.append((state, action, reward, done, next_state))

            state = next_state

            rewards[-1] += reward

            if len(vfa_update_data) >= BATCH_SIZE:
                vfa_update(Q, optimizer, *zip(*vfa_update_data))
                
                # replay
                replay_buffer.append(vfa_update_data)  # enqueue
                to_replay = np.random.uniform() < 0.5;
                if to_replay:
                    times = np.random.randint(queue_len)
                    for i in range(times):
                        if len(replay_buffer) == 0:
                            break
                        replay_data = replay_buffer.popleft()  # FIFO
                        vfa_update(Q, optimizer, *zip(*replay_data))
                
                vfa_update_data = []

            if done:
                break

        exploration_rate = max(exploration_rate_decay * exploration_rate, min_exploration_rate)
        if episode % (num_episodes / 100) == 0:
            print("Mean Reward: ", np.mean(rewards[-int(num_episodes / 100):]))
    return Q, rewards


def evaluation(env, Q):
    obs = env.reset()

    state = obs
    current_total_reward = 0
    for _ in range(1000):
        # env.render(current_total_reward)
        action = policy(Q, env, state, exploration_rate=0.0)
        obs, get_reward, done, _ = env.step(action)
        next_state = obs
        state = next_state
        current_total_reward += get_reward
        if done:
            print(f"with action from q_learning get reward {current_total_reward}")
            break
        print(f'state: {state}, action: {action}')

        # time.sleep(2)


if __name__ == "__main__":
    # env = gym.make('LunarLander-v2')
    env = TreeEnv()
    obs = env.reset()
    Q, rewards = q_learning(env, 10000)

    # save model
    torch.save(Q.state_dict(), 'deep_q_learning_model')

    _, ax = plt.subplots()
    ax.step([i for i in range(1, len(rewards) + 1)], rewards, linewidth=1.0)
    ax.grid()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    plt.title('Version 1 & Deep Q-Learning')
    plt.show()

    print(f'Mean reward: {np.mean(rewards)}')
    print(f'Standard deviation: {np.std(rewards)}')
    print(f'Max reward: {np.max(rewards)}')
    print(f'Min reward: {np.min(rewards)}')

    # read model
    Q = Q_Net(env)
    Q.load_state_dict(torch.load('deep_q_learning_model'))

    # evaluation
    evaluation(env, Q)

    env.close()
