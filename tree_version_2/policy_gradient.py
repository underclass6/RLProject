import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from Tree_env_1 import TreeEnv
import time
MAX_EPISODE_LENGTH = 1000
DISCOUNT_FACTOR = 1.0
SEED = 0
MIN_BATCH_SIZE = 128

env = TreeEnv()

loss = 0  # global loss var


class Policy(nn.Module):
    """Define policy network"""

    def __init__(self, observation_space, action_space, hidden_size=256):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), hidden_size),
            nn.Linear(hidden_size, 64),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(64, hidden_size),
            nn.Linear(hidden_size, action_space.n),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        output = self.net(x)
        return output


policy = Policy(env.observation_space, env.action_space)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)


def compute_returns(rewards, discount_factor=DISCOUNT_FACTOR):
    """Compute discounted returns"""
    returns = []
    for i in range(len(rewards)):
        G = np.sum([reward * discount_factor ** j for j, reward in enumerate(rewards[i:])])
        returns.append(G)  # all returns correspond to each time step
    return returns


def policy_improvement(log_probs, rewards):
    """Compute REINFORCE policy gradient and perform gradient ascent step"""
    returns = compute_returns(rewards)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std())  # normalized return
    global loss
    loss = 0
    for log_prob, ret in zip(log_probs, returns):
        loss -= log_prob * ret  # without baseline
    return float(loss.data)  # this only return scalar value, not tensor


def act(state):
    """ Use policy to sample an action and return probability for gradient update"""
    state = torch.tensor(state)
    state = state.float()
    prob_dist = policy(state)  # probability distribution of actions given state
    actions = [i for i in range(env.action_space.n)]  # all possible actions
    action = np.random.choice(actions,
                              p=prob_dist.clone().detach().numpy())  # sample an action in terms of probability distribution
    return action, torch.log(prob_dist[action])


def policy_gradient(num_episodes):
    rewards = []
    for episode in range(num_episodes):
        rewards.append(0)
        trajectory = []
        state = env.reset()
        state_flatten = state.flatten('F')

        for t in range(MAX_EPISODE_LENGTH):
            #             if episode % (num_episodes / 100) == 0:
            #                 env.render()

            action, log_prob = act(state_flatten)

            # next_state, reward, done, _ = env.step(action.item())
            next_state, reward, done, _ = env.step(action)
            next_state_flatten = next_state.flatten('F')

            trajectory.append((log_prob, reward))

            state_flatten = next_state_flatten
            rewards[-1] += reward

            if done:
                break

        policy_improvement(*zip(*trajectory))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # backpropagation and gradient ascent

        if episode % (num_episodes / 100) == 0:
            print("Mean Reward: ", np.mean(rewards[-int(num_episodes / 100):]))
    return rewards


def evaluation(env, fix_seed=True, seed=0):
    state = env.reset(fix_seed, seed)
    state_flatten = state.flatten('F')

    current_total_reward = 0
    for _ in range(1000):
        # env.render(current_total_reward)
        action, log_prob = act(state_flatten)
        next_state, get_reward, done, _, = env.step(action)
        next_state_flatten = next_state.flatten('F')
        state_flatten = next_state_flatten
        current_total_reward += get_reward
        if done:
            print(f"with action from q_learning get reward {current_total_reward}")
            break
        print(f'state: {state}, action: {action}')

        # time.sleep(2)
    return current_total_reward


if __name__ == "__main__":
    time_start = time.time()
    rewards = policy_gradient(10000)
    time_end = time.time()
    print("time cost", time_end-time_start,'s')


    _, ax = plt.subplots()
    ax.step([i for i in range(1, len(rewards) + 1)], rewards, linewidth=0.5)
    ax.grid()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward & CO2 absorbency')
    plt.title('Version 5 & Policy Gradient')
    plt.show()

    print(f'Mean reward: {np.mean(rewards)}')
    print(f'Standard deviation: {np.std(rewards)}')
    print(f'Max reward: {np.max(rewards)}')
    print(f'Min reward: {np.min(rewards)}')

    # evaluation
    eval_rewards = []
    for seed in range(0, 50):
        r = evaluation(env, False, seed)
        eval_rewards.append(r)

    # random simulation
    sim_rewards = []
    for seed in range(0, 50):
        obs = env.reset(False, seed)
        current_total_reward = 0
        for _ in range(1000):
            obs, reward, done, _ = env.step(np.random.randint(0, 8))
            current_total_reward += reward
            if done:
                break
        sim_rewards.append(reward)
    _, ax1 = plt.subplots()
    ax1.bar([i for i in range(len(eval_rewards))], eval_rewards)
    ax1.set_xlabel('seed')
    ax1.set_ylabel('reward')
    _, ax2 = plt.subplots()
    ax2.bar([i for i in range(len(eval_rewards))], eval_rewards)
    ax2.bar([i for i in range(len(eval_rewards))], sim_rewards)
    ax2.set_xlabel('seed')
    ax2.set_ylabel('reward')
    plt.show()

    env.close()
