import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from Tree_env_1 import TreeEnv

MAX_EPISODE_LENGTH = 1000
DISCOUNT_FACTOR = 1.0
SEED = 0
MIN_BATCH_SIZE = 256

env = TreeEnv()
# env = gym.make('LunarLander-v2')

loss = 0  # global loss var


class Policy(nn.Module):
    """Define policy network"""

    def __init__(self, observation_space, action_space, hidden_size=128):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.n, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space.n),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        output = self.net(x)
        return output


policy = Policy(env.observation_space, env.action_space)
optimizer = optim.Adam(policy.parameters(), lr=1e-5)


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
    GHGs = []
    for episode in range(num_episodes):
        rewards.append(0)
        GHGs.append(0)
        trajectory = []
        state = env.reset()
        # state_flatten = state.flatten('F')

        for t in range(MAX_EPISODE_LENGTH):
            #             if episode % (num_episodes / 100) == 0:
            #                 env.render()

            action, log_prob = act(state)

            # next_state, reward, done, _ = env.step(action.item())
            next_state, reward, done, _, GHG = env.step(action)
            # next_state_flatten = next_state.flatten('F')

            trajectory.append((log_prob, reward))

            state_flatten = next_state
            rewards[-1] += reward
            GHGs[-1] = GHG

            if done:
                break

        policy_improvement(*zip(*trajectory))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # backpropagation and gradient ascent

        if episode % (num_episodes / 100) == 0:
            print("Mean Reward: ", np.mean(rewards[-int(num_episodes / 100):]))
            print(f"Mean GHG:  {np.mean(GHGs[-int(num_episodes / 100):])}")
    return rewards, GHGs


if __name__ == "__main__":
    rewards, GHGs = policy_gradient(10000)

    _, ax = plt.subplots()
    ax.step([i for i in range(1, len(rewards) + 1)], rewards, linewidth=0.5)
    ax.step([i for i in range(1, len(rewards) + 1)], GHGs, linewidth=0.5)
    ax.grid()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward & CO2 absorbency')
    plt.legend(['reward', 'CO2 absorbency'])
    plt.title('Version 2 & Policy Gradient')
    plt.show()

    print(f'Mean reward: {np.mean(rewards)}')
    print(f'Standard deviation: {np.std(rewards)}')
    print(f'Max reward: {np.max(rewards)}')
    print(f'Min reward: {np.min(rewards)}')
    print(f'Mean CO2 absorbency: {np.mean(GHGs)}')
    print(f'Standard deviation: {np.std(GHGs)}')
    print(f'Max CO2 absorbency: {np.max(GHGs)}')
    print(f'Min CO2 absorbency: {np.min(GHGs)}')

    env.close()