import gym
from gym import spaces
import numpy as np
from typing import Optional
import pygame
import os
from sprites import *
import random
import math

weight_timber = 1.0
weight_greenhouse_gas = 0.25

max_fertility = 3
""" the fertility of land will not more than 3"""

minimum_req_ghg_10years = 0
""" People will protest if the minimum is not reached(get a negative reward)"""

minimum_req_timber_1year = 0
""" If you don't meet the minimum you won't be able to pay the rent(get a negative reward)"""

random_seed= 10
"""seed for get a state by random, 
initial is 10 can also assigned by reset function
like reset(seed=10)
"""

value_of_tree_fn = lambda x: 0 if x == -1 else math.pi*((0.5*x) ** 2)  # calculate the value of tree wrt. age 1，4，9，16,25,36,49
value_of_greenhouse_gas_uptake_fn = lambda x: 0 if x == -1 else (x * 10)  # calculate this ability wrt. age * 10,  10 20 30 40 50 60 70
tree_growth_fn = lambda x: x / max_fertility  # calculate the ability of growth of tree wrt. fertility
absorb_fertility_ability_fn = lambda x: 0.05 * x  # calculate the absorbency of tree wrt. age


def reward_cauculator(timber, greenhouse_gas):
    """
    Calculate the weighted sum

    Arguments:
        timber -- reward from timber
        greenhouse_gas -- reward from greenhouse_gas_uptake

    Returns:
        A sum of weighted reward form timber and greenhouse gas uptake

    Can add more input, but also need set more parameters as correspond weight
    """
    return timber * weight_timber + greenhouse_gas * weight_greenhouse_gas

def print_info(action, reward, reward_timber, reward_co2reward, year, total_reward_timber, total_co2reward,state):
    """if needed, print some useful information for data analysis"""
    print(f"year {year}, take action: {round(action,3)}, sum of reward: {round(reward,3)}-> reward from tree: {round(reward_timber,3)}, reward from ghg: {round(reward_co2reward,3)}")
    print(f"totoal reward_timber: {round(total_reward_timber,3)}, total_ghg: {round(total_co2reward,3)}")
    # print("get new state")
    # print(state)
    print("======================================================================================")

class TreeEnv(gym.Env):
    """
    ### Description
    The Tree environment is try to Simulate the growth of trees in a piece of land,
    cutting down trees will gain monetary benefits, and retaining trees will gain
    environmental benefits (such as greenhouse gas uptake, or restoration of soil fertility),
    the goal is to find a balance between monetary benefits and environmental protection.
    the system show a land with a size of 10*10, action 0 means don't cut down any tree,
    action from 1-7 means cutting down the tree of the corresponding age

    ### Attrubutes:
        action_space: The action is discrete, deterministic, and  represents which age of trees will be cut down for Timber.
            | Num   | Action                                   |
            |------|-------------------------------------------|
            |0     | Do nothing, all trees will keep grow      |
            |x=1-7 | the trees in age x will be cut down and other trees will grow|

        observation_space: is a 'nd-array' with shape '(2)' that provides information about a piece of land
        | Num | Observation                | Min                  | Max                |
        |-----|----------------------------|----------------------|--------------------|
        | 0   | information of trees                | -1                 | 7                |
        | 1   | information of  max_fertility      | 0                 | 3                |
        where
        -information of trees ,is a float, -1 means no trees, 0 means have a seed, and will grow up next year,
        0.001-7.000 means, the age of trees. by 'value_of_tree_fn' can calculate the value of tree in this piece
        -information of  fertility, is a float, 0-3 means the fertility of this piece, by 'tree_growth_fn''
        can calculate how much the tree in this piece will grow.(like from age 1.0 to 1.5)

        ### Rewards
        the goal is to Provide the highest possible economic benefits(value of Timber) while ensuring environmental protection
        (for now means greenhouse gas uptake)

        ### Version History
        - v0
        - v1
        - v2
        - v3
    """

    def __init__(self):
        self.action_space = spaces.Discrete(8)  # in version_1, 1-7 means cut down the tree of the specified year
        # and to the next year,0 means do nothing direct to the next year
        # self.observation_space = spaces.Discrete(100)
        self.observation_space = spaces.Box(-1, 7, (100, 2))
        self.reward_range = (-1000, 10000)
        self.state = None
        self.viewer = None
        self.year = 0
        self.total_co2reward = 0
        self.total_reward_timber = 0
        self.total_reward = 0
        self.reward_timber = 0
        self.reward_co2reward = 0
        np.random.seed(2)
        self._age_fixed = np.random.randint(size=100, low=-1, high=8)

    def reset(self, fix=True, seed=0):
        if fix:
            age = self._age_fixed
        else:
            np.random.seed(seed)
            age = np.random.randint(size=100, low=-1, high=8)
        # fertility = np.random.random(100)
        fertility = np.zeros(100)
        for i in range(100):
            temp = age[i]
            if temp <= 0:
                fertility[i] = 3
            else:
                fertility[i] = 3
                for j in range(temp):
                    fertility[i] = fertility[i] - age[i] * 0.05

        self.state = np.column_stack((age, fertility))
        self.year = 0
        self.total_co2reward = 0
        self.total_reward = 0
        self.total_reward_timber = 0
        return self.state

    def step(self, action):
        reward = 0
        self.reward_timber = 0
        self.reward_co2reward = 0
        if action == 0:

            for i in range(100):
                self.reward_co2reward += value_of_greenhouse_gas_uptake_fn(self.state[i, 0])
            self.total_reward_timber += self.reward_timber
            self.total_co2reward += self.reward_co2reward
            reward = reward_cauculator(self.reward_timber, self.reward_co2reward)

        elif 1 <= action <= 7:
            # cut the specific trees with age as same as action-number

            if len(self.state[((action <= self.state[:, 0]) & (self.state[:, 0] < action + 1))]) != 0:
                self.reward_timber = sum(
                    [value_of_tree_fn(i) for i in self.state[((action <= self.state[:, 0]) & (self.state[:, 0] < action + 1))][:, 0]]
                )
                # cut down tree with age from specified age to specified age plus 1
                # and the fertility recovered to 3
                self.state[((action <= self.state[:, 0]) & (self.state[:, 0] < action + 1))] = (-1, max_fertility)



            for i in range(100):
                 self.reward_co2reward += value_of_greenhouse_gas_uptake_fn(self.state[i, 0])
                # print(self.total_co2reward)


            reward = reward_cauculator(self.reward_timber, self.reward_co2reward)
            self.total_reward_timber += self.reward_timber
            self.total_co2reward += self.reward_co2reward



        done = False
        self.year += 1

        # tree growing, state around 7-years-old tree will be planted
        for i in range(100):
            # absorb fertility
            if self.state[i, 0] != -1:
                self.state[i, 1] = max(0, self.state[i, 1] - absorb_fertility_ability_fn(self.state[i, 0]))

            if self.state[i, 0] >= 7:
                if i - 11 > 0:
                    if self.state[i - 11, 0] == -1:
                        self.state[i - 11, 0] = 0
                    if self.state[i - 10, 0] == -1:
                        self.state[i - 10, 0] = 0
                    if self.state[i - 9, 0] == -1:
                        self.state[i - 9, 0] = 0
                if i % 10 != 0 and self.state[i - 1, 0] == -1:
                    self.state[i - 1, 0] = 0
                if i % 10 != 9 and self.state[i + 1, 0] == -1:
                    self.state[i + 1, 0] = 0
                if i + 11 < 100:
                    if self.state[i + 11, 0] == -1:
                        self.state[i + 11, 0] = 0
                    if self.state[i + 10, 0] == -1:
                        self.state[i + 10, 0] = 0
                    if self.state[i + 9, 0] == -1:
                        self.state[i + 9, 0] = 0

            if self.state[i, 0] != -1 and self.state[i, 0] != 7:
                self.state[i, 0] += tree_growth_fn(self.state[i, 1])

        # If there is no more trees or passed 10 years stop, and if total green house gas uptake less than setting, return a negative reward.
        if np.all(self.state[:, 0] == -1) or self.year > 15:
            done = True
            if (self.total_co2reward <= minimum_req_ghg_10years ):#10000
                reward = -2000
            # if (self.total_co2reward <= 10000 or self.total_reward_timber<=100):
            #     reward = -self.total_reward + reward
            self.total_reward = 0

        meta_info = {"year": self.year}

        # If reward from timber less than 30 this year, retrun a negative reward
        if self.reward_timber <= minimum_req_timber_1year:
            reward = -200

        # print_info(action, reward, self.reward_timber, self.reward_co2reward, self.year, self.total_reward_timber,
        #            self.total_co2reward, self.state)
        return self.state, reward, done, meta_info   # , self.total_co2reward

    def render(self, current_total_reward=0):
        pygame.init()
        pygame.display.set_caption("Tree_cpation(template)")
        screen = pygame.display.set_mode((600, 700))
        screen.fill((0, 0, 0))
        clock = pygame.time.Clock()
        ground_paths = [os.getcwd() + r'/assets/PixelTrees/ground' + str(i + 1) + '.png' for i in range(4)]
        background = pygame.surface.Surface((600, 600))
        for i in range(10):
            for j in range(10):
                g = StaticObject(ground_paths[random.randint(0, 3)])
                g.resize(59, 59)
                g.set_pos((i * 60, j * 60))
                background.blit(g.image, g.rect)
        screen.blit(background, (0, 0))
        # create font of hint for the number of timber
        # timber_value = 0.0  # value of single timber
        # timber_profit = 0.0  # total profit
        timber_num_font = pygame.font.SysFont('arial', 25)
        tn_surface = timber_num_font.render(r'Timber: ' + str(round(self.reward_timber,3)), False, (130, 182, 217))
        screen.blit(tn_surface, (25, 620))

        # the number of year and its font
        year_num = 0
        year_num_font = pygame.font.SysFont('arial', 25)
        year_num_font_surface = year_num_font.render(r'Year: ' + str(round(self.year,3)), False, (130, 182, 217))
        screen.blit(year_num_font_surface, (500, 620))

        self.total_co2reward_font = pygame.font.SysFont('arial', 25)
        self.total_co2reward_font_surface = self.total_co2reward_font.render(
            r'Value of GHG: ' + str(round(self.total_co2reward,3)), False, (130, 182, 217))
        screen.blit(self.total_co2reward_font_surface, (200, 620))

        # create timber
        got_timber, got_timbers = False, False
        timber = RigidBody(os.getcwd() + r'/assets/timber.png')
        timber.set_acceleration(0.001)
        timbers = []  # store all timbers

        # get stump frames
        stump_frames = [
            pygame.image.load(os.getcwd() + r'/assets/PixelTrees/gif/stump/tile000.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/PixelTrees/gif/stump/tile001.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/PixelTrees/gif/stump/tile002.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/PixelTrees/gif/stump/tile003.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/PixelTrees/gif/stump/tile004.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/PixelTrees/gif/stump/tile005.png').convert_alpha()
        ]

        # get tree frames
        tree_frames = [
            pygame.image.load(os.getcwd() + r'/assets/trees-blackland/tree4/tree4_00.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/trees-blackland/tree4/tree4_01.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/trees-blackland/tree4/tree4_02.png').convert_alpha(),
            pygame.image.load(os.getcwd() + r'/assets/trees-blackland/tree4/tree4_03.png').convert_alpha()
        ]

        trees = []

        for _ in range(10):
            trees.append([])
            for _ in range(10):
                trees[-1].append(Tree([
                    os.getcwd() + r'/assets/trees-blackland/tree4/tree4_00.png',
                    os.getcwd() + r'/assets/trees-blackland/tree4/tree4_01.png',
                    os.getcwd() + r'/assets/trees-blackland/tree4/tree4_02.png',
                    os.getcwd() + r'/assets/trees-blackland/tree4/tree4_03.png'
                ], random.randint(2, 10)))


        for i in range(10):
            for j in range(10):
                tree = trees[i][j]
                tree.age = self.state[i * 10 + j, 0]  # random generation

                if tree.age == -1.0:
                    tree.is_chopped = True
                    tree.set_frames(stump_frames)
                tree.resize(45, 45)
                tree.set_pos((i * 60 + 7, j * 60 + 4))
                screen.blit(tree.image, tree.rect)

        # create font of hint for tree age
        age_font = pygame.font.SysFont('arial', 10)
        for row in trees:
            for tree in row:
                af_surface = age_font.render(r'age: ' + str(round(tree.age,1)), False, (130, 182, 180))
                screen.blit(af_surface, tree.rect.move(0, 40))

        # create select cursor
        select_cursor = AnimeObject([
            os.getcwd() + r'/assets/PixelTrees/gif/selectcursor/tile000.png',
            os.getcwd() + r'/assets/PixelTrees/gif/selectcursor/tile001.png',
            os.getcwd() + r'/assets/PixelTrees/gif/selectcursor/tile002.png',
            os.getcwd() + r'/assets/PixelTrees/gif/selectcursor/tile003.png',
            os.getcwd() + r'/assets/PixelTrees/gif/selectcursor/tile004.png',
            os.getcwd() + r'/assets/PixelTrees/gif/selectcursor/tile005.png'
        ])
        select_cursor.resize(65, 65)
        select_cursor.set_pos((-3, -3))
        select_cursor.draw(screen)

        pygame.display.flip()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        return True