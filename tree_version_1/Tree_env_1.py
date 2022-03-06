import gym
from gym import spaces
import numpy as np
from typing import Optional
import pygame
import os
from sprites import *
import random


value_of_tree = [0, 1, 4, 9, 16, 25, 36, 49]


class TreeEnv(gym.Env):
    # the value of tree from age 1 to age 7
    def __init__(self):
        self.action_space = spaces.Discrete(8)  # in version_1, 1-7 means cut down the tree of the specified year
        # and to the next year,0 means do nothing direct to the next year
        self.observation_space = spaces.Discrete(100)
        self.reward_range = (0, 10000)
        self.state = None
        self.viewer = None
        self.year = 0
        np.random.seed(0)
        self._age_fixed = np.random.randint(size=100, low=-1, high=8)

    def reset(self, fix=True, seed=0):
        if fix:
            self.state = self._age_fixed.copy()
        else:
            np.random.seed(seed)
            self.state = np.random.randint(size=100, low=-1, high=8)
        self.year = 0
        return self.state

    def step(self, action):
        if action == 0:

            reward = 0

        elif action >= 1 and action <= 7:
            # cut all trees with specified aged
            reward = 0
            for i in range(100):
                if (self.state[i] == action):
                    self.state[i] = -1
                    reward += value_of_tree[action]

        done = False

        self.year += 1
        # trees grow, up to 7
        for i in range(100):
            if (self.state[i] != -1) and (self.state[i] != 7):
                self.state[i] += 1

            if (self.state[i] == 7):
                if i - 11 > 0:
                    if self.state[i - 11] == -1:
                        self.state[i - 11] = 0
                    if self.state[i - 10] == -1:
                        self.state[i - 10] = 0
                    if self.state[i - 9] == -1:
                        self.state[i - 9] = 0
                if i % 10 != 0 and self.state[i - 1] == -1:
                    self.state[i - 1] = 0
                if i % 10 != 9 and self.state[i + 1] == -1:
                    self.state[i + 1] = 0
                if i + 11 < 100:
                    if self.state[i + 11] == -1:
                        self.state[i + 11] = 0
                    if self.state[i + 10] == -1:
                        self.state[i + 10] = 0
                    if self.state[i + 9] == -1:
                        self.state[i + 9] = 0

        if np.all(self.state[:] == -1) or self.year > 10:
            done = True

        meta_info = {"year": self.year}
        return self.state, reward, done, meta_info

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
        timber_num_font = pygame.font.SysFont('arial', 50)
        tn_surface = timber_num_font.render(r'Timber: ' + str(current_total_reward), False, (130, 182, 217))
        screen.blit(tn_surface, (50, 620))

        # the number of year and its font
        year_num = 0
        year_num_font = pygame.font.SysFont('arial', 50)
        year_num_font_surface = year_num_font.render(r'Year: ' + str(self.year), False, (130, 182, 217))
        screen.blit(year_num_font_surface, (400, 620))

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

        print(self.state)
        for i in range(10):
            for j in range(10):
                tree = trees[i][j]
                tree.age = self.state[i * 10 + j]  # random generation

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
                af_surface = age_font.render(r'age: ' + str(tree.age), False, (130, 182, 180))
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
