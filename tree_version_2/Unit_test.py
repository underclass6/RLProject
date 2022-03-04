import unittest
from Tree_env_1 import TreeEnv
import numpy as np

class Test_Unit:

    def test_start(self):
        print("test start now")

    def test_reward_1(self):
        env = TreeEnv()

        state = env.reset()
        _, reward, _, _=env.step(1)
        assert reward == 671.1393797973719

    def test_reward_2(self):
        env = TreeEnv()

        state = env.reset()
        _, reward, _, _=env.step(2)
        assert reward == 675.1327412287184

    def test_keep_cutdown(self):
        env = TreeEnv()
        state = env.reset()
        for i in range(8):
            state, reward, _, _=env.step(7)
        print(state)
        assert state == 0