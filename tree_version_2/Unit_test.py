import unittest
from Tree_env_1 import TreeEnv
import numpy as np

class Test_Unit:


    def testStart(self):
        print("test start now")


    def testEnvInit(self):
        assert "h" in "this"

    def testRandomSeed(self):
        env = TreeEnv()

        state = env.reset(10)
        np.random.seed(10)
        age = np.random.randint(size=100, low=-1, high=8)
        fertility = np.random.random(100)
        test_state = np.column_stack((age, fertility))

        assert (state == test_state).all()

    def testReward_1(self):
        env = TreeEnv()

        state = env.reset()
        _, reward, _, _=env.step(1)
        assert reward == 671.1393797973719

    def testReward_2(self):
        env = TreeEnv()

        state = env.reset()
        _, reward, _, _=env.step(2)
        assert reward == 675.1327412287184

    def keepCutDown(self):
        env = TreeEnv()
        state = env.reset()
        for i in range(8):
            state, reward, _, _=env.step(7)
        print(state)
        assert state == 0