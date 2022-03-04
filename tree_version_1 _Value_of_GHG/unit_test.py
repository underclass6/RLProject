from Tree_env_1 import TreeEnv


class Test_Unit:

    def test_start(self):
        print("test start now")

    def test_reward_1(self):
        """
        test environment initial succesful or not
        """
        env = TreeEnv()

        state = env.reset()
        _, reward, _, _=env.step(1)
        assert reward != None

    # def test_reward_2(self):
    #     env = TreeEnv()
    #
    #     state = env.reset()
    #     _, reward, _, _=env.step(2)
    #     assert reward == 675.1327412287184
    #
    def test_reset(self):
        """
        test environment default content
        """
        env = TreeEnv()
        state = env.reset()
        for i in state:
            if i[0] == 0 or i[0] == -1:
                assert i[1] == 3.0
            assert (-1 <= i[0] <= 7).all()
            assert (0 <= i[1] <= 3).all()


    def test_keep_cutdown(self):

        """tree can only generate seed after it is 7 ages,
        if keep cut trees at 7, it will have no chance to generate seed
        so, after some times it will no tree left"""

        env = TreeEnv()
        state = env.reset()
        for i in range(16):
            state, reward, _, _=env.step(7)
        print(state)
        for i in state:

            assert (i[0] == -1 and i[1]==3)

    def test_fertility(self):
        env = TreeEnv()
        state = env.reset()
        for i in state:
            if i[0] == 0 or i[0] == -1:
                assert i[1] == 3.0
