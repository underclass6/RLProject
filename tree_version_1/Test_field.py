import gym
from Tree_env_1 import TreeEnv


env = TreeEnv()

average_reward=0
average_count=10 # get 10 times random and caculate average reward
for i in range(average_count):
    env.reset()
    total_reward=0
    for _ in range(10000):
        env.render()
        current_state, get_reward, done, meta_info= env.step(env.action_space.sample()) # take a random action

        total_reward+=get_reward
        if done:
            print("done")
            print(f"get reward {total_reward}")
            average_reward+=total_reward
            if meta_info["year"]>10:
                print(f"10 years passed ")
                print(current_state)
            else:
                print(f"no avaliable tree now")
                print(current_state)
            break

    env.close()
print(f"random reward can be {average_reward/average_count}")