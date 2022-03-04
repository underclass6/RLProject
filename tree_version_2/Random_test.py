import gym
from Tree_env_1 import TreeEnv

def random_field(counts=10):
    """
    do x times random test to get a average reward
    if x is None do 10 times
    """

    average_count = counts  # get 10 times random and caculate average reward

    average_reward=0


    env = TreeEnv()

    for i in range(average_count):
        env.reset()
        total_reward=0
        for _ in range(10000):
            env.render(total_reward)
            current_state, get_reward, done, meta_info= env.step(env.action_space.sample()) # take a random action

            total_reward+=get_reward
            if done:
                print("done")
                print(f"get reward {total_reward}")
                average_reward+=total_reward
                # if meta_info["year"]>10:
                #     print(f"10 years passed ")
                #     #print(current_state)
                # else:
                #     print(f"no avaliable tree now, which meanings all place are -1 now ")
                #     print(current_state)
                break

        env.close()
    print(f"run {average_count} times, average random reward can be {average_reward/average_count}")

if __name__ == "__main__":
    counts=12
    """ times you want to run for a average reward"""

    random_field(counts)