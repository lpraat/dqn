import numpy as np


class Agent:

    def __init__(self, env, reward_f):
        self.env = env
        self.reward_f = reward_f
        self.num_actions = self.env.action_space.n
        self.s = self.env.reset()
        self.r = 0

        self.total_rewards = []

    def act(self, a):
        self.env.render()
        observation, reward, end, info = self.env.step(a)
        self.r += reward

        if end:
            print(f"End of episode - total reward: {self.r}")
            self.total_rewards.append(self.r)

            if len(self.total_rewards) == 100:
                mean_reward = np.mean(self.total_rewards)
                print(f"Mean reward in 100 episodes: {mean_reward}")
                self.total_rewards = []

            self.r = 0
            self.s = self.env.reset()
        else:
            self.s = observation

        return observation, reward, end, info




