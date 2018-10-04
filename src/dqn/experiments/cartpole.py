import gym

from src.dqn.dqn import DQN
import time


cart = gym.make('CartPole-v0')

dqn = DQN(gamma=0.999,
          epsilon=1,
          epsilon_decay=lambda eps, step: eps - step / 10000000,
          epsilon_min=0.02,
          learning_rate=0.00025,
          replay_size=100000,
          mini_batch_size=64,
          update_freq=1,
          target_udpate_freq=100,
          env=cart,
          path="/Users/lpraat/Desktop/cart_pole/pole" + str(time.time())
          )

dqn.run()
