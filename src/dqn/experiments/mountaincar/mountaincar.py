import time

import gym

from src.dqn.dqn import DQN

cart = gym.make('MountainCar-v0')

dqn = DQN(gamma=0.99,
          epsilon=1,
          epsilon_decay=lambda eps, step: eps - step / 10000000,
          epsilon_min=1e-3,
          learning_rate=1e-3,
          replay_size=100000,
          mini_batch_size=64,
          update_freq=4,
          target_udpate_freq=500,
          clip_grad=True,
          prioritized_replay=True,
          prioritized_replay_alpha=0.9,
          prioritized_replay_beta=0.4,
          prioritized_replay_beta_grow=lambda beta, train_step: beta + 1 / 100000,
          env=cart,
          tb_path="/Users/lpraat/Desktop/mountaincar/car" + str(time.time()),
          save_path="/Users/lpraat/Desktop/mountaincar/model.ckpt"
          )
dqn.run()