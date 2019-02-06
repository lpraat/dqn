import time

import gym

from src.dqn.dqn import DQN

lander = gym.make('LunarLander-v2')

dqn = DQN(gamma=0.99,
          epsilon=1,
          epsilon_decay=lambda eps, step: eps - step / 10000000,
          epsilon_min=0.01,
          learning_rate=0.0005,
          replay_size=100000,
          mini_batch_size=256,
          update_freq=4,
          target_udpate_freq=250,
          clip_value=True,
          env=lander,
          tb_path="/Users/lpraat/Desktop/lunarlander/lander" + str(time.time()),
          save_path="/Users/lpraat/Desktop/lunarlander/save"
          )

dqn.run()
