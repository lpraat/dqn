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
          clip_grad=True,  # gradients are clipped between -1 and 1
          prioritized_replay=True,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta=0.4,
          prioritized_replay_beta_grow=lambda beta, train_step: beta + 1 / 200000,
          env=lander,
          tb_path="/Users/lpraat/Desktop/lunarlander/lander" + str(time.time()),
          save_path="/Users/lpraat/Desktop/lunarlander/model.ckpt",
          )

dqn.run()
# dqn.run_from_model()
