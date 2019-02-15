import time

import gym

from src.dqn.dqn import DQN

cart = gym.make('CartPole-v0')

dqn = DQN(gamma=1,
          epsilon=1,
          epsilon_decay=lambda eps, step: eps - step / 10000000,
          epsilon_min=0.01,
          learning_rate=0.00025,
          replay_size=100000,
          mini_batch_size=64,
          update_freq=1,
          target_udpate_freq=100,
          clip_grad=False,
          env=cart,
          tb_path="/Users/lpraat/Desktop/cartpole/pole" + str(time.time()),
          save_path="/Users/lpraat/Desktop/lunarlander/model.ckpt",
          )

# PER
# dqn = DQN(gamma=1,
#           epsilon=1,
#           epsilon_decay=lambda eps, step: eps - step / 10000000,
#           epsilon_min=0.01,
#           learning_rate=0.00025,
#           replay_size=100000,
#           mini_batch_size=64,
#           update_freq=1,
#           target_udpate_freq=100,
#           prioritized_replay=True,
#           prioritized_replay_alpha=0.3,
#           prioritized_replay_beta=1,  # keep beta capped at 1
#           env=cart,
#           tb_path="/Users/lpraat/Desktop/cartpole/pole" + str(time.time()),
#           save_path="/Users/lpraat/Desktop/lunarlander/model.ckpt",
#           save_freq=1000
#           )

dqn.run()
# dqn.run_from_model()
