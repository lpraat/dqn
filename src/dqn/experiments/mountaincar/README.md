# MountainCar-v0

### Model
- Layers:
    - Common: state_size -> 256
    - Value: common -> 128 -> 1
    - Advantage: common -> 128 -> num_actions
- Optimizer: RMSProp
```python
dqn = DQN(gamma=0.99,
          epsilon=1,
          epsilon_decay=lambda eps, step: eps - step / 10000000,
          epsilon_min=1e-3,
          learning_rate=1e-3,
          replay_size=100000,
          mini_batch_size=64,
          update_freq=4,
          target_udpate_freq=500,
          clip_grad=True,  # gradients are clipped between -1 and 1
          prioritized_replay=True,
          prioritized_replay_alpha=0.9,
          prioritized_replay_beta=0.4,
          prioritized_replay_beta_grow=lambda beta, train_step: beta + 1 / 100000,
          env=gym.make('MountainCar-v0')
          )
```

***Note***: By default, in PER, td errors are clipped between -1 and 1 and as a consequence the max priority is 1.

## Results
The model solves the problem(it learns how to reach the top of the hill -- there are no "Solved Requirements" for this environment).
It reaches an average reward over 100 episodes of ~-101.

In the graphs the x-axis corresponds to the "relative" time in tensorboard(number of hours since the start of the training) while the
y-axis corresponds to the total reward at the end of an episode.

| (Orange)Run                                                              |
|:------------------------------------------------------------------------:|  
|<img src="https://i.imgur.com/MXJEp7g.png" width="550" height="350"/>     |

