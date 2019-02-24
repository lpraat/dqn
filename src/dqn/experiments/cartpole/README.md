# CartPole-v0

### Model1
- Layers:
    - Common: state_size -> 256
    - Value: common -> 128 -> 1
    - Advantage: common -> 128 -> num_actions
- Optimizer: RMSProp
```python
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
          env=gym.make('CartPole-v0')
          )
```

### Model2
- Layers:
    - Common: state_size -> 256
    - Value: common -> 128 -> 1
    - Advantage: common -> 128 -> num_actions
- Optimizer: RMSProp
```python
dqn = DQN(gamma=1,
          epsilon=1,
          epsilon_decay=lambda eps, step: eps - step / 10000000,
          epsilon_min=0.01,
          learning_rate=0.00025,
          replay_size=100000,
          mini_batch_size=64,
          update_freq=1,
          target_udpate_freq=100,
          clip_grad=True,  # gradients are clipped between -1 and 1
          prioritized_replay=True,
          prioritized_replay_alpha=0.3,
          prioritized_replay_beta=1,  # keep beta capped at 1
          env=gym.make('CartPole-v0')
          )
```

***Note***: By default, in PER, td errors are clipped between -1 and 1 and as a consequence the max priority is 1(to the power of alpha).

## Results
Both models solve the problem reaching an average reward over 100 episodes of 200.

In the graphs the x-axis corresponds to the "relative" time in tensorboard(number of hours since the start of the training) while the
y-axis corresponds to the total reward at the end of an episode.

| Compare1: Model1(Blue) vs Model2(Orange)                              |
|:---------------------------------------------------------------------:|  
|<img src="https://i.imgur.com/FSFv2Vb.png" width="550" height="350"/>  |

| Compare2: Model1(Light-Blue) vs Model2(Red)                           |
|:---------------------------------------------------------------------:|  
|<img src="https://i.imgur.com/ekaGlgw.png" width="550" height="350"/>  |
