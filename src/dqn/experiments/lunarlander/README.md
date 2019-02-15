# LunarLander-v2

#### Model
- Layers:
    - Common: state_size -> 256
    - Value: common -> 128 -> 1
    - Advantage: common -> 128 -> num_actions
- Optimizer: RMSProp
```python
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
          env=gym.make('LunarLander-v2')
          )
```

***Note***: By default, in PER, td errors are clipped between -1 and 1 and as a consequence the max priority is 1.

## Results
The model solves the problem reaching an average reward over 100 episodes of ~260.
In the blue run below an average reward over 100 episodes of ~278 has been reached.

| (Blue)Run                                                              |
|:----------------------------------------------------------------------:|  
|<img src="https://i.imgur.com/9WTLnHi.png" width="550" height="350"/>   |

| (Orange)Run                                                            |
|:----------------------------------------------------------------------:|  
|<img src="https://i.imgur.com/C81I5VM.png" width="550" height="350"/>   |
