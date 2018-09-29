import gym
import tensorflow as tf


from src.dqn.agent import Agent
from src.dqn.dqn import DQN


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_dim=4))
    model.add(tf.keras.layers.Dense(2))
    model.compile(loss=tf.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.RMSprop(lr=0.00025))
    return model


cart_agent = Agent(gym.make('CartPole-v0'), lambda r: r)

dqn = DQN(gamma=0.99,
          epsilon=1,
          epsilon_decay=lambda eps, step: eps - step / 1000000,
          epsilon_min=0.01,
          model_creator=create_model,
          replay_size=100000,
          mini_batch_size=64,
          update_freq=1,
          target_udpate_freq=500,
          agent=cart_agent)

dqn.run()
