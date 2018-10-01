import numpy as np
import tensorflow as tf

from src.dqn.model import Model
from src.dqn.replay_memory import ReplayMemory
from src.utils import one_hot

Q_NETWORK_NAME = "q_network"
TARGET_Q_NETWORK_NAME = "target_q_network"


class DQN:
    def __init__(self, gamma, epsilon, epsilon_decay, epsilon_min,
                 learning_rate, replay_size, mini_batch_size,
                 update_freq, target_udpate_freq, env, path):
        self.env = env
        self.s = self.env.reset()

        self.num_actions = self.env.action_space.n
        self.state_size = len(self.s)

        self.r = 0
        self.total_rewards = []
        self.mean_reward = 0
        self.episode_reward = 0

        self.Q = Model(self.state_size, self.num_actions, learning_rate, Q_NETWORK_NAME)
        self.targetQ = Model(self.state_size, self.num_actions, learning_rate, TARGET_Q_NETWORK_NAME)
        self.q_params = tf.trainable_variables(Q_NETWORK_NAME)
        self.target_q_params = tf.trainable_variables(TARGET_Q_NETWORK_NAME)
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_size = replay_size
        dims = (self.state_size, 1, 1, self.state_size, 1)
        self.replay_memory = ReplayMemory(self.replay_size, dims)

        self.mini_batch_size = mini_batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_udpate_freq

        # Summaries
        self.writer = tf.summary.FileWriter(path)
        self.episode_reward_tf = tf.placeholder(dtype=tf.float32)
        self.episode_reward_summary = tf.summary.scalar("reward", self.episode_reward_tf)
        self.summaries = [*self.Q.summaries, self.episode_reward_summary]
        self.merged_summaries = tf.summary.merge(self.summaries)

        self.sess = None

    def q_step(self):
        s = self.s.reshape(1, self.state_size)

        if np.random.rand() > self.epsilon:
            a = np.argmax(self.sess.run(self.Q.output, feed_dict={self.Q.x: s}))
        else:
            # random action
            a = np.random.randint(self.num_actions)

        next_s, reward, done, _ = self.act(a)
        # reward = self.reward_f(reward)
        return np.array((s, a, reward, next_s, done)).reshape(1, 5)

    def act(self, a):
        self.env.render()
        observation, reward, end, info = self.env.step(a)
        self.r += reward

        if end:
            print(f"End of episode. Total reward: {self.r}")
            self.episode_reward = self.r
            self.total_rewards.append(self.r)

            if len(self.total_rewards) == 100:
                self.mean_reward = np.mean(self.total_rewards)
                print(f"Mean reward in 100 episodes: {self.mean_reward}")
                self.total_rewards = []

            self.r = 0
            self.s = self.env.reset()

        else:
            self.s = observation

        return observation, reward, end, info

    def train(self):

        states, actions, rewards, next_states, ends = self.replay_memory.sample_batch(self.mini_batch_size)
        one_hot_actions = one_hot(actions, self.num_actions)
        preds = self.sess.run(self.Q.output, feed_dict={self.Q.x: states})
        preds_t = self.sess.run(self.targetQ.output, feed_dict={self.targetQ.x: next_states})

        targets = np.zeros((self.mini_batch_size, preds.shape[1]))
        targets += rewards + self.gamma * np.max(preds_t, axis=1, keepdims=True) * (1 - ends)
        targets *= one_hot_actions
        targets += preds * (1 - one_hot_actions)

        self.fitQ(states, targets)

    def fitQ(self, states, targets):
        _, loss = self.sess.run((self.Q.optimizer, self.Q.loss), feed_dict={self.Q.x: states, self.Q.y: targets})
        self.writer.add_summary(self.sess.run(self.merged_summaries,
                                              feed_dict={self.Q.x: states, self.Q.y: targets,
                                                         self.episode_reward_tf: self.episode_reward}))

    def update_targetQ(self):
        for i in range(len(self.q_params)):
            self.sess.run(self.target_q_params[i].assign(self.q_params[i]))

    def run(self):

        step = 0

        with tf.Session() as sess:

            self.sess = sess
            self.sess.run(tf.global_variables_initializer())

            self.update_targetQ()

            self.writer.add_graph(sess.graph)

            while True:

                new_experience = self.q_step()
                self.epsilon = max(self.epsilon_min, self.epsilon_decay(self.epsilon, step))
                # print(self.epsilon)
                self.replay_memory.add_sample(new_experience)  # store transition in replay memory

                if step % self.update_freq == 0 and len(self.replay_memory.buffer) >= self.mini_batch_size:
                    self.train()

                if step % self.target_update_freq == 0:
                    self.update_targetQ()

                step += 1
