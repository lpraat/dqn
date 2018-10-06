import numpy as np
import tensorflow as tf

from src.dqn.graph import new_targets_graph, new_dueling_model_graph, new_update_target_graph
from src.dqn.replay_memory import ReplayMemory

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
        self.episode_reward = 0
        self.total_rewards = []
        self.curr_mean = 0

        self.mini_batch_size = mini_batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_udpate_freq

        # create tensorflow graphs
        # q graph
        self.g_q = new_dueling_model_graph(Q_NETWORK_NAME, self.state_size, self.num_actions, learning_rate, clipvalue=True)
        # target q graph
        self.g_target_q = new_dueling_model_graph(TARGET_Q_NETWORK_NAME, self.state_size, self.num_actions, learning_rate, clipvalue=False)

        # update target graph
        q_params = tf.trainable_variables(Q_NETWORK_NAME)
        target_q_params = tf.trainable_variables(TARGET_Q_NETWORK_NAME)
        self.g_update_target_q = new_update_target_graph(q_params, target_q_params)

        # targets graph
        self.g_targets = new_targets_graph(self.mini_batch_size, self.num_actions)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_size = replay_size
        dims = (self.state_size, 1, 1, self.state_size, 1)
        self.replay_memory = ReplayMemory(self.replay_size, dims)

        # summaries
        self.writer = tf.summary.FileWriter(path)

        # create a summary for total reward per episode
        self.episode_reward_tf = tf.placeholder(dtype=tf.float32)
        self.episode_reward_summary = tf.summary.scalar("reward", self.episode_reward_tf)

        # create a summary for moving average
        self.curr_mean_tf = tf.placeholder(dtype=tf.float32)
        self.curr_mean_summary = tf.summary.scalar("mean_reward", self.curr_mean_tf)

        # merge all the summaries
        self.summaries = [*self.g_q.summaries, self.episode_reward_summary, self.curr_mean_summary]
        self.merged_summaries = tf.summary.merge(self.summaries)
        self.sess = None

    def q_step(self):
        s = self.s.reshape(1, self.state_size)

        if np.random.rand() > self.epsilon:
            a = np.argmax(self.sess.run(self.g_q.output, feed_dict={self.g_q.states: s}))
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
            self.total_rewards.append(self.r)
            self.episode_reward = self.r
            print(f"End of episode. Reward: {self.episode_reward}")

            # weighted average with beta=0.99 to consider approx 1/(1-beta)=100 last episode rewards
            self.curr_mean = 0.99 * self.curr_mean + 0.01 * self.r

            if len(self.total_rewards) == 100:
                mean_reward = np.mean(self.total_rewards)
                print(f"Mean reward in 100 episodes: {mean_reward}")
                self.total_rewards = []

            self.r = 0
            self.s = self.env.reset()

        else:
            self.s = observation

        return observation, reward, end, info

    def train(self, write_summaries):
        states, actions, rewards, next_states, ends = self.replay_memory.sample_batch(self.mini_batch_size)

        preds_next = self.sess.run(self.g_q.output, feed_dict={self.g_q.states: next_states})
        preds_t = self.sess.run(self.g_target_q.output, feed_dict={self.g_target_q.states: next_states})

        targets = self.sess.run(self.g_targets.targets, feed_dict={
            self.g_targets.actions: actions,
            self.g_targets.preds_next: preds_next,
            self.g_targets.preds_t: preds_t,
            self.g_targets.rewards: rewards,
            self.g_targets.ends: ends,
            self.g_targets.gamma: self.gamma
        })

        _, loss = self.sess.run((self.g_q.optimizer, self.g_q.loss),
                                feed_dict={self.g_q.states: states,
                                           self.g_q.targets: targets,
                                           self.g_q.actions: actions})

        if write_summaries:
            self.writer.add_summary(self.sess.run(self.merged_summaries,
                                                  feed_dict={self.g_q.states: states,
                                                             self.g_q.targets: targets,
                                                             self.g_q.actions: actions,
                                                             self.episode_reward_tf: self.episode_reward,
                                                             self.curr_mean_tf: self.curr_mean}))

    def update_targetQ(self):
        self.sess.run(self.g_update_target_q)

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

                step += 1

                if step % self.update_freq == 0 and len(self.replay_memory.buffer) >= self.mini_batch_size:

                    if step % 100 == 0:  # push summaries to event file every 50 step
                        self.train(write_summaries=True)
                    else:
                        self.train(write_summaries=False)

                if step % self.target_update_freq == 0:
                    self.update_targetQ()

