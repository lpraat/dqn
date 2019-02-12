import numpy as np
import tensorflow as tf

from src.dqn.graph import new_targets_graph, new_dueling_model_graph, new_update_target_graph
from src.dqn.per.per_memory import PERMemory
from src.dqn.replay_memory import ReplayMemory

Q_NETWORK_NAME = "q_network"
TARGET_Q_NETWORK_NAME = "target_q_network"


class DQN:
    # TODO add doc
    def __init__(self,
                 env,
                 gamma=0.99,
                 epsilon=1,
                 epsilon_decay=lambda eps, step: eps - step / 100000,
                 epsilon_min=0.01,
                 learning_rate=0.00025,
                 replay_size=100000,
                 mini_batch_size=64,
                 clip_value=False,
                 update_freq=4,
                 target_udpate_freq=500,
                 tb_path=None,
                 push_summaries_freq=100,
                 save_path=None,
                 save_freq=10000):
        # TODO add custom network defined by the user
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

        # Create tensorflow graphs
        # Q graph
        self.g_q = new_dueling_model_graph(Q_NETWORK_NAME, self.state_size, self.num_actions, learning_rate,
                                           clipvalue=clip_value)
        # Target Q graph
        self.g_target_q = new_dueling_model_graph(TARGET_Q_NETWORK_NAME, self.state_size, self.num_actions,
                                                  learning_rate, clipvalue=False)

        # Update target graph
        q_params = tf.trainable_variables(Q_NETWORK_NAME)
        target_q_params = tf.trainable_variables(TARGET_Q_NETWORK_NAME)
        self.g_update_target_q = new_update_target_graph(q_params, target_q_params)

        # Targets graph
        self.g_targets = new_targets_graph(self.mini_batch_size, self.num_actions)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_size = replay_size
        self.replay_memory = PERMemory(self.replay_size, self.state_size)

        # Summaries
        self.writer = tf.summary.FileWriter(tb_path)
        self.push_summaries_freq = push_summaries_freq

        # Create a summary for total reward per episode
        self.episode_reward_tf = tf.placeholder(dtype=tf.float32)
        self.episode_reward_summary = tf.summary.scalar("reward", self.episode_reward_tf)

        # Create a summary for moving average
        self.curr_mean_tf = tf.placeholder(dtype=tf.float32)
        self.curr_mean_summary = tf.summary.scalar("mean_reward", self.curr_mean_tf)

        # Merge all the summaries
        self.summaries = [*self.g_q.summaries, self.episode_reward_summary, self.curr_mean_summary]
        self.merged_summaries = tf.summary.merge(self.summaries)
        self.sess = None

        # Model saver
        self.save_path = save_path
        if self.save_path:
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(Q_NETWORK_NAME) +
                                                 tf.trainable_variables(TARGET_Q_NETWORK_NAME))
            self.save_freq = save_freq

    def q_step(self):
        s = self.s.reshape(1, self.state_size)

        if np.random.rand() > self.epsilon:
            a = np.argmax(self.sess.run(self.g_q.output, feed_dict={self.g_q.states: s}))
        else:
            # Random action
            a = np.random.randint(self.num_actions)

        next_s, reward, done, _ = self.act(a)
        return np.array((s, a, reward, next_s, done))

    def act(self, a):
        self.env.render()
        observation, reward, end, info = self.env.step(a)
        self.r += reward

        if end:
            self.total_rewards.append(self.r)
            self.episode_reward = self.r
            print(f"End of episode. Reward: {self.episode_reward}")

            # Exponentially weighted moving average with beta=0.99
            # Considers approximately 1/(1-beta)=100 last episode rewards
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
        states, actions, rewards, next_states, ends, is_weights, node_indices = \
            self.replay_memory.sample_batch(self.mini_batch_size)

        preds_next, preds_t = self.sess.run((self.g_q.output, self.g_target_q.output),
                                            feed_dict={
                                                self.g_q.states: next_states,
                                                self.g_target_q.states: next_states
                                            })

        targets = self.sess.run(self.g_targets.targets, feed_dict={
            self.g_targets.actions: actions,
            self.g_targets.preds_next: preds_next,
            self.g_targets.preds_t: preds_t,
            self.g_targets.rewards: rewards,
            self.g_targets.ends: ends,
            self.g_targets.gamma: self.gamma
        })

        _, abs_td_errors, merged_summaries = self.sess.run(
            (self.g_q.optimizer, self.g_q.abs_td_errors, self.merged_summaries),
            feed_dict={self.g_q.states: states,
                       self.g_q.targets: targets,
                       self.g_q.actions: actions,
                       self.g_q.is_weights: is_weights,
                       self.episode_reward_tf: self.episode_reward,
                       self.curr_mean_tf: self.curr_mean
                       })

        self.replay_memory.update_priorities(node_indices, abs_td_errors)

        if write_summaries:
            self.writer.add_summary(merged_summaries)

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

                # Store transition in replay memory
                self.replay_memory.add_sample(new_experience)

                step += 1

                if step % self.update_freq == 0 and self.replay_memory.added_samples >= self.mini_batch_size:

                    # Push summaries to event file
                    if step % self.push_summaries_freq == 0:
                        self.train(write_summaries=True)
                    else:
                        self.train(write_summaries=False)

                if step % self.target_update_freq == 0:
                    self.update_targetQ()

                if self.save_path and step % self.save_freq == 0:
                    print(f"Saving model...")
                    self.saver.save(sess, self.save_path)

    def run_from_model(self):
        self.epsilon = 0
        with tf.Session() as sess:
            self.sess = sess
            self.saver.restore(sess, self.save_path)
            while True:
                self.q_step()
