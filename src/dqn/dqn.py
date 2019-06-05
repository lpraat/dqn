import numpy as np
import tensorflow as tf

from src.dqn.replay.per_memory import PERMemory
from src.dqn.replay.replay_memory import ReplayMemory

class DQN:
    # TODO doc
    def __init__(self,
                 env,
                 gamma=0.99,
                 epsilon=1,
                 epsilon_decay=lambda eps, step: eps - step / 100000,
                 epsilon_min=0.01,
                 learning_rate=0.00025,
                 replay_size=100000,
                 mini_batch_size=64,
                 clip_grad=True,
                 update_freq=4,
                 prioritized_replay=False,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_beta_grow=lambda beta, train_step: beta + 1 / 100000,
                 prioritized_replay_epsilon=1e-3,
                 target_udpate_freq=500,
                 total_timesteps=np.inf,
                 tb_path=None,
                 save_path=None,
                 save_freq=5000,
                 ):
        self.env = env
        self.s = self.env.reset()
        self.num_actions = self.env.action_space.n
        self.state_size = len(self.s)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.total_timesteps = total_timesteps
        self.r = 0
        self.total_rewards = []
        self.curr_mean = 0

        self.mini_batch_size = mini_batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_udpate_freq
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
        self.prioritized_replay = prioritized_replay
        self.replay_size = replay_size

        self.q = self.new_dueling_model("q_network")
        self.target_q = self.new_dueling_model("target_q_network")

        if self.prioritized_replay:
            self.replay_memory = PERMemory(self.replay_size,
                                           self.state_size,
                                           alpha=prioritized_replay_alpha,
                                           beta=prioritized_replay_beta,
                                           epsilon=prioritized_replay_epsilon,
                                           beta_grow=prioritized_replay_beta_grow)
        else:
            self.replay_memory = ReplayMemory(self.replay_size, self.state_size)

        # Summaries
        self.tb_path = tb_path
        self.writer = tf.summary.create_file_writer(self.tb_path)

        # Model saving
        self.save_path = save_path
        if self.save_path:
            self.save_freq = save_freq

    def q_step(self):
        s = self.s.reshape(1, self.state_size)

        if np.random.rand() > self.epsilon:
            a = np.argmax(self.q_predict(s))
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
            print(f"End of episode. Reward: {self.r}")

            with self.writer.as_default():
                tf.summary.scalar("reward", self.r, step=0)

            if len(self.total_rewards) == 100:
                mean_reward = np.mean(self.total_rewards)
                print(f"Mean reward in 100 episodes: {mean_reward}")
                self.total_rewards = []

            self.r = 0
            self.s = self.env.reset()
        else:
            self.s = observation

        return observation, reward, end, info

    @tf.function
    def q_predict(self, next_states):
        return self.q(next_states, training=False)

    @tf.function
    def target_q_predict(self, next_states):
        return self.target_q(next_states, training=False)

    @tf.function
    def train_step(self, states, actions, targets, is_weights):
        with tf.GradientTape() as tape:
            outputs = self.q(states, training=True)
            q_values = tf.multiply(outputs, (tf.one_hot(tf.squeeze(actions), self.num_actions)))
            loss_value = tf.reduce_mean(is_weights * tf.losses.mean_squared_error(targets, q_values))

        grads = tape.gradient(loss_value, self.q.trainable_variables)

        selected_q_values = tf.reduce_sum(q_values, axis=1)
        selected_targets = tf.reduce_sum(targets, axis=1)
        td_errors = tf.clip_by_value(selected_q_values - selected_targets, -1.0, 1.0)

        if self.clip_grad:
            self.optimizer.apply_gradients(zip([tf.clip_by_value(grad, -1.0, 1.0) for grad in grads], self.q.trainable_variables))
        else:
            self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))

        return td_errors


    @tf.function
    def get_targets(self, actions, preds_next, preds_t, rewards, ends):
        one_hot_next_actions = tf.one_hot(tf.argmax(preds_next, axis=1), self.num_actions)
        next_qs = tf.reduce_sum(preds_t * one_hot_next_actions, axis=1, keepdims=True)
        targets = tf.zeros((self.mini_batch_size, self.num_actions))
        targets += rewards + (self.gamma * next_qs) * (1 - ends)
        targets *= tf.squeeze(tf.one_hot(actions, self.num_actions))
        return targets

    def update_target_q(self):
        self.target_q.set_weights(self.q.get_weights())

    def train(self, step):
        if self.prioritized_replay:
            states, actions, rewards, next_states, ends, is_weights, node_indices = \
                self.replay_memory.sample_batch(self.mini_batch_size)
        else:
            states, actions, rewards, next_states, ends = self.replay_memory.sample_batch(self.mini_batch_size)
            is_weights = np.ones_like(ends, dtype=np.float32)

        preds_next = self.q_predict(next_states)
        preds_t = self.target_q_predict(next_states)

        # TODO graph visualization does not work
        # see https://github.com/tensorflow/tensorboard/issues/1961
        # tf.summary.trace_on(graph=True, profiler=False)
        targets = self.get_targets(actions, preds_next, preds_t, rewards, ends)
        # with self.writer.as_default():
        #     tf.summary.trace_export(
        #         name="get_targets_graph",
        #         step=step)

        # tf.summary.trace_on(graph=True, profiler=False)
        td_errors = self.train_step(states, actions, targets, is_weights)
        # with self.writer.as_default():
        #     tf.summary.trace_export(
        #         name="train_step_graph",
        #         step=step)

        if self.prioritized_replay:
            self.replay_memory.update_priorities(node_indices, td_errors)

    def run(self):
        step = 0

        self.update_target_q()

        while step < self.total_timesteps:
            new_experience = self.q_step()
            self.epsilon = max(self.epsilon_min, self.epsilon_decay(self.epsilon, step))

            # Store transition in replay memory
            self.replay_memory.add_sample(new_experience)

            step += 1

            if step % self.update_freq == 0 and self.replay_memory.added_samples >= self.mini_batch_size:
                self.train(step)

            if step % self.target_update_freq == 0:
                self.update_target_q()

            if self.save_path and step % self.save_freq == 0:
                print(f"Saving model...")
                self.q.save_weights(self.save_path)

    # TODO add also a default CNN
    def new_dueling_model(self, name):
        states = tf.keras.layers.Input(shape=(self.state_size,))
        h1 = tf.keras.layers.Dense(256, activation='relu')(states)

        # State value function
        value_h2 = tf.keras.layers.Dense(128, activation='relu')(h1)
        value_output = tf.keras.layers.Dense(1, activation=None)(value_h2)

        # Advantage function
        advantage_h2 = tf.keras.layers.Dense(128, activation='relu')(h1)
        advantage_output = tf.keras.layers.Dense(self.num_actions, activation=None)(advantage_h2)

        outputs = value_output + (advantage_output - tf.reduce_mean(advantage_output, axis=1, keepdims=True))

        model = tf.keras.Model(inputs=states, outputs=outputs, name=name)

        return model

    def run_from_model(self):
        self.epsilon = 0
        self.q.load_weights(self.save_path)
        self.target_q.load_weights(self.save_path)

        while True:
            self.q_step()
