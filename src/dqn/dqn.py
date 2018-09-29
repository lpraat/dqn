import numpy as np

from src.dqn.replay_memory import ReplayMemory
from src.utils import one_hot


class DQN:

    def __init__(self, gamma, epsilon, epsilon_decay, epsilon_min,
                 model_creator, replay_size, mini_batch_size,
                 update_freq, target_udpate_freq, agent):
        self.agent = agent
        self.Q = model_creator()
        self.targetQ = model_creator()
        self.update_targetQ()

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_size = replay_size

        self.num_states = self.Q.layers[0].input_shape[1]
        dims = (self.num_states, 1, 1, self.num_states, 1)
        self.replay_memory = ReplayMemory(self.replay_size, dims)

        self.mini_batch_size = mini_batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_udpate_freq
        self.output_size = self.Q.layers[-1].output_shape[1]

    def q_step(self):
        s = self.agent.s.reshape(1, self.num_states)

        if np.random.rand() > self.epsilon:
            a = np.argmax(self.Q.predict(s))
        else:
            # random action
            a = np.random.randint(self.agent.num_actions)

        next_s, reward, done, _ = self.agent.act(a)
        reward = self.agent.reward_f(reward)
        return np.array((s, a, reward, next_s, done)).reshape(1, 5)

    def train(self):

        states, actions, rewards, next_states, ends = self.replay_memory.sample_batch(self.mini_batch_size)
        one_hot_actions = one_hot(actions, self.agent.num_actions)
        preds = self.Q.predict(states)
        preds_t = self.targetQ.predict(next_states)

        targets = np.zeros((self.mini_batch_size, preds.shape[1]))
        targets += rewards + self.gamma * np.max(preds_t, axis=1, keepdims=True) * (1-ends)
        targets *= one_hot_actions
        targets += preds * (1 - one_hot_actions)

        self.fitQ(states, targets)

    def fitQ(self, states, targets):
        self.Q.fit(x=states, y=targets, batch_size=self.mini_batch_size, epochs=1, verbose=0)

    def update_targetQ(self):
        self.targetQ.set_weights(self.Q.get_weights())

    def run(self):

        step = 0

        while True:

            new_experience = self.q_step()
            self.epsilon = max(self.epsilon_min, self.epsilon_decay(self.epsilon, step))
            self.replay_memory.add_sample(new_experience)  # store transition in replay memory

            if step % self.update_freq == 0 and len(self.replay_memory.buffer) >= self.mini_batch_size:
                self.train()

            if step % self.target_update_freq == 0:
                self.update_targetQ()

            step += 1


