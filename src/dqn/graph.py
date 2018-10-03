import tensorflow as tf
from collections import namedtuple


def new_model_graph(name, input_size, output_size, learning_rate):
    with tf.variable_scope(name):
        states = tf.placeholder(tf.float32, shape=[None, input_size])
        targets = tf.placeholder(tf.float32, shape=[None, output_size])
        actions = tf.placeholder(tf.int32, shape=[None, 1])

        h1_layer = tf.layers.dense(
            inputs=states,
            units=128,
            activation=tf.nn.relu
        )

        h2_layer = tf.layers.dense(
            inputs=h1_layer,
            units=64,
            activation=tf.nn.relu
        )

        output = tf.layers.dense(
            inputs=h2_layer,
            units=output_size,
            activation=tf.nn.relu
        )

        values = tf.multiply(output, (tf.one_hot(tf.squeeze(actions), output_size)))

        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=targets, predictions=values))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

        summaries = []
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        for v in train_vars:
            summaries.append(tf.summary.histogram(v.name, v))

        summaries.append(tf.summary.scalar("loss", loss))

        ModelGraph = namedtuple(name, ['states', 'targets', 'actions', 'output',
                                       'values', 'loss', 'optimizer', 'summaries'])
        return ModelGraph(states, targets, actions, output, values, loss, optimizer, summaries)


def new_targets_graph(mini_batch_size, num_actions):
    actions = tf.placeholder(dtype=tf.int32, shape=(mini_batch_size, 1))
    preds_next = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, num_actions))
    preds_t = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, num_actions))
    rewards = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, 1))
    ends = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, 1))
    gamma = tf.placeholder(dtype=tf.float32)

    one_hot_next_actions = tf.one_hot(tf.argmax(preds_next, axis=1), num_actions)
    next_qs = tf.reduce_sum(preds_t * one_hot_next_actions, axis=1, keepdims=True)
    targets = tf.zeros((mini_batch_size, num_actions))
    targets += rewards + (gamma * next_qs) * (1 - ends)
    targets *= tf.squeeze(tf.one_hot(actions, num_actions))

    TargetsGraph = namedtuple('TargetsGraph', ['actions', 'preds_next', 'preds_t',
                                               'rewards', 'ends', 'gamma', 'targets'])
    return TargetsGraph(actions, preds_next, preds_t, rewards, ends, gamma, targets)
