from collections import namedtuple

import tensorflow as tf

def new_dueling_model_graph(name, input_size, output_size, learning_rate, clip_grad=False):
    with tf.variable_scope(name):
        states = tf.placeholder(tf.float32, shape=[None, input_size])
        targets = tf.placeholder(tf.float32, shape=[None, output_size])
        actions = tf.placeholder(tf.int32, shape=[None, 1])
        is_weights = tf.placeholder(tf.float32, shape=[None, 1])

        # TODO add possibility to define a custom network here
        h1_layer = tf.layers.dense(
            inputs=states,
            units=256,
            activation=tf.nn.relu
        )

        # State value function
        value_h2_layer = tf.layers.dense(
            inputs=h1_layer,
            units=128,
            activation=tf.nn.relu
        )

        value_output = tf.layers.dense(
            inputs=value_h2_layer,
            units=1,
            activation=None
        )

        # Advantage function
        advantage_h2_layer = tf.layers.dense(
            inputs=h1_layer,
            units=128,
            activation=tf.nn.relu
        )

        advantage_output = tf.layers.dense(
            inputs=advantage_h2_layer,
            units=output_size,
            activation=None
        )

        output = value_output + (advantage_output - tf.reduce_mean(advantage_output, axis=1, keepdims=True))

        q_values = tf.multiply(output, (tf.one_hot(tf.squeeze(actions), output_size)))

        selected_q_values = tf.reduce_sum(q_values, axis=1)
        selected_targets = tf.reduce_sum(targets, axis=1)
        td_errors = tf.clip_by_value(selected_q_values - selected_targets, -1.0, 1.0)

        loss = tf.reduce_mean(is_weights * tf.losses.mean_squared_error(labels=targets, predictions=q_values))

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        if clip_grad:
            grads_and_vars = optimizer.compute_gradients(loss)
            clipped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
            optimizer = optimizer.apply_gradients(clipped_grads_and_vars)
        else:
            optimizer = optimizer.minimize(loss)

        summaries = []
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        for v in train_vars:
            summaries.append(tf.summary.histogram(v.name, v))

        summaries.append(tf.summary.scalar("loss", loss))

        ModelGraph = namedtuple(name, ['states', 'targets', 'actions', 'output', 'q_values', 'is_weights',
                                       'td_errors', 'loss', 'optimizer', 'summaries'])

        return ModelGraph(states, targets, actions, output, q_values, is_weights,
                          td_errors, loss, optimizer, summaries)


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


def new_update_target_graph(q_params, target_q_params):
    return [tf.assign(target_q_params[i], q_params[i]) for i in range(len(q_params))]
