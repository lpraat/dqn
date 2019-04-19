from collections import namedtuple
from src.dqn.utils import clip_gradient_by_value

import tensorflow as tf


def new_dueling_model(name, input_size, output_size, learning_rate, clip_grad=False):
    states = tf.keras.layers.Input(shape=(input_size,))
    h1 = tf.keras.layers.Dense(256, activation='relu')(states)

    # State value function
    value_h2 = tf.keras.layers.Dense(128, activation='relu')(h1)
    value_output = tf.keras.layers.Dense(1)(value_h2)

    # Advantage function
    advantage_h2 = tf.keras.layers.Dense(128, activation='relu')(h1)
    advantage_output = tf.keras.layers.Dense(output_size)(advantage_h2)

    outputs = value_output + (advantage_output - tf.reduce_mean(advantage_output, axis=1, keepdims=True))

    model = tf.keras.Model(inputs=states, outputs=outputs, name=name)

    return model

# @tf.function  # This decoration does not work see https://stackoverflow.com/questions/55766641/gradients-are-none-when-using-tf-function-decorator
# TODO once this works add it
def q_train(states, actions, targets, is_weights, model, output_size, learning_rate, clip_grad):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    with tf.GradientTape() as tape:
        outputs = model(states)
        q_values = tf.multiply(outputs, (tf.one_hot(tf.squeeze(actions), output_size)))

        mse = tf.keras.losses.MeanSquaredError()(y_pred=q_values, y_true=targets)
        loss_value = tf.reduce_mean(is_weights * mse)

    grads = tape.gradient(loss_value, model.trainable_variables)

    selected_q_values = tf.reduce_sum(q_values, axis=1)
    selected_targets = tf.reduce_sum(targets, axis=1)
    td_errors = tf.clip_by_value(selected_q_values - selected_targets, -1.0, 1.0)

    if clip_grad:
        optimizer.apply_gradients(zip([tf.clip_by_value(grad, -1.0, 1.0) for grad in grads], model.trainable_variables))
    else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return td_errors


@tf.function
def get_targets(mini_batch_size, num_actions, actions, preds_next, preds_t, rewards, ends, gamma):
    one_hot_next_actions = tf.one_hot(tf.argmax(preds_next, axis=1), num_actions)
    next_qs = tf.reduce_sum(preds_t * one_hot_next_actions, axis=1, keepdims=True)
    targets = tf.zeros((mini_batch_size, num_actions))
    targets += rewards + (gamma * next_qs) * (1 - ends)
    targets *= tf.squeeze(tf.one_hot(actions, num_actions))

    return targets


@tf.function
def update_target_q(model, target_q_model):
    for i in range(len(model.trainable_variables)):
        target_q_model.trainable_variables[i].assign(model.trainable_variables[i])