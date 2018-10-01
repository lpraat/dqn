import tensorflow as tf


class Model:
    def __init__(self, input_size, output_size, learning_rate, name):
        self.name = name

        with tf.variable_scope(self.name):
            self.x = tf.placeholder(tf.float32, shape=[None, input_size])
            self.y = tf.placeholder(tf.float32, shape=[None, output_size])

            self.h1_layer = tf.layers.dense(
                inputs=self.x,
                units=128,
                activation=tf.nn.relu
            )

            self.h2_layer = tf.layers.dense(
                inputs=self.h1_layer,
                units=64,
                activation=tf.nn.relu
            )

            self.output = tf.layers.dense(
                inputs=self.h2_layer,
                units=output_size,
                activation=tf.nn.relu
            )

            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.output))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)

            self.summaries = []
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            for v in train_vars:
                self.summaries.append(tf.summary.histogram(v.name, v))

            self.summaries.append(tf.summary.scalar("loss", self.loss))







