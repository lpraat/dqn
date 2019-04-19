import tensorflow as tf

@tf.custom_gradient
def clip_gradient_by_value(x, value):
  y = tf.identity(x)

  def grad_fn(grad):
    return tf.clip_by_value(grad, -value, value)

  return y, grad_fn