import tensorflow as tf
from tensorflow.contrib import layers

alpha_l_relu = 0.1
def fully_conv(input_size, num_action):

    #state, minimap_input = cnn_block(input_size)
    #flatten_state = layers.flatten(state)
    #fc1 = layers.fully_connected(flatten_state, num_outputs=32)
    #value = tf.squeeze(layers.fully_connected(fc1, num_outputs=1, activation_fn=None), axis=1)
    #print("value shape is {}".format(value.shape))
    #logits = layers.conv2d(state, num_outputs=32, kernel_size=1, activation_fn=None, data_format="NCHW")
    #logits = leakyReLu(state, alpha_l_relu)
    #logits = layers.flatten(logits)
    #logits = layers.fully_connected(logits, num_outputs=64)
    #logits = leakyReLu(logits, alpha_l_relu)
    #logits = layers.fully_connected(logits, num_outputs=num_action)
    #    return [policy, value], minimap_input

    #blockinput = tf.placeholder(tf.float32, [None, 210, 160, 3])
    #blockinput = tf.placeholder(tf.float32, [None, input_size, input_size])
    blockinput = tf.placeholder(tf.float32, [None, input_size* input_size])
    #conv1 = layers.conv2d(blockinput, num_outputs=1, kernel_size=5, data_format="NCHW")
    #conv1 = leakyReLu(conv1, alpha_l_relu)
    #conv1 = layers.conv2d(blockinput, num_outputs=1, kernel_size=2, data_format="NCHW")
    #fc1 = leakyReLu(conv1, alpha_l_relu)
    #fc1 = layers.fully_connected(conv1, num_outputs=128)
    fc1 = tf.layers.dense(inputs=blockinput, units=200, activation=tf.nn.relu)
    #fc1 = leakyReLu(fc1, alpha_l_relu)
    value = tf.squeeze(layers.fully_connected(fc1, num_outputs=1))
    #fc2 = layers.fully_connected(fc1, num_outputs=32)
    #fc2 = tf.layers.dense(inputs=fc1, units=64, activation=tf.tanh)

    #fc2 = leakyReLu(fc2, alpha_l_relu)
    #value = tf.squeeze(layers.fully_connected(fc1, num_outputs=1))
    #logits = layers.fully_connected(fc2, num_outputs=num_action)
    logits = tf.layers.dense(inputs=fc1, units=3, activation=None)
    #print(logits.shape)
    policy = tf.nn.softmax(layers.flatten(logits))
    #policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layers.flatten(logits))
    #policy = tf.nn.sigmoid(logits)
    #print(policy.shape)
    #exit()
    return [policy, value], blockinput


def carpole_net(num_action):
    block_input = tf.placeholder(tf.float32, [None, 4])
    """
    flatten_state = layers.flatten(block_input)
    print(flatten_state.shape)
    value = tf.squeeze(layers.fully_connected(flatten_state, num_outputs=1, activation_fn=None), axis=1)
    logits = layers.fully_connected(flatten_state, num_outputs=num_action)
    logits = leakyReLu(logits, alpha_l_relu)
    policy = tf.nn.softmax(layers.flatten(logits))
    """
    B0 = tf.layers.dense(
        inputs=block_input,
        units=128,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
        bias_initializer=tf.constant_initializer(0.1),
    )
    B1 = tf.layers.dense(
        inputs=B0,
        units=32,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
        bias_initializer=tf.constant_initializer(0.1),
    )

    value = tf.squeeze(layers.fully_connected(B1, num_outputs=1, activation_fn=None), axis=1)

    A0 = tf.layers.dense(
        inputs=block_input,
        units=128,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
        bias_initializer=tf.constant_initializer(0.1),
    )
    A0 = tf.layers.dense(
        inputs=A0,
        units=num_action,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
        bias_initializer=tf.constant_initializer(0.1),
    )
    policy = tf.nn.softmax(A0)

    return [policy, value], block_input







def cnn_block(sz):
    block_input = tf.placeholder(tf.float32, [None, sz, sz])
    conv1 = layers.conv2d(block_input, num_outputs=16, kernel_size=5, data_format="NCHW")
    conv1 = leakyReLu(conv1, alpha_l_relu)
    #conv1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format="NHWC")
    #conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=3, data_format="NCHW")
    #conv2 = leakyReLu(conv2, alpha_l_relu)
    #conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=3, data_format="NHWC")
    return conv1, block_input


def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))