import numpy as np
import tensorflow as tf


def conv_weight_variable(shape, name):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.get_variable(name=name, dtype=tf.float32, initializer=initializer(shape))


def fc_weight_variable(shape, name):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, dtype=tf.float32, initializer=initializer(shape))


def bias_variable(shape, name):
    initializer = tf.constant(0.0, shape=shape)
    return tf.get_variable(name=name, dtype=tf.float32, initializer=initializer)


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(name, x, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME"):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        w = conv_weight_variable(shape=filter_shape, name='weight')
        b = bias_variable(shape=[num_filters], name='bias')
        res = tf.nn.relu(tf.nn.conv2d(x, w, stride_shape, pad) + b, name='res')
    return res


def linear(x, size, name):
    w_shape = [x.get_shape()[1].value, size]
    with tf.variable_scope(name):
        w = fc_weight_variable(shape=w_shape, name='weight')
        b = bias_variable(shape=[size], name='bias')
        res = tf.nn.relu(tf.nn.relu(tf.matmul(x, w) + b), name='res')
    return res


def maxpool(x, poolsize=(2, 2), stride=(1, 1), pad="SAME"):
    p_shape = [1, poolsize[0], poolsize[1], 1]
    stride_shape = [1, stride[0], stride[1], 1]
    res = tf.nn.max_pool(x, p_shape, stride_shape, pad)
    return res
