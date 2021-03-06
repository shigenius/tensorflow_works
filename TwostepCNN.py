#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.contrib.layers.python.layers.layers import batch_norm

class PrimaryCNN:
    def __init__(self):
        self.image_size = 227
        self.num_classes = 3

    def inference(self, images_placeholder, keep_prob, is_training):
        with tf.variable_scope('Primary') as scope:
            return base_architecture(self, images_placeholder, keep_prob, is_training)


class SecondaryCNN:
    def __init__(self):
        self.image_size = 227
        self.num_classes = 10

    def inference(self, images_placeholderA, images_placeholderB, keep_prob, is_training):
        with tf.variable_scope('Primary', reuse=True) as scope:
            imageA = tf.reshape(images_placeholderA, [-1, self.image_size, self.image_size, 3])
            primary_conv = base_cnns_with_bn(imageA, is_training)

        with tf.variable_scope('Secondary') as scope:
            imageB = tf.reshape(images_placeholderB, [-1, self.image_size, self.image_size, 3])
            secondary_conv = base_cnns_with_bn(imageB, is_training)

            with tf.name_scope('concat') as scope:
                h_concated = tf.concat([primary_conv, secondary_conv], 3)  # (?, x, y, z)

            with tf.variable_scope('fc1') as scope:
                h_concated_flat = tf.reshape(h_concated, [-1, 6 * 6 * 128 * 2])

                W_fc1 = weight_variable([6 * 6 * 128 * 2, 256], "w")
                b_fc1 = bias_variable([256], "b")
                h_fc1 = tf.nn.relu(tf.matmul(h_concated_flat, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            with tf.variable_scope('fc2') as scope:
                W_fc2 = weight_variable([256, self.num_classes], "w")
                b_fc2 = bias_variable([self.num_classes], "b")

            with tf.name_scope('softmax') as scope:
                y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            return y_conv


def base_architecture(self, images_placeholder, keep_prob, is_training):

    x_image = tf.reshape(images_placeholder, [-1, self.image_size, self.image_size, 3])

    # h_conv4 = base_cnns(x_image)
    h_conv4 = base_cnns_with_bn(x_image, is_training)

    with tf.variable_scope('fc1') as scope:
        h_conv4_flat = tf.reshape(h_conv4, [-1, 6 * 6 * 128])

        W_fc1 = weight_variable([6 * 6 * 128, 1024], "w")
        b_fc1 = bias_variable([1024], "b")
        h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, 256], "w")
        b_fc2 = bias_variable([256], "b")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.variable_scope('fc3') as scope:
        W_fc3 = weight_variable([256, self.num_classes], "w")
        b_fc3 = bias_variable([self.num_classes], "b")
        h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    with tf.name_scope('softmax') as scope:
        y = tf.nn.softmax(h_fc3)

    return y

def base_cnns(x_image):

    with tf.variable_scope('conv1') as scope:
        W_conv1 = weight_variable([11, 11, 3, 64], "w")
        b_conv1 = bias_variable([64], "b")
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, 64, 64], "w")
        b_conv2 = bias_variable([64], "b")
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)

    with tf.variable_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 64, 128], "w")
        b_conv3 = bias_variable([128], "b")
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 2, 2, 1], padding="VALID") + b_conv3)

    with tf.variable_scope('conv4') as scope:
        # h_pool2 = max_pool_2x2(h_conv2)
        W_conv4 = weight_variable([3, 3, 128, 128], "w")
        b_conv4 = bias_variable([128], "b")
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 2, 2, 1], padding="VALID") + b_conv4)

    return h_conv4

def base_cnns_with_bn(x_image, is_training):

    with tf.variable_scope('conv1') as scope:
        W_conv1 = weight_variable([11, 11, 3, 64], "w")
        b_conv1 = bias_variable([64], "b")
        h_conv1 = tf.nn.relu(batch_norm_layer(tf.nn.conv2d(x_image, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1, is_training))

    with tf.variable_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, 64, 64], "w")
        b_conv2 = bias_variable([64], "b")
        h_conv2 = tf.nn.relu(batch_norm_layer(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2, is_training))

    with tf.variable_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 64, 128], "w")
        b_conv3 = bias_variable([128], "b")
        h_conv3 = tf.nn.relu(batch_norm_layer(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 2, 2, 1], padding="VALID") + b_conv3, is_training))

    with tf.variable_scope('conv4') as scope:
        # h_pool2 = max_pool_2x2(h_conv2)
        W_conv4 = weight_variable([3, 3, 128, 128], "w")
        b_conv4 = bias_variable([128], "b")
        h_conv4 = tf.nn.relu(batch_norm_layer(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 2, 2, 1], padding="VALID") + b_conv4, is_training))

    return h_conv4

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial, trainable=True)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial, trainable=True)

def batch_norm_layer(x, train_phase, scope_bn='bn'):
    bn_train = batch_norm(x, decay=0.999, epsilon=1e-3, center=True, scale=True,
            updates_collections=None,
            is_training=True,
            reuse=None, # is this right?
            trainable=True,
            scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, epsilon=1e-3, center=True, scale=True,
            updates_collections=None,
            is_training=False,
            reuse=True, # is this right?
            trainable=True,
            scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
#                           strides=[1, 2, 2, 1], padding='VALID')

#
# class PrimaryCNN:
#     def __init__(self):
#         self.image_size = 227
#         self.num_classes = 53
#
#     def inference(self, images_placeholder, keep_prob):
#         with tf.variable_scope('Primary') as scope:
#             def weight_variable(shape, name):
#                 initial = tf.truncated_normal(shape, stddev=0.1)
#                 return tf.get_variable(name, initializer=initial, trainable=True)
#
#             def bias_variable(shape, name):
#                 initial = tf.constant(0.1, shape=shape)
#                 return tf.get_variable(name, initializer=initial, trainable=True)
#
#             def max_pool_2x2(x):
#                 return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
#                                       strides=[1, 2, 2, 1], padding='VALID')
#
#             x_image = tf.reshape(images_placeholder, [-1, self.image_size, self.image_size, 3])
#
#             with tf.variable_scope('conv1') as scope:
#                 W_conv1 = weight_variable([11, 11, 3, 64], "w")
#                 b_conv1 = bias_variable([64], "b")
#                 h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)
#
#             with tf.variable_scope('conv2') as scope:
#                 W_conv2 = weight_variable([3, 3, 64, 64], "w")
#                 b_conv2 = bias_variable([64], "b")
#                 h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)
#
#             with tf.variable_scope('conv3') as scope:
#                 W_conv3 = weight_variable([3, 3, 64, 128], "w")
#                 b_conv3 = bias_variable([128], "b")
#                 h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 2, 2, 1], padding="VALID") + b_conv3)
#
#             with tf.variable_scope('conv4') as scope:
#                 # h_pool2 = max_pool_2x2(h_conv2)
#                 W_conv4 = weight_variable([3, 3, 128, 128], "w")
#                 b_conv4 = bias_variable([128], "b")
#                 h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 2, 2, 1], padding="VALID") + b_conv4)
#
#             with tf.variable_scope('fc1') as scope:
#                 h_conv4_flat = tf.reshape(h_conv4, [-1, 6 * 6 * 128])
#
#                 W_fc1 = weight_variable([6 * 6 * 128, 1024], "w")
#                 b_fc1 = bias_variable([1024], "b")
#                 h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
#                 h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#             with tf.variable_scope('fc2') as scope:
#                 W_fc2 = weight_variable([1024, 256], "w")
#                 b_fc2 = bias_variable([256], "b")
#                 h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#                 h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#
#             with tf.variable_scope('fc3') as scope:
#                 W_fc3 = weight_variable([256, self.num_classes], "w")
#                 b_fc3 = bias_variable([self.num_classes], "b")
#                 h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
#
#         with tf.name_scope('softmax') as scope:
#             y = tf.nn.softmax(h_fc3)
#
#         return y
#
#
# class SecondaryCNN:
#     def __init__(self):
#         self.image_size = 227
#         self.num_classes = 10
#
#     def inference(self, images_placeholder1, images_placeholder2, keep_prob):
#         with tf.variable_scope('Primary', reuse=True) as scope:
#             def weight_variable(shape, name):
#                 initial = tf.truncated_normal(shape, stddev=0.1)
#                 return tf.get_variable(name, initializer=initial, trainable=True)  ### trainable=False
#
#             def bias_variable(shape, name):
#                 initial = tf.constant(0.1, shape=shape)
#                 return tf.get_variable(name, initializer=initial, trainable=True)
#
#             def max_pool_2x2(x):
#                 return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
#                                       strides=[1, 2, 2, 1], padding='VALID')
#
#             x_image = tf.reshape(images_placeholder1, [-1, self.image_size, self.image_size, 3])  # あとで2入力にする
#
#             with tf.variable_scope('conv1', reuse=True) as scope:
#                 W_P_conv1 = weight_variable([11, 11, 3, 64], "w")
#                 b_P_conv1 = bias_variable([64], "b")
#                 h_P_conv1 = tf.nn.relu(
#                     tf.nn.conv2d(x_image, W_P_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_P_conv1)
#
#             with tf.name_scope('pool1') as scope:
#                 h_P_pool1 = max_pool_2x2(h_P_conv1)
#
#             with tf.variable_scope('conv2', reuse=True) as scope:
#                 W_P_conv2 = weight_variable([3, 3, 64, 128], "w")
#                 b_P_conv2 = bias_variable([128], "b")
#                 h_P_conv2 = tf.nn.relu(
#                     tf.nn.conv2d(h_P_pool1, W_P_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_P_conv2)
#
#             with tf.name_scope('pool2') as scope:
#                 h_P_pool2 = max_pool_2x2(h_P_conv2)
#
#         with tf.variable_scope('Secondary') as scope:
#             def weight_variable(shape, name):
#                 initial = tf.truncated_normal(shape, stddev=0.1)
#                 return tf.get_variable(name, initializer=initial, trainable=True)
#
#             def bias_variable(shape, name):
#                 initial = tf.constant(0.1, shape=shape)
#                 return tf.get_variable(name, initializer=initial, trainable=True)
#
#             def max_pool_2x2(x):
#                 return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
#                                       strides=[1, 2, 2, 1], padding='VALID')
#
#             x_sub_image = tf.reshape(images_placeholder2, [-1, self.image_size, self.image_size, 3])
#
#             with tf.variable_scope('conv1') as scope:
#                 W_F_conv1 = weight_variable([11, 11, 3, 64], "w")
#                 b_F_conv1 = bias_variable([64], "b")
#                 h_F_conv1 = tf.nn.relu(
#                     tf.nn.conv2d(x_sub_image, W_F_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_F_conv1)
#
#             with tf.name_scope('pool1') as scope:
#                 h_F_pool1 = max_pool_2x2(h_F_conv1)
#
#             with tf.variable_scope('conv2') as scope:
#                 W_F_conv2 = weight_variable([3, 3, 64, 128], "w")
#                 b_F_conv2 = bias_variable([128], "b")
#                 h_F_conv2 = tf.nn.relu(
#                     tf.nn.conv2d(h_F_pool1, W_F_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_F_conv2)
#
#             with tf.name_scope('pool2') as scope:
#                 h_F_pool2 = max_pool_2x2(h_F_conv2)
#
#             with tf.name_scope('concat_PF') as scope:
#                 h_pool2 = tf.concat([h_P_pool2, h_F_pool2], 3)  # (?, x, y, z)
#
#             with tf.variable_scope('fc1') as scope:
#                 h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 128 * 2])
#
#                 W_fc1 = weight_variable([6 * 6 * 128 * 2, 256], "w")
#                 b_fc1 = bias_variable([256], "b")
#                 h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#                 h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#             with tf.variable_scope('fc2') as scope:
#                 W_fc2 = weight_variable([256, self.num_classes], "w")
#                 b_fc2 = bias_variable([self.num_classes], "b")
#
#             with tf.name_scope('softmax') as scope:
#                 y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
#             return y_conv
