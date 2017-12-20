#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

class PrimaryCNN:
    def __init__(self):
        self.image_size = 227
        self.num_classes = 53

    def inference(self, images_placeholder, keep_prob):
        with tf.variable_scope('Primary') as scope:
            def weight_variable(shape, name):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.get_variable(name, initializer=initial, trainable=True)
        
            def bias_variable(shape, name):
                initial = tf.constant(0.1, shape=shape)
                return tf.get_variable(name, initializer=initial, trainable=True)
        
            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1], padding='VALID')
            
            x_image = tf.reshape(images_placeholder, [-1, self.image_size, self.image_size, 3])
        
            with tf.variable_scope('conv1') as scope:
                W_conv1 = weight_variable([11, 11, 3, 64], "w")
                b_conv1 = bias_variable([64], "b")
                h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,4,4,1], padding="VALID") + b_conv1)

            with tf.name_scope('pool1') as scope:
                h_pool1 = max_pool_2x2(h_conv1)

            with tf.variable_scope('conv2') as scope:
                W_conv2 = weight_variable([3, 3, 64, 128], "w")
                b_conv2 = bias_variable([128], "b")
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,2,2,1], padding="VALID") + b_conv2)

            with tf.name_scope('pool2') as scope:
                h_pool2 = max_pool_2x2(h_conv2)

            with tf.variable_scope('fc1') as scope:
                h_pool2_flat = tf.reshape(h_pool2, [-1, 6*6*128])
    
                W_fc1 = weight_variable([6*6*128, 256], "w")
                b_fc1 = bias_variable([256], "b")
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            with tf.variable_scope('fc2') as scope:
                W_fc2 = weight_variable([256, self.num_classes], "w")
                b_fc2 = bias_variable([self.num_classes], "b")

        with tf.name_scope('softmax') as scope:
            y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
        return y_conv


class SecondaryCNN:
    def __init__(self):
        self.image_size = 227
        self.num_classes = 10

    def inference(self, images_placeholder1, images_placeholder2, keep_prob):
        with tf.variable_scope('Primary', reuse=True) as scope:
            def weight_variable(shape, name):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.get_variable(name, initializer=initial, trainable=True) ### trainable=False
        
            def bias_variable(shape, name):
                initial = tf.constant(0.1, shape=shape)
                return tf.get_variable(name, initializer=initial, trainable=True)
        
            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1], padding='VALID')
            
            x_image = tf.reshape(images_placeholder1, [-1, self.image_size, self.image_size, 3]) # あとで2入力にする
        
            with tf.variable_scope('conv1', reuse=True) as scope:
                W_P_conv1 = weight_variable([11, 11, 3, 64], "w")
                b_P_conv1 = bias_variable([64], "b")
                h_P_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_P_conv1, strides=[1,4,4,1], padding="VALID") + b_P_conv1)

            with tf.name_scope('pool1') as scope:
                h_P_pool1 = max_pool_2x2(h_P_conv1)

            with tf.variable_scope('conv2', reuse=True) as scope:
                W_P_conv2 = weight_variable([3, 3, 64, 128], "w")
                b_P_conv2 = bias_variable([128], "b")
                h_P_conv2 = tf.nn.relu(tf.nn.conv2d(h_P_pool1, W_P_conv2, strides=[1,2,2,1], padding="VALID") + b_P_conv2)

            with tf.name_scope('pool2') as scope:
                h_P_pool2 = max_pool_2x2(h_P_conv2)


        with tf.variable_scope('Secondary') as scope:
            def weight_variable(shape, name):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.get_variable(name, initializer=initial, trainable=True)
        
            def bias_variable(shape, name):
                initial = tf.constant(0.1, shape=shape)
                return tf.get_variable(name, initializer=initial, trainable=True)
        
            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1], padding='VALID')
            
            x_sub_image = tf.reshape(images_placeholder2, [-1, self.image_size, self.image_size, 3])
        
            with tf.variable_scope('conv1') as scope:
                W_F_conv1 = weight_variable([11, 11, 3, 64], "w")
                b_F_conv1 = bias_variable([64], "b")
                h_F_conv1 = tf.nn.relu(tf.nn.conv2d(x_sub_image, W_F_conv1, strides=[1,4,4,1], padding="VALID") + b_F_conv1)

            with tf.name_scope('pool1') as scope:
                h_F_pool1 = max_pool_2x2(h_F_conv1)

            with tf.variable_scope('conv2') as scope:
                W_F_conv2 = weight_variable([3, 3, 64, 128], "w")
                b_F_conv2 = bias_variable([128], "b")
                h_F_conv2 = tf.nn.relu(tf.nn.conv2d(h_F_pool1, W_F_conv2, strides=[1,2,2,1], padding="VALID") + b_F_conv2)

            with tf.name_scope('pool2') as scope:
                h_F_pool2 = max_pool_2x2(h_F_conv2)

            with tf.name_scope('concat_PF') as scope:
                h_pool2 = tf.concat([h_P_pool2, h_F_pool2], 3) # (?, x, y, z)
    
            with tf.variable_scope('fc1') as scope:
                h_pool2_flat = tf.reshape(h_pool2, [-1, 6*6*128*2])
        
                W_fc1 = weight_variable([6*6*128*2, 256], "w")
                b_fc1 = bias_variable([256], "b")
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
            with tf.variable_scope('fc2') as scope:
                W_fc2 = weight_variable([256, self.num_classes], "w")
                b_fc2 = bias_variable([self.num_classes], "b")
    
            with tf.name_scope('softmax') as scope:
                y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        
            return y_conv

