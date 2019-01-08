import tensorflow as tf
import numpy as np

import argparse
from datetime import datetime
from TwoInputDataset import TwoInputDataset
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow.contrib.slim as slim

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from nets.inception_v4 import inception_v4, inception_v4_arg_scope
from nets.vgg import vgg_16, vgg_arg_scope
from nets.resnet_v2 import resnet_v2_152, resnet_arg_scope
import cv2
import time

archs = {
    'inception_v4': {'fn': inception_v4, 'arg_scope': inception_v4_arg_scope, 'extract_point': 'PreLogitsFlatten'},
    # 'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': 'shigeNet_v1/vgg_16/fc7'}# shape=(?, 14, 14, 512) dtype=float32
'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': "shigeNet_v1/vgg_16/pool5"}, # shape=(?, 14, 14, 512) dtyp    e=float32
'resnet_v2' : {'fn': resnet_v2_152, 'arg_scope': resnet_arg_scope}
}
# shigeNet_v1/vgg_16/fc7
# 'shigeNet_v1/vgg_16/conv5/conv5_3' # shape=(?, 14, 14, 512) dtype=float32


def shigeNet_v1(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True, scope='shigeNet_v1', reuse=None, extractor_name='inception_v4'):
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v1', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g, is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g, is_training=False, reuse=True)
                feature_c = end_points_c['shigeNet_v1/vgg_16/pool5']
                feature_o = end_points_o['shigeNet_v1/vgg_16_1/pool5']
                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points


def shigeNet_v2(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True, scope='shigeNet_v2', reuse=None, extractor_name='vgg_16'):
    # vgg16のfreezeを解く．vggもトレーニングする．
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v2', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                with tf.variable_scope('orig') as scope:
                    logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g, is_training=True, reuse=None)
                    feature_o = end_points_o["shigeNet_v2/orig/vgg_16/fc7"]
                with tf.variable_scope('crop') as scope:
                    logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g, is_training=True, reuse=None)
                    feature_c = end_points_c["shigeNet_v2/crop/vgg_16/fc7"]

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v2/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v2/crop/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v2/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v2/orig/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points


def shigeNet_v3(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True, scope='shigeNet_v3', reuse=None, extractor_name='vgg_16'):
    # vgg16のfreezeを解く．vggもトレーニングする．
    # softmax後をconcatする
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v3', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                with tf.variable_scope('orig') as scope:
                    logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g, is_training=True, reuse=None)
                    print(end_points_o)
                    feature_o = end_points_o['shigeNet_v3/orig/vgg_16/fc8']
                with tf.variable_scope('crop') as scope:
                    logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g, is_training=True, reuse=None)
                    feature_c = end_points_c['shigeNet_v3/crop/vgg_16/fc8']

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v3/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v3/crop/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v3/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v3/orig/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points

def shigeNet_v4(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True, scope='shigeNet_v4', reuse=None, extractor_name='inception_v4'):
    # vggはfreezeする．featureをfc8に
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v4', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g, is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g, is_training=False, reuse=True)

                feature_c = end_points_c['shigeNet_v4/vgg_16/fc8']
                feature_o = end_points_o['shigeNet_v4/vgg_16/fc8']

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v4/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v4/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v4/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v4/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points

def shigeNet_v5(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True, scope='shigeNet_v5', reuse=None, extractor_name='inception_v4'):
    # extract後，それぞれのfeatureをconv(trainable)する
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v5', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g, is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g, is_training=False, reuse=True)
                feature_c = end_points_c['shigeNet_v5/vgg_16/pool5']
                feature_o = end_points_o['shigeNet_v5/vgg_16_1/pool5']

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v5/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v5/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v5/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v5/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            with tf.variable_scope('ext_conv'):
                feature_c = slim.repeat(feature_c, 2, slim.conv2d, 512, [3, 3], scope='conv1_c')
                feature_c = slim.max_pool2d(feature_c, [2, 2], scope='pool_1_c')
                feature_o = slim.repeat(feature_o, 2, slim.conv2d, 512, [3, 3], scope='conv1_o')
                feature_o = slim.max_pool2d(feature_o, [2, 2], scope='pool_1_o')

            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points

def shigeNet_v6(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v6', reuse=None, extractor_name='inception_v4'):
    # extratorのすべてのconv層の出力を用いる．
    # それぞれのconv層をresizeしたあとに
    pool_size = 7 # ここは色々ためそう
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v6', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=True)
                feature_c = []
                feature_o = []
                feature_c.append(end_points_c['shigeNet_v6/vgg_16/pool1']) # shape=(?, 112, 112, 64)
                feature_c.append(end_points_c['shigeNet_v6/vgg_16/pool2']) # shape=(?, 56, 56, 128)
                feature_c.append(end_points_c['shigeNet_v6/vgg_16/pool3'])  # shape=(?, 28, 28, 256)
                feature_c.append(end_points_c['shigeNet_v6/vgg_16/pool4'])  # shape=(?, 14, 14, 512)
                feature_c.append(end_points_c['shigeNet_v6/vgg_16/pool5'])  # shape=(?, 7, 7, 512)

                feature_o.append(end_points_o['shigeNet_v6/vgg_16_1/pool1'])  # shape=(?, 112, 112, 64)
                feature_o.append(end_points_o['shigeNet_v6/vgg_16_1/pool2'])  # shape=(?, 56, 56, 128)
                feature_o.append(end_points_o['shigeNet_v6/vgg_16_1/pool3'])  # shape=(?, 28, 28, 256)
                feature_o.append(end_points_o['shigeNet_v6/vgg_16_1/pool4'])  # shape=(?, 14, 14, 512)
                feature_o.append(end_points_o['shigeNet_v6/vgg_16_1/pool5'])  # shape=(?, 7, 7, 512)

                # サイズを揃える
                # slim.repeat(feature_c, 2, slim.max_pool2d, 512, [3, 3], scope='conv1_c')
                # slim.max_pool2d(feature_c, [2, 2], scope='pool_1_c')
                feature_c[0] = tf.image.resize_images(feature_c[0], (pool_size, pool_size)) # ここpoolで代用してもいいかも
                feature_c[1] = tf.image.resize_images(feature_c[1], (pool_size, pool_size))
                feature_c[2] = tf.image.resize_images(feature_c[2], (pool_size, pool_size))
                feature_c[3] = tf.image.resize_images(feature_c[3], (pool_size, pool_size))
                feature_c[4] = tf.image.resize_images(feature_c[4], (pool_size, pool_size))

                feature_o[0] = tf.image.resize_images(feature_o[0], (pool_size, pool_size))
                feature_o[1] = tf.image.resize_images(feature_o[1], (pool_size, pool_size))
                feature_o[2] = tf.image.resize_images(feature_o[2], (pool_size, pool_size))
                feature_o[3] = tf.image.resize_images(feature_o[3], (pool_size, pool_size))
                feature_o[4] = tf.image.resize_images(feature_o[4], (pool_size, pool_size))

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.concat(feature_c, 3), tf.concat(feature_o, 3)], 3)  # (?, x, y, z)
                # concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)],
                #                              1)  # (?, x, y, z)
                # concated_feature = tf.layers.Flatten()(slim.conv2d(concated_feature, 2944, [7, 7], padding='VALID', scope='conv_concat'))
                concated_feature = tf.layers.Flatten()(slim.max_pool2d(concated_feature, [7, 7], scope='pool_concat'))
                # print(concated_feature)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points


def shigeNet_v7(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True, scope='shigeNet_v7', reuse=None, extractor_name='inception_v4'):
    # v1ベースにextractorをresnet_v2_152に
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v7', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = resnet_v2_152(cropped_images, num_classes=num_classes_g, is_training=False, reuse=None)
                logits_o, end_points_o = resnet_v2_152(original_images, num_classes=num_classes_g, is_training=False, reuse=True)
                feature_c = end_points_c['shigeNet_v7/resnet_v2_152/block4']
                feature_o = end_points_o['shigeNet_v7/resnet_v2_152/block4'] # (?, 7, 7, 2048)
                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                # tf.summary.image('shigeNet_v7/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v7/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                # tf.summary.image('shigeNet_v7/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v7/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points

def calc_loss(output, supervisor_t):
  # cross_entropy = -tf.reduce_sum(supervisor_t * tf.log(y_pred))
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=supervisor_t)) # with_logitsは内部でソフトマックスも計算してくれる
  return cross_entropy

def training(loss):
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

def train(args):
    extractor_name = args.extractor

    model_path = args.model_path
    image_size = archs[extractor_name]['fn'].default_image_size
    num_classes_s = args.num_classes_s
    num_classes_g = args.num_classes_g
    val_freq = 1# Nstep毎にvalidate
    store_freq = 10

    arch_name = "shigeNet_v7"
    if arch_name == "shigeNet_v1":
        model = shigeNet_v1
    elif arch_name == "shigeNet_v2":
        model = shigeNet_v2
    elif arch_name == "shigeNet_v3":
        model = shigeNet_v3
    elif arch_name == "shigeNet_v4":
        model = shigeNet_v4
    elif arch_name == "shigeNet_v5":
        model = shigeNet_v5
    elif arch_name == "shigeNet_v6":
        model = shigeNet_v6
    elif arch_name == "shigeNet_v7":
        model = shigeNet_v7

    # Define placeholders
    with tf.name_scope('input'):
        with tf.name_scope('cropped_images'):
            x_c = tf.placeholder(dtype="float32", shape=(None, image_size,  image_size,  3))
        with tf.name_scope('original_images'):
            x_o = tf.placeholder(dtype="float32", shape=(None, image_size, image_size, 3))
        with tf.name_scope('labels'):
            t = tf.placeholder(dtype="float32", shape=(None, num_classes_s))
        keep_prob = tf.placeholder(dtype="float32")
        is_training = tf.placeholder(dtype="bool")  # train flag

        # # メモreshapeはずしてみる
        # tf.summary.image('cropped_images', tf.reshape(x_c, [-1, image_size, image_size, 3]), max_outputs=args.batch_size)
        # tf.summary.image('original_images', tf.reshape(x_o, [-1, image_size, image_size, 3]), max_outputs=args.batch_size)

    # Build the graph
    y = model(cropped_images=x_c, original_images=x_o, extractor_name=extractor_name, num_classes_s=num_classes_s, num_classes_g=num_classes_g, is_training=is_training, keep_prob=keep_prob)
    # logits = tf.squeeze(end_points["Logits"], [1, 2])
    y_logit = y["Logits"]
    y_pred = y["Predictions"]

    # Get restored vars name in checkpoint
    #def name_in_checkpoint(var):
    #    if arch_name+"/orig" in var.op.name:
    #        return var.op.name.replace(arch_name+"/orig/", "")
    #    if arch_name+"/crop" in var.op.name:
    #        return var.op.name.replace(arch_name+"/crop/", "")

    def name_in_checkpoint(var):
        if arch_name in var.op.name:
            return var.op.name.replace(arch_name+"/", "")

    # Get vars restored
    variables_to_restore = slim.get_variables_to_restore()
    # print(variables_to_restore)
    # dict of {name in checkpoint: var.op.name}
    variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore if extractor_name in var.op.name}
    #print(variables_to_restore)
    #print([key for key in variables_to_restore.keys() if key == None])
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=None)# save all vars

    # Train ops
    with tf.name_scope('loss'):
        # loss = tf.losses.softmax_cross_entropy(t, logits)
        loss = calc_loss(y_logit, t)
        tf.summary.scalar("loss", loss)

    with tf.name_scope('train') as scope:
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_step = slim.optimize_loss(loss, tf.train.get_or_create_global_step(), learning_rate=args.learning_rate, optimizer='Adam')
        train_step = training(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)

    dataset = TwoInputDataset(train_c=args.train_c, train_o=args.train_o, test_c=args.test_c, test_o=args.test_o,
                              num_classes=num_classes_s, image_size=image_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, model_path)
        print("Model restored from:", model_path)

        # summary
        if args.cvflag != None:  # if running cross-valid
            train_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/recorded_at_' + args.cvflag + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/recorded_at_' + args.cvflag + '/test')
        else:
            train_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(args.summary_dir + '/transfer_l/' + datetime.now().isoformat() + '/test')

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        merged = tf.summary.merge_all()

        num_batch = int(len(dataset.train_path_c) / args.batch_size)
        sum_iter = 0

        # Train cycle
        for step in range(args.max_steps):
            dataset.shuffle()

            train_acc_l = []
            train_loss_l = []
            step_start_time = time.time()

            # Train proc
            for i in range(num_batch-1):  # i : batch index
                cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, i)
                start_time = time.time()
                train_acc, train_loss, _ = sess.run([accuracy, loss, train_step],
                                         feed_dict={x_c: cropped_batch['batch'],
                                                    x_o: orig_batch['batch'],
                                                    t: labels,
                                                    keep_prob: args.dropout_prob,
                                                    is_training: True})
                elapsed_time = time.time() - start_time

                train_acc_l.append(train_acc)
                train_loss_l.append(train_loss)
                print("step%03d" % step, i, "of", num_batch, "train acc:",  train_acc, ", train loss:", train_loss, "elappsed time:", elapsed_time)
                train_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="train/acc_by_1iter", simple_value=train_acc)
                ]), sum_iter)
                train_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="train/loss_by_1iter", simple_value=train_loss)
                ]), sum_iter)
                sum_iter += 1

            # Final batch proc: get summary and train_trace
            cropped_batch, orig_batch, labels = dataset.getTrainBatch(args.batch_size, num_batch-1)
            summary = sess.run(merged,
                              feed_dict={x_c: cropped_batch['batch'],
                                         x_o: orig_batch['batch'],
                                         t: labels,
                                         keep_prob: args.dropout_prob,
                                         is_training: True},
                              options=run_options,
                              run_metadata=run_metadata)

            train_acc_l.append(train_acc)
            train_loss_l.append(train_loss)
            mean_train_accuracy = sum(train_acc_l) / len(train_acc_l)
            mean_train_loss = sum(train_loss_l) / len(train_loss_l)
            print('step %d: training accuracy %g,\t loss %g' % (step, mean_train_accuracy, mean_train_loss), "elapsed time:", step_start_time - time.time())

            # Write summary
            train_summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            # train_summary_writer.add_summary(summary, step)
            train_summary_writer.add_summary(summary, step)
            train_summary_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="train/mean_accuracy", simple_value=mean_train_accuracy)
            ]), step)
            train_summary_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="train/mean_loss", simple_value=mean_train_loss)
            ]), step)



            # Validation proc
            if step % val_freq == 0:
                num_test_batch = int(len(dataset.test_path_c) / args.batch_size)
                test_acc_l = []
                test_loss_l = []
                for i in range(num_test_batch):
                    test_cropped_batch, test_orig_batch, test_labels = dataset.getTestData(args.batch_size, i)
                    summary_test, test_accuracy, test_loss = sess.run([merged, accuracy, loss],
                                                                      feed_dict={x_c: test_cropped_batch['batch'],
                                                                                 x_o: test_orig_batch['batch'],
                                                                                 t: test_labels,
                                                                                 keep_prob: 1.0,
                                                                                 is_training: False})
                    test_acc_l.append(test_accuracy)
                    test_loss_l.append(test_loss)
                    print("Valid step%03d" % step, i, "of", num_test_batch, "acc:", test_accuracy, ", loss:", test_loss)

                print("test acc_list", test_acc_l)
                mean_test_acc = sum(test_acc_l) / len(test_acc_l)
                mean_test_loss = sum(test_loss_l) / len(test_loss_l)


                # Write valid summary
                # test_summary_writer.add_summary(summary_test, step)
                test_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid/accuracy", simple_value=mean_test_acc)
                ]), step)
                test_summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid/loss", simple_value=mean_test_loss)
                ]), step)

                print('step %d: test mean accuracy %g,\t mean loss %g' % (step, mean_test_acc, mean_test_loss))

            if step % store_freq == 0:
                # Save checkpoint model
                saver.save(sess, args.save_path, global_step=step)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_c', help='File name of train data(cropped)',   default='/Users/shigetomi/Desktop/dataset_walls/train2.txt')
    parser.add_argument('--train_o', help='File name of train data (original)', default='/Users/shigetomi/Desktop/dataset_walls/train1.txt')
    parser.add_argument('--test_c', help='File name of test data(cropped)',     default='/Users/shigetomi/Desktop/dataset_walls/test2.txt')
    parser.add_argument('--test_o', help='File name of test data (original)',   default='/Users/shigetomi/Desktop/dataset_walls/test1.txt')
    parser.add_argument('-extractor', help='extractor architecture name', default='vgg_16')

    parser.add_argument('--max_steps', '-s', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=20)
    parser.add_argument('--num_classes_s', '-ns', type=int, default=6)
    parser.add_argument('--num_classes_g', '-ng', type=int, default=9)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.5)

    parser.add_argument('--model_path', '-model', default='/Users/shigetomi/Downloads/vgg_16.ckpt', help='FullPath of inception-v4 model(ckpt)')
    parser.add_argument('--save_path', '-save', default='/Users/shigetomi/workspace/tensorflow_works/model/transl.ckpt', help='FullPath of saving model')
    parser.add_argument('--summary_dir', '-summary', default='/Users/shigetomi/workspace/tensorflow_works/log/', help='TensorBoard log')

    parser.add_argument('--cvflag', '-cv', default=None) # usually, dont use this

    args = parser.parse_args()

    train(args)

    print("all process finished　without a hitch. save the trained model at :", args.save_path)
