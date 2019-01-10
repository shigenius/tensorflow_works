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
from nets.resnet_v2 import resnet_v2_50, resnet_arg_scope

archs = {
    'inception_v4': {'fn': inception_v4, 'arg_scope': inception_v4_arg_scope, 'extract_point': 'PreLogitsFlatten'},
    # 'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': 'shigeNet_v1/vgg_16/fc7'}# shape=(?, 14, 14, 512) dtype=float32
    'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': "shigeNet_v1/vgg_16/pool5"},
# shape=(?, 14, 14, 512) dtyp    e=float32
    'resnet_v2': {'fn': resnet_v2_50, 'arg_scope': resnet_arg_scope}
}


# shigeNet_v1/vgg_16/fc7
# 'shigeNet_v1/vgg_16/conv5/conv5_3' # shape=(?, 14, 14, 512) dtype=float32


def shigeNet_v1(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v1', reuse=None, extractor_name='inception_v4'):
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v1', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=True)
                feature_c = end_points_c['shigeNet_v1/vgg_16/pool5']
                feature_o = end_points_o['shigeNet_v1/vgg_16_1/pool5']
                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_c', tf.reshape(
                    tf.transpose(end_points_c['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                                 10)
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_o', tf.reshape(
                    tf.transpose(end_points_o['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                                 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)],
                                             1)  # (?, x, y, z)

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


def shigeNet_v2(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v2', reuse=None, extractor_name='vgg_16'):
    # vgg16のfreezeを解く．vggもトレーニングする．
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v2', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                with tf.variable_scope('orig') as scope:
                    logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g,
                                                                         is_training=True, reuse=None)
                    feature_o = end_points_o["shigeNet_v2/orig/vgg_16/fc7"]
                with tf.variable_scope('crop') as scope:
                    logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g,
                                                                         is_training=True, reuse=None)
                    feature_c = end_points_c["shigeNet_v2/crop/vgg_16/fc7"]

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v2/vgg_16/conv5/conv5_3_c', tf.reshape(
                    tf.transpose(end_points_c['shigeNet_v2/crop/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]),
                    [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v2/vgg_16/conv5/conv5_3_o', tf.reshape(
                    tf.transpose(end_points_o['shigeNet_v2/orig/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]),
                    [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)],
                                             1)  # (?, x, y, z)

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


def shigeNet_v3(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v3', reuse=None, extractor_name='vgg_16'):
    # vgg16のfreezeを解く．vggもトレーニングする．
    # softmax後をconcatする
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v3', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                with tf.variable_scope('orig') as scope:
                    logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g,
                                                                         is_training=True, reuse=None)
                    print(end_points_o)
                    feature_o = end_points_o['shigeNet_v3/orig/vgg_16/fc8']
                with tf.variable_scope('crop') as scope:
                    logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g,
                                                                         is_training=True, reuse=None)
                    feature_c = end_points_c['shigeNet_v3/crop/vgg_16/fc8']

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v3/vgg_16/conv5/conv5_3_c', tf.reshape(
                    tf.transpose(end_points_c['shigeNet_v3/crop/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]),
                    [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v3/vgg_16/conv5/conv5_3_o', tf.reshape(
                    tf.transpose(end_points_o['shigeNet_v3/orig/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]),
                    [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)],
                                             1)  # (?, x, y, z)

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


def shigeNet_v4(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v4', reuse=None, extractor_name='inception_v4'):
    # vggはfreezeする．featureをfc8に
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v4', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=True)

                feature_c = end_points_c['shigeNet_v4/vgg_16/fc8']
                feature_o = end_points_o['shigeNet_v4/vgg_16/fc8']

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v4/vgg_16/conv5/conv5_3_c', tf.reshape(
                    tf.transpose(end_points_c['shigeNet_v4/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                                 10)
                tf.summary.image('shigeNet_v4/vgg_16/conv5/conv5_3_o', tf.reshape(
                    tf.transpose(end_points_o['shigeNet_v4/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                                 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)],
                                             1)  # (?, x, y, z)

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


def shigeNet_v5(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v5', reuse=None, extractor_name='inception_v4'):
    # extract後，それぞれのfeatureをconv(trainable)する
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v5', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, num_classes=num_classes_g,
                                                                     is_training=False, reuse=True)
                feature_c = end_points_c['shigeNet_v5/vgg_16/pool5']
                feature_o = end_points_o['shigeNet_v5/vgg_16_1/pool5']

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v5/vgg_16/conv5/conv5_3_c', tf.reshape(
                    tf.transpose(end_points_c['shigeNet_v5/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                                 10)
                tf.summary.image('shigeNet_v5/vgg_16/conv5/conv5_3_o', tf.reshape(
                    tf.transpose(end_points_o['shigeNet_v5/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                                 10)

            with tf.variable_scope('ext_conv'):
                feature_c = slim.repeat(feature_c, 2, slim.conv2d, 512, [3, 3], scope='conv1_c')
                feature_c = slim.max_pool2d(feature_c, [2, 2], scope='pool_1_c')
                feature_o = slim.repeat(feature_o, 2, slim.conv2d, 512, [3, 3], scope='conv1_o')
                feature_o = slim.max_pool2d(feature_o, [2, 2], scope='pool_1_o')

            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)],
                                             1)  # (?, x, y, z)

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
    pool_size = 7  # ここは色々ためそう
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
                feature_c.append(end_points_c['shigeNet_v6/vgg_16/pool1'])  # shape=(?, 112, 112, 64)
                feature_c.append(end_points_c['shigeNet_v6/vgg_16/pool2'])  # shape=(?, 56, 56, 128)
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
                feature_c[0] = tf.image.resize_images(feature_c[0], (pool_size, pool_size))  # ここpoolで代用してもいいかも
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


def shigeNet_v7(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v7', reuse=None, extractor_name='resnet_v2'):
    # v1ベースにextractorをresnet_v2_152に
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v7', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = resnet_v2_50(cropped_images, num_classes=num_classes_g, is_training=False,
                                                      reuse=None)
                logits_o, end_points_o = resnet_v2_50(original_images, num_classes=num_classes_g, is_training=False,
                                                      reuse=True)
                feature_c = end_points_c['shigeNet_v7/resnet_v2_50/block4']
                feature_o = end_points_o['shigeNet_v7/resnet_v2_50/block4']  # (?, 7, 7, 2048)
                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                # tf.summary.image('shigeNet_v7/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v7/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                # tf.summary.image('shigeNet_v7/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v7/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)],
                                             1)  # (?, x, y, z)

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


def shigeNet_v8(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True,
                scope='shigeNet_v8', reuse=None, extractor_name='resnet_v2'):
    # v7ベースにそれぞれのfeatureをconvとpoint-wise convを行う．結合関係を学習させる．fcを削除
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v8', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = resnet_v2_50(cropped_images, num_classes=num_classes_g, is_training=False,
                                                      reuse=None)
                logits_o, end_points_o = resnet_v2_50(original_images, num_classes=num_classes_g, is_training=False,
                                                      reuse=True)
                feature_c = end_points_c['shigeNet_v8/resnet_v2_50/block4']
                feature_o = end_points_o['shigeNet_v8/resnet_v2_50/block4']  # (?, 7, 7, 2048)
                end_points['feature_c'] = feature_c
                end_points['feature_o'] = feature_o
                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                # tf.summary.image('shigeNet_v7/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v7/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                # tf.summary.image('shigeNet_v7/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v7/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            with tf.variable_scope('Branch_C') as scope:
                # feature_c = slim.conv2d(feature_c, 1096, [7, 7], scope='conv1', padding="VALID")
                #
                # feature_c = slim.conv2d(feature_c, num_classes_s, [1, 1], scope='conv2')
                feature_c = slim.conv2d(feature_c, 1096, [1, 1], scope='pw-conv')

            with tf.variable_scope('Branch_O') as scope:
                # feature_o = slim.conv2d(feature_o, 1096, [7, 7], scope='conv1', padding="VALID")
                #
                # feature_o = slim.conv2d(feature_o, num_classes_s, [1, 1], scope='conv2')
                feature_o = slim.conv2d(feature_o, 1096, [1, 1], scope='pw-conv')

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([feature_c, feature_o], 3)  # (?, x, y, z)
                # print(concated_feature)
                # w = tf.Variable(1.0, trainable=True, name='w')  # 結合関係を学習する．
                # net = feature_c + feature_o * w  # element-wise sum
                print(net)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    # net = slim.dropout(concated_feature, keep_prob, scope='dropout1')
                    # net = slim.conv2d(net, 1096, [1, 1], scope='fc1')
                    # net = slim.dropout(net, keep_prob, scope='dropout2')
                    # net = slim.conv2d(net, num_classes_s, [1, 1],
                    #      activation_fn=None,
                    #      normalizer_fn=None,
                    #      scope='fc2')

                    net = slim.conv2d(concated_feature, 512, [7, 7], scope='conv1', padding="VALID")
                    net = slim.conv2d(net, num_classes_s, [1, 1], scope='conv2')

                    net = tf.squeeze(net, [1, 2], name='fc2/squeezed')
                    print(net)
                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points