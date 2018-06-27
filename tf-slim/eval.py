import tensorflow as tf
import numpy as np
import cv2

import argparse
from datetime import datetime
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow.contrib.slim as slim

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from nets.inception_v4 import inception_v4, inception_v4_arg_scope
from nets.vgg import vgg_16, vgg_arg_scope

import csv

archs = {
    'inception_v4': {'fn': inception_v4, 'arg_scope': inception_v4_arg_scope, 'extract_point': 'PreLogitsFlatten'},
    'vgg_16': {'fn': vgg_16, 'arg_scope': vgg_arg_scope, 'extract_point': 'shigeNet_v1/vgg_16/fc7'}# shape=(?, 14, 14, 512) dtype=float32
}
# shigeNet_v1/vgg_16/fc7
# 'shigeNet_v1/vgg_16/conv5/conv5_3' # shape=(?, 14, 14, 512) dtype=float32


def shigeNet_v1(cropped_images, original_images, num_classes, keep_prob=1.0, is_training=True, scope='shigeNet_v1', reuse=None, extractor_name='inception_v4'):
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v1', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(archs[extractor_name]['arg_scope']()):
                logits_c, end_points_c = archs[extractor_name]['fn'](cropped_images, is_training=False, reuse=None)
                logits_o, end_points_o = archs[extractor_name]['fn'](original_images, is_training=False, reuse=True)

                feature_c = end_points_c[archs[extractor_name]['extract_point']]
                feature_o = end_points_o[archs[extractor_name]['extract_point']]

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
                    net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points


def eval(args):
    extractor_name = args.extractor

    restore_path = args.restore_path
    image_size = archs[extractor_name]['fn'].default_image_size
    num_classes = args.num_classes # road sign

    # Define placeholders
    with tf.name_scope('input'):
        with tf.name_scope('cropped_images'):
            cropped_images_placeholder = tf.placeholder(dtype="float32", shape=(None, image_size,  image_size,  3))
        with tf.name_scope('original_images'):
            original_images_placeholder = tf.placeholder(dtype="float32", shape=(None, image_size, image_size, 3))
        with tf.name_scope('labels'):
            labels_placeholder = tf.placeholder(dtype="float32", shape=(None, num_classes))
        keep_prob = tf.placeholder(dtype="float32")
        is_training = tf.placeholder(dtype="bool")  # train flag

    # Build the graph
    end_points = shigeNet_v1(cropped_images=cropped_images_placeholder,
                             original_images=original_images_placeholder,
                             extractor_name=extractor_name,
                             num_classes=num_classes, is_training=is_training, keep_prob=keep_prob)
    predictions = end_points["Predictions"]


    variables_to_restore = slim.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # dataset = TwoInputDataset(train_c=args.train_c, train_o=args.train_o, test_c=args.test_c, test_o=args.test_o,
    #                           num_classes=num_classes, image_size=image_size)

    def getPathandLabel(path, num_classes):
        with open(path, 'r') as f:
            f_ = [line.rstrip().split() for line in f]
            image_paths = [l[0] for l in f_]

            labels = []  # 1-of-kで用意する
            for l in f_:
                tmp = [0] * num_classes
                tmp[int(l[1])] = 1
                labels.append(tmp)

            return image_paths, labels

    pathes_c, labels_c = getPathandLabel(args.c, num_classes)
    pathes_o, labels_o = getPathandLabel(args.o, num_classes)

    # log
    f = open(args.log, 'w')

    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['path_c', 'result', 'prediction', 'GroundTruth'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, restore_path)
        print("Model restored from:", restore_path)

        for path_c, path_o, label in zip(pathes_c, pathes_o, labels_c):

            c = cv2.cvtColor(cv2.imread(path_c), cv2.COLOR_BGR2RGB)
            o = cv2.cvtColor(cv2.imread(path_o), cv2.COLOR_BGR2RGB)
            c = np.asarray([cv2.resize(c, (image_size, image_size))])
            o = np.asarray([cv2.resize(o, (image_size, image_size))])
            label = [label]

            result, pred = sess.run([correct_prediction, predictions],
                                   feed_dict={cropped_images_placeholder: c,
                                              original_images_placeholder: o,
                                              labels_placeholder: label,
                                              keep_prob: 1.0,
                                              is_training: False})

            print(path_c, result[0], np.argmax(pred), np.argmax(label))
            print(pred, label)
            writer.writerow([path_c, result[0], np.argmax(pred), np.argmax(label)])

    f.close()
    print('finished')

def eval_vgg16(args):
    restore_path = args.restore_path
    image_size = archs['vgg_16']['fn'].default_image_size
    num_classes = args.num_classes # road sign

    # Define placeholders
    with tf.name_scope('input'):
        with tf.name_scope('cropped_images'):
            cropped_images_placeholder = tf.placeholder(dtype="float32", shape=(None, image_size,  image_size,  3))
        with tf.name_scope('labels'):
            labels_placeholder = tf.placeholder(dtype="float32", shape=(None, num_classes))
        keep_prob = tf.placeholder(dtype="float32")
        is_training = tf.placeholder(dtype="bool")  # train flag

    # Build the graph
    with slim.arg_scope(vgg_arg_scope()):
        logits, _ = vgg_16(cropped_images_placeholder, num_classes=args.num_classes, is_training=True, reuse=None)

    predictions = tf.nn.softmax(logits, name='Predictions')

    variables_to_restore = slim.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def getPathandLabel(path, num_classes):
        with open(path, 'r') as f:
            f_ = [line.rstrip().split() for line in f]
            image_paths = [l[0] for l in f_]

            labels = []  # 1-of-kで用意する
            for l in f_:
                tmp = [0] * num_classes
                tmp[int(l[1])] = 1
                labels.append(tmp)

            return image_paths, labels

    pathes_c, labels_c = getPathandLabel(args.c, num_classes)

    # log
    f = open(args.log, 'w')

    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['path_c', 'result', 'prediction', 'GroundTruth'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, restore_path)
        print("Model restored from:", restore_path)

        for path_c, label in zip(pathes_c, labels_c):

            c = cv2.cvtColor(cv2.imread(path_c), cv2.COLOR_BGR2RGB)
            c = np.asarray([cv2.resize(c, (image_size, image_size))])
            label = [label]

            result, pred = sess.run([correct_prediction, predictions],
                                    feed_dict={cropped_images_placeholder: c,
                                               labels_placeholder: label,
                                               keep_prob: 1.0,
                                               is_training: False})

            print(path_c, result[0], np.argmax(pred), np.argmax(label))
            writer.writerow([path_c, result[0], np.argmax(pred), np.argmax(label)])

    f.close()
    print('finished')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', help='File name of test data(cropped)', default='/Users/shigetomi/Desktop/dataset_roadsign/2018_04_11test_crop.txt')
    parser.add_argument('-o', help='File name of test data (original)', default='/Users/shigetomi/Desktop/dataset_roadsign/2018_04_11test_orig.txt')
    parser.add_argument('-extractor', help='extractor architecture name', default='vgg_16')
    parser.add_argument('-net', help='network name', default='shigeNet')
    parser.add_argument('-log', help='log path', default='/Users/shigetomi/Desktop/log.csv')
    parser.add_argument('--num_classes', '-nc', type=int, default=6)
    parser.add_argument('--restore_path', '-r', default='/Users/shigetomi/workspace/tensorflow_works/tf-slim/model/twostep_roadsign.ckpt', help='ckpt path to restore')

    args = parser.parse_args()
    if args.net == 'shigeNet':
        eval(args)
    elif args.net == 'vgg_16':
        eval_vgg16(args)
    else:
        pass
