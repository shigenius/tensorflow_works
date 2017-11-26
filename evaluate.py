#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import argparse

from train import *
from Dataset import Dataset

def main():
    archs = {
        'SimpleCNN': SimpleCNN
    }
    parser = argparse.ArgumentParser(description='Evaluate your saved model')
    parser.add_argument('test', help='File name of train data')
    parser.add_argument('model', help='FullPath of loading model (i.e. ./model.ckpt)')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='SimpleCNN', help='Convnet architecture')

    args = parser.parse_args()

    arch = archs[args.arch]() # model

    dataset = Dataset(test=args.test, num_classes=arch.num_classes, image_size=arch.image_size)

    with tf.Session() as sess:
        # imageとlabelを入れる仮のTensor
        images_placeholder = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
        labels_placeholder = tf.placeholder(dtype="float", shape=(None, arch.num_classes))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder(dtype="float")

        # モデルを作る
        logits = arch.inference(images_placeholder, keep_prob)
        # lossを計算
        loss_value = loss(logits, labels_placeholder)
        # accyracyの計算
        acc = accuracy(logits, labels_placeholder)

        # model parameterのrestore
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        print("Model restored from : ", args.model)

        test_data, test_labels = dataset.getTestData()
        print("test accuracy %g" % sess.run(acc, feed_dict={
            images_placeholder: test_data,
            labels_placeholder: test_labels,
            keep_prob: 1.0}))


if __name__ == '__main__':
    main()