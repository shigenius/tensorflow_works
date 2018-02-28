#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import argparse
import random
from datetime import datetime 

from Dataset import Dataset
from TwoInputDataset import TwoInputDataset
from TwostepCNN import PrimaryCNN, SecondaryCNN


### あとで消す
#
# def loss(logits, labels):
#     # 交差エントロピーの計算
#     # log(0) = NaN になる可能性があるので1e-10~1の範囲で正規化
#     cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10,1)))
#     # TensorBoardで表示するよう指定
#     tf.summary.scalar("cross_entropy", cross_entropy)
#     return cross_entropy
#
# def training(loss, learning_rate):
#     # 重みの更新
#     train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#     return train_step
#
# def accuracy(logits, labels, train=True):
#     if train == True:
#         with tf.name_scope('train') as scope:
#             correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
#             accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#             tf.summary.scalar("accuracy", accuracy)
#
#     else:
#         with tf.name_scope('test') as scope:
#             correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
#             accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#             tf.summary.scalar("accuracy", accuracy)
#
#     return accuracy

def primaryTrain(args) : 

    arch = PrimaryCNN()
    dataset = Dataset(train=args.train1, test=args.test1, num_classes=arch.num_classes, image_size=arch.image_size)
    
    with tf.Graph().as_default():
        # imageとlabelを入れるTensor
        images_placeholder = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
        labels_placeholder = tf.placeholder(dtype="float", shape=(None, arch.num_classes))
        # dropout率を入れるTensor
        keep_prob = tf.placeholder(dtype="float")
    
        # 計算モデルのop
        logits = arch.inference(images_placeholder, keep_prob)

        # log(0) = NaN になる可能性があるので1e-10~1の範囲で正規化
        loss = -tf.reduce_sum(labels_placeholder*tf.log(tf.clip_by_value(logits, 1e-10,1)))

        # accuracyのop
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.name_scope('train') as scope:
            # trainのop
            train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
            acc_summary_train = tf.summary.scalar("train_accuracy", accuracy)
            loss_summary_train = tf.summary.scalar("train_loss", loss)

        with tf.name_scope('test') as scope:
            acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)
            loss_summary_test = tf.summary.scalar("test_loss", loss)

        # 保存の準備
        saver = tf.train.Saver()

        # Sessionの作成
        sess = tf.Session()

        # # variables
        # vars_all = tf.global_variables()
        # print("\nvars_all", vars_all, len(vars_all))

        # 変数の初期化
        sess.run(tf.global_variables_initializer())

        # TensorBoardで表示する値の設定
        #summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.logdir+"/Primary/"+datetime.now().isoformat(), sess.graph)

        training_op_list = [accuracy, acc_summary_train, loss_summary_train]
        val_op_list = [accuracy, acc_summary_test, loss_summary_test]

        # 訓練の実行
        for step in range(args.max_steps):
            dataset.shuffle() # バッチで取る前にデータセットをshuffleする   
            #batch処理
            for i in range(int(len(dataset.train_image_paths)/args.batch_size)): # iがbatchのindexになる #バッチのあまりが出る

                batch, labels = dataset.getTrainBatch(args.batch_size, i)

                # バッチを学習
                sess.run(train_step, feed_dict={images_placeholder: batch, labels_placeholder: labels, keep_prob: args.dropout_prob})

                # 最終バッチの処理
                if i >= int(len(dataset.train_image_paths)/args.batch_size)-1:

                    # 最終バッチの学習のあと，そのバッチを使って評価．
                    result = sess.run(training_op_list, feed_dict={images_placeholder: batch, labels_placeholder: labels, keep_prob: 1.0})

                    # 必要なサマリーを追記
                    for j in range(1, len(result)):
                        summary_writer.add_summary(result[j], step)

                    print("step %d  training final-batch accuracy: %g"%(step, result[0]))
    
                    # validation
                    test_data, test_labels = dataset.getTestData()
                    val_result = sess.run(val_op_list, feed_dict={images_placeholder: test_data, labels_placeholder: test_labels, keep_prob: 1.0})

                    # 必要なサマリーを追記
                    for j in range(1, len(result)):
                        summary_writer.add_summary(result[j], step)

                    print("test accuracy %g" % val_result[0])
    
    # 最終的なモデルを保存
    save_path = saver.save(sess, args.save_path)
    print("save the trained model at :", save_path)


def secondaryTrain(args):

    arch    = PrimaryCNN()
    subarch = SecondaryCNN()
    
    dataset = TwoInputDataset(train1=args.train1, train2=args.train2, test1=args.test1, test2=args.test2,  num_classes=arch.num_classes, image_size=arch.image_size)

    with tf.Graph().as_default():

        with tf.Session() as sess:
            # placeholderたち
            images_placeholderA = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
            images_placeholderB = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3))
            labels_placeholder = tf.placeholder(dtype="float", shape=(None, arch.num_classes))
            keep_prob = tf.placeholder(dtype="float")

            # 仮のinference (ckptに保存されている変数の数と，tf.train.Saver()を呼ぶ前に宣言する変数数を揃える必要があるため．) もうちょいいい書き方があるかもしれない
            P_logits = arch.inference(images_placeholderA, keep_prob)
    
            # restore
            saver = tf.train.Saver()
            saver.restore(sess, args.model)
            print("Model restored from : ", args.model)
            vars_restored = tf.global_variables() # restoreしてきたvariableのリスト
            # print("\nvars_restored", vars_restored)

            # 計算モデルのop
            logits = subarch.inference(images_placeholderA, images_placeholderB, keep_prob) # namescope "Primary/"以下はrestoreしたVariableになっている，はず

            # 更新するvariableをもとめる
            vars_all = tf.global_variables()
            vars_S = list(set(vars_all) - (set(vars_restored))) # SecondaryCNNにのみ定義してあるvariable
            # print("\nvars_all", vars_all, len(vars_all))
            # print("\nvars_S", vars_S, len(vars_S))

            # loss_value = loss(logits, labels_placeholder)
            # train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_value, var_list=vars_S) # vars_Sのみloss用いて重みを更新する．
            # acc = accuracy(logits, labels_placeholder)


            # log(0) = NaN になる可能性があるので1e-10~1の範囲で正規化
            loss = -tf.reduce_sum(labels_placeholder * tf.log(tf.clip_by_value(logits, 1e-10, 1)))

            # accuracyのop
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            with tf.name_scope('train') as scope:
                # trainのop
                train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, var_list=vars_S)
                acc_summary_train = tf.summary.scalar("train_accuracy", accuracy)
                loss_summary_train = tf.summary.scalar("train_loss", loss)

            with tf.name_scope('test') as scope:
                acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)


            vars_all = tf.global_variables()
            vars_init = list(set(vars_all) - (set(vars_restored)))
            # print("\nvars_all", vars_all)
            # print("\nvars_init", vars_init)

            # variableの初期化．restoreしてきたものは初期化しない．
            sess.run(tf.variables_initializer(vars_init))

            # print(vars_restored, vars_restored[0], sess.run(vars_restored[0])) # 重みの取得
    
            saver = tf.train.Saver() # 再びsaverを呼び出す

            # # variables
            # vars_all = tf.global_variables()
            # print("\nvars_all", vars_all, len(vars_all))

            # TensorBoardで表示する値の設定
            summary_writer = tf.summary.FileWriter(args.logdir + "/Secondary/" + datetime.now().isoformat(), sess.graph)

            training_op_list = [accuracy, acc_summary_train, loss_summary_train]
            val_op_list = [accuracy, acc_summary_test]

            # 訓練の実行
            for step in range(args.max_steps):
                dataset.shuffle()  # バッチで取る前にデータセットをshuffleする
                # batch処理
                for i in range(int(len(dataset.train2_path) / args.batch_size)):  # iがbatchのindexになる #バッチのあまりが出る

                    batchA, batchB, labels = dataset.getTrainBatch(args.batch_size, i)

                    # バッチを学習
                    sess.run(train_step, feed_dict={images_placeholderA: batchA, images_placeholderB: batchB, labels_placeholder: labels,
                                                    keep_prob: args.dropout_prob})

                    # 最終バッチの処理
                    if i >= int(len(dataset.train2_path) / args.batch_size) - 1:

                        # 最終バッチの学習のあと，そのバッチを使って評価．
                        result = sess.run(training_op_list,
                                          feed_dict={images_placeholderA: batchA, images_placeholderB: batchB, labels_placeholder: labels,
                                                     keep_prob: 1.0})

                        # 必要なサマリーを追記
                        for j in range(1, len(result)):
                            summary_writer.add_summary(result[j], step)

                        print("step %d  training final-batch accuracy: %g" % (step, result[0]))

                        # validation
                        test_dataA, test_dataB, test_labels = dataset.getTestData()
                        val_result = sess.run(val_op_list,
                                              feed_dict={images_placeholderA: test_dataA, images_placeholderB: test_dataB, labels_placeholder: test_labels,
                                                         keep_prob: 1.0})

                        summary_writer.add_summary(val_result[1], step)

                        print("test accuracy %g" % val_result[0])

                        # print(vars_restored, vars_restored[0], sess.run(vars_restored[0]))  # 重みの取得

            # 最終的なモデルを保存
            save_path = saver.save(sess, args.save_path)
            print("save the trained model at :", save_path)


def main():
    parser = argparse.ArgumentParser(description='Learning your dataset, and evaluate the trained model')
    parser.add_argument('train1', help='File name of train data')
    parser.add_argument('--train2', help='File name of train data (subset)')
    parser.add_argument('test1', help='File name of train data')
    parser.add_argument('--test2', help='File name of train data (subset)')

    parser.add_argument('--max_steps', '-s', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=10)

    parser.add_argument('--save_path', '-save', default='/home/akalab/tensorflow_works/model.ckpt', help='FullPath of output model')
    parser.add_argument('--logdir', '-log', default='/tmp/data', help='Directory to put the training data. (TensorBoard)')

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.5)

    parser.add_argument('--model', '-m', default='/home/akalab/tensorflow_works/model.ckpt', help='FullPath of loading model')

    parser.add_argument('--flag', '-f', default='P', help='Train \"P\"rimary or \"S\"econdary')

    args = parser.parse_args()

    if args.flag == 'P':
        primaryTrain(args)
    elif args.flag == 'S':
        secondaryTrain(args)
    else:
        print("It is unknown Flag (plz assign flag P or S)")

if __name__ == '__main__':
    main()
