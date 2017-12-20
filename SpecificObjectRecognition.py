#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import argparse
import random

from Dataset import Dataset
from TwoInputDataset import TwoInputDataset
from TwostepCNN import PrimaryCNN, SecondaryCNN

def loss(logits, labels):
    # 交差エントロピーの計算
    # log(0) = NaN になる可能性があるので1e-10~1の範囲で正規化
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10,1)))
    # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    # 重みの更新
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


def primaryTrain(args) : 

    arch = PrimaryCNN()
    dataset = Dataset(train=args.train1, test=args.test1, num_classes=arch.num_classes, image_size=arch.image_size)
    
    with tf.Graph().as_default():
        # imageとlabelを入れる仮のTensor
        images_placeholder = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
        labels_placeholder = tf.placeholder(dtype="float", shape=(None, arch.num_classes))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder(dtype="float")
    
        # モデルを作る
        logits = arch.inference(images_placeholder, keep_prob)
        #print(tf.global_variables())
        saver = tf.train.Saver()
        # lossを計算
        loss_value = loss(logits, labels_placeholder)
        # 訓練のoperation
        train_op = training(loss_value, args.learning_rate)
        # accuracyの計算
        acc = accuracy(logits, labels_placeholder)
    
        # 保存の準備
        #saver = tf.train.Saver()
        # Sessionの作成
        sess = tf.Session()
        # 変数の初期化
        sess.run(tf.global_variables_initializer())
        # TensorBoardで表示する値の設定
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph_def)
    
        # 訓練の実行
        for step in range(args.max_steps):
            dataset.shuffle() # バッチで取る前にデータセットをshuffleする   
            #batch処理
            for i in range(int(len(dataset.train_image_paths)/args.batch_size)): # iがbatchのindexになる #バッチのあまりが出る

                batch, labels = dataset.getTrainBatch(args.batch_size, i)

                # 最終バッチの処理
                if i >= int(len(dataset.train_image_paths)/args.batch_size)-1:
                    # 最終バッチの学習のあと，そのバッチを使って評価．毎step毎にデータセット全体をシャッフルしてるから多少は有効な値が取れそう(母集団に対して)
                    train_accuracy = sess.run(acc, feed_dict={
                    images_placeholder: batch,
                    labels_placeholder: labels,
                    keep_prob: 1.0})
                    print("step %d  training final-batch accuracy %g"%(step, train_accuracy))
    
                    # testdataのaccuracy
                    test_data, test_labels = dataset.getTestData()
                    print("test accuracy %g" % sess.run(acc, feed_dict={
                    images_placeholder: test_data,
                    labels_placeholder: test_labels,
                    keep_prob: 1.0}))

                    # 1step終わるたびにTensorBoardに表示する値を追加する
                    summary_str = sess.run(summary_op, feed_dict={
                        images_placeholder: batch,
                        labels_placeholder: labels,
                        keep_prob: 1.0})
                    summary_writer.add_summary(summary_str, step)
    
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                  images_placeholder: batch,
                  labels_placeholder: labels,
                  keep_prob: args.dropout_prob})
    
    
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
            images_placeholder1 = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
            images_placeholder2 = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3))
            labels_placeholder = tf.placeholder(dtype="float", shape=(None, arch.num_classes))
            keep_prob = tf.placeholder(dtype="float")
    
            # 仮のinference (ckptに保存されている変数の数と，tf.train.Saver()を呼ぶ前に宣言する変数数を揃える必要があるため．) もうちょいいい書き方があるかもしれない
            P_logits = arch.inference(images_placeholder1, keep_prob)
    
            # restore
            saver = tf.train.Saver()
            saver.restore(sess, args.model)
            print("Model restored from : ", args.model)
            vars_restored = tf.global_variables() # restoreしてきたvariableのリスト
            
            logits = subarch.inference(images_placeholder1, images_placeholder2, keep_prob) # namescope "Primary/"以下はrestoreしたVariableになっている，はず

            # 初期化するvariableをもとめる
            vars_all = tf.global_variables()
            vars_F = list(set(vars_all) - (set(vars_restored))) # SecondaryCNNにのみ定義してあるvariable
            # print("\nvars_all", vars_all, len(vars_all))
            # print("\nvars_F", vars_F, len(vars_F))

            loss_value = loss(logits, labels_placeholder)
            train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_value, var_list=vars_F) # vars_Fのみloss用いて重みを更新する．
            acc = accuracy(logits, labels_placeholder)

            vars_all = tf.global_variables()
            vars_init = list(set(vars_all) - (set(vars_restored)))

            # variableの初期化．restoreしてきたものは初期化しない．
            sess.run(tf.variables_initializer(vars_init))

            #print(vars_all, vars_all[0], sess.run(vars_all[0])) # 重みの取得
    
            saver = tf.train.Saver() # 再びsaverを呼び出す
    
            # TensorBoardで表示する値の設定
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph_def)
    
            ### trainの処理 
    
            # 訓練の実行
            for step in range(args.max_steps):
                dataset.shuffle() # バッチで取る前にデータセットをshuffleする   
                #batch処理
                for i in range(int(len(dataset.train2_path)/args.batch_size)): # iがbatchのindexになる #バッチのあまりが出る
                    batch1, batch2, labels = dataset.getTrainBatch(args.batch_size, i)
    
                    # feed_dictでplaceholderに入れるデータを指定する
                    sess.run(train_op, feed_dict={
                      images_placeholder1: batch1,
                      images_placeholder2: batch2,
                      labels_placeholder: labels,
                      keep_prob: args.dropout_prob})
    
                    # 最終バッチの処理
                    if i >= int(len(dataset.train2_path)/args.batch_size)-1:
                        # 最終バッチの学習のあと，そのバッチを使って評価．毎step毎にデータセット全体をシャッフルしてるから多少は有効な値が取れそう(母集団に対して)
                        train_accuracy = sess.run(acc, feed_dict={
                        images_placeholder1: batch1,
                        images_placeholder2: batch2,
                        labels_placeholder: labels,
                        keep_prob: 1.0})
                        print("step %d  training final-batch accuracy %g"%(step, train_accuracy))
    
                        # 1step終わるたびにTensorBoardに表示する値を追加する
                        summary_str = sess.run(summary_op, feed_dict={
                            images_placeholder1: batch1,
                            images_placeholder2: batch2,
                            labels_placeholder: labels,
                            keep_prob: 1.0})
                        summary_writer.add_summary(summary_str, step)
            
            # testdataのaccuracy
            test1_data, test2_data, test_labels = dataset.getTestData()
            print("test accuracy %g" % sess.run(acc, feed_dict={
                images_placeholder1: test1_data,
                images_placeholder2: test2_data,
                labels_placeholder: test_labels,
                keep_prob: 1.0}))
            
            #print(vars_all[0], sess.run(vars_all[0])) # 重みの取得 (ちゃんとPrimaryのVariablesの重みがfixされてる？)
    
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
    parser.add_argument('--train_dir', '-dir', default='/tmp/data', help='Directory to put the training data. (TensorBoard)')

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
