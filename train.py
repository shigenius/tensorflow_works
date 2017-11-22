#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import argparse

NUM_CLASSES = 10
IMAGE_SIZE = 227
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('train', 'train.txt', 'File name of train data')
# flags.DEFINE_string('test', 'test.txt', 'File name of train data')
# flags.DEFINE_string('train_dir', '/tmp/data', 'Directory to put the training data.')
# flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
# flags.DEFINE_integer('batch_size', 10, 'Batch size'
#                      'Must divide evenly into the dataset sizes.')
# flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
# 

parser = argparse.ArgumentParser(description='Learning original dataset')
parser.add_argument('train', help='File name of train data')
parser.add_argument('test', help='File name of train data')
parser.add_argument('--train_dir', '-dir', default='/tmp/data',
                        help='Directory to put the training data.')
parser.add_argument('--max_steps', '-s', type=int, default=100)
parser.add_argument('--batch_size', '-batch', type=float, default=5)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
args = parser.parse_args()

def inference(images_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数: 
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplace_holder

    返り値:
      y_conv: 各クラスの確率(のようなもの)
    """
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='VALID')
    
    # 入力を256x256x3に変形
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([11, 11, 3, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,4,4,1], padding="VALID") + b_conv1)
        print("h_conv1:", h_conv1.shape)
    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        print("h_pool1:", h_pool1.shape) 
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,2,2,1], padding="VALID") + b_conv2)
        print("h_conv2:", h_conv2.shape)
    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
        print("h_pool2:", h_pool2.shape)
    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([6*6*128, 128])
        b_fc1 = bias_variable([128])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 6*6*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([128, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv

def loss(logits, labels):
    """ lossを計算する関数

    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      cross_entropy: 交差エントロピーのtensor, float

    """

    # 交差エントロピーの計算
    # log(0) = NaN になる可能性があるので1e-10~1の範囲で正規化
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10,1)))
    # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    """ 訓練のOpを定義する関数

    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    返り値:
      train_step: 訓練のOp

    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数

    引数: 
      logits: inference()の結果
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      accuracy: 正解率(float)

    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

def main():

    # ファイルを開く
    f = open(args.train, 'r')
    # データを入れる配列
    train_image = []
    train_label = []
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # データを読み込んで256x256に縮小
        img = cv2.imread(l[0])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        # 一列にした後、0-1のfloat値にする
        train_image.append(img.flatten().astype(np.float32)/255.0)
        # ラベルを1-of-k方式で用意する
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        train_label.append(tmp)
    # numpy形式に変換
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label) 
    f.close()

    f = open(args.test, 'r')
    test_image = []
    test_label = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(l[0])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32)/255.0)
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        test_label.append(tmp)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()
    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        images_placeholder = tf.placeholder(dtype="float", shape=(None, IMAGE_PIXELS)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder(dtype="float", shape=(None, NUM_CLASSES))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder(dtype="float")

        # inference()を呼び出してモデルを作る
        logits = inference(images_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = training(loss_value, args.learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()
        # Sessionの作成
        sess = tf.Session()
        # 変数の初期化
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        # TensorBoardで表示する値の設定
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph_def)
        # 訓練の実行
        for step in range(args.max_steps):
            for i in range(int(len(train_image)/args.batch_size)):
                # batch_size分の画像に対して訓練の実行
                batch = args.batch_size*i
                # feed_dictでplaceholderに入れるデータを指定する
                #print(train_image[batch:batch+args.batch_size].shape)
                #print(train_label[batch:batch+args.batch_size].shape)
                # print("session run:", i)
                sess.run(train_op, feed_dict={
                  images_placeholder: train_image[batch:batch+args.batch_size],
                  labels_placeholder: train_label[batch:batch+args.batch_size],
                  keep_prob: 0.5})

            # 1 step終わるたびに精度を計算する
            
            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            print("step %d, training accuracy %g"%(step, train_accuracy))

            # 1 step終わるたびにTensorBoardに表示する値を追加する
            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

    # 訓練が終了したらテストデータに対する精度を表示
    print("test accuracy %g" % sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

    # 最終的なモデルを保存
    save_path = saver.save(sess, "/home/akalab/tensorflow_works/model.ckpt")
    print("save the trained model at :", save_path)
if __name__ == '__main__':
    main()
