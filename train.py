#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import argparse
import random

# めんば変数にする
NUM_CLASSES = 10
IMAGE_SIZE = 227
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

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

    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='VALID')
    
    # 入力を226x227x3に変形
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    print("input: ", x_image.shape)

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
        print("h_fc1:", h_fc1.shape)
    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([128, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        print("y :", y_conv.shape)
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

class Dataset:

    def __init__(self, train, test, num_classes):
        """
        読み込むtxtファイルは，
        <image1_path> class
        <image2_path> class
        みたいに記述しておく．
        """
        # メンバ変数
        self.train_image_paths = []
        self.train_labels = [] # [[0,0,0,1,0,0,0,0,0],...] みたいな感じ (1-of-k)
        self.test_image_paths = []
        self.test_labels = []

        # パスとラベルを取得する
        with open(train, 'r') as f:
            f_ = [line.rstrip().split() for line in f]
            self.train_image_paths = [l[0] for l in f_]

            for l in f_:
                tmp = [0]*num_classes
                tmp[int(l[1])] = 1
                self.train_labels.append(tmp) 

        self.train_labels = np.asarray(self.train_labels) #numpyにしておく
        
        with open(test, 'r') as f:
            f_ = [line.rstrip().split() for line in f]
            self.test_image_paths = [l[0] for l in f_]

            for l in f_:
                tmp = [0]*num_classes
                tmp[int(l[1])] = 1
                self.test_labels.append(tmp) 

        self.test_labels = np.asarray(self.test_labels) 

    def shuffle(self):
        # shuffle (dataとlabelの対応が崩れないように)

        lst = [i for i in range(len(self.train_image_paths))]
        shuffled_lst = list(lst)
        random.shuffle(shuffled_lst)

        shuffled_data = self.train_image_paths
        shuffled_labels = self.train_labels

        for i, (train, label) in enumerate(zip(self.train_image_paths, self.train_labels)):
            shuffled_data[shuffled_lst[i]] = train
            shuffled_labels[shuffled_lst[i]] = label

        self.train_image_paths = shuffled_data
        self.train_labels = shuffled_labels

        # 対応関係が破壊されてないかの確認用
        #for i, (train, label) in enumerate(zip(self.train_image_paths, self.train_labels)):
        #    print(i, train, label)


    def getTrainBatch(self, batchsize, index, image_size):
        """
        指定したindexからバッチサイズ分，データセットを読み込んでflattenなndarrayとして返す．(resizeもする．あとでaugumentationも実装したい)
        """

        train_batch = []
        start = batchsize*index

        for path in self.train_image_paths[start:start+batchsize]:
            image = cv2.imread(path)
            image = cv2.resize(image, (image_size, image_size)) 

            # 一列にした後、0-1のfloat値にする
            train_batch.append(image.flatten().astype(np.float32)/255.0)

        train_batch = np.asarray(train_batch)
        labels_batch = self.train_labels[start:start+batchsize]

        return train_batch, labels_batch
        

    def getTestData():
        pass

def main():
    parser = argparse.ArgumentParser(description='Learning your dataset, and evaluate the trained model')
    parser.add_argument('train', help='File name of train data')
    parser.add_argument('test', help='File name of train data')
    parser.add_argument('--train_dir', '-dir', default='/tmp/data',
                        help='Directory to put the training data. (TensorBoard)')
    parser.add_argument('--save_path', '-save', default='/home/akalab/tensorflow_works/model.ckpt',
                        help='FullPath of output model')
    parser.add_argument('--max_steps', '-s', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.5)
    args = parser.parse_args()

    num_classes = 10
    image_size = 227

    #データセットの準備
    dataset = Dataset(args.train, args.test, num_classes)
    #dataset.shuffle()

    with tf.Graph().as_default():
        # imageとlabelを入れる仮のTensor
        images_placeholder = tf.placeholder(dtype="float", shape=(None, image_size*image_size*3)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
        labels_placeholder = tf.placeholder(dtype="float", shape=(None, num_classes))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder(dtype="float")

        # モデルを作る
        logits = inference(images_placeholder, keep_prob)
        # lossを計算
        loss_value = loss(logits, labels_placeholder)
        # 訓練のoperation
        train_op = training(loss_value, args.learning_rate)
        # accyracyの計算
        acc = accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()
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

            for i in range(int(len(dataset.train_image_paths)/args.batch_size)): # iがbatchのindexになる
                batch, labels = dataset.getTrainBatch(args.batch_size, i, image_size)

                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                  images_placeholder: batch,
                  labels_placeholder: labels,
                  keep_prob: args.dropout_prob})

                # とりあえずのbatch accuracy．あとで消す
                train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: batch,
                labels_placeholder: labels,
                keep_prob: 1.0})
                print("step %d, batch %d,  training batch accuracy %g"%(step, i, train_accuracy))


            """ *保留* step毎にtrain accyracyをどうにか計算したいけど，全データセットをインスタンス化するのは無理．案1 : 最後のbatchのaccuracyをtrain accuracyとするみたいな．step毎にdataset.shuffle()すれば偏りはなくなるはず
            # 1step終わるたびにaccuracyを計算する
            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: dataset.train_image,
                labels_placeholder: dataset.train_label,
                keep_prob: 1.0})
            print("step %d, training accuracy %g"%(step, train_accuracy))

            # 1step終わるたびにTensorBoardに表示する値を追加する
            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)
            """
    """
    # testdataのaccuracy
    print("test accuracy %g" % sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))
    """
    # 最終的なモデルを保存
    save_path = saver.save(sess, args.save_path)
    print("save the trained model at :", save_path)


if __name__ == '__main__':
    main()
