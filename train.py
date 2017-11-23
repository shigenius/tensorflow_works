#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import argparse
import random

class SimpleCNN:

    def __init__(self):
        self.image_size = 227
        self.num_classes = 10

    def inference(self, images_placeholder, keep_prob):
        # 重みを標準偏差0.1の正規分布で初期化
        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)
    
        # biasを標準偏差0.1の正規分布で初期化
        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)
    
        # pool層のテンプレ
        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                strides=[1, 2, 2, 1], padding='VALID')
        
        # 入力を227x227x3にresize
        x_image = tf.reshape(images_placeholder, [-1, self.image_size, self.image_size, 3])
        print("input: ", x_image.shape)
    
        # conv層1の作成
        with tf.name_scope('conv1') as scope:
            W_conv1 = weight_variable([11, 11, 3, 64])
            b_conv1 = bias_variable([64])
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,4,4,1], padding="VALID") + b_conv1)
            print("h_conv1:", h_conv1.shape)
        # pool層1の作成
        with tf.name_scope('pool1') as scope:
            h_pool1 = max_pool_2x2(h_conv1)
            print("h_pool1:", h_pool1.shape) 
        # conv層2の作成
        with tf.name_scope('conv2') as scope:
            W_conv2 = weight_variable([3, 3, 64, 128])
            b_conv2 = bias_variable([128])
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,2,2,1], padding="VALID") + b_conv2)
            print("h_conv2:", h_conv2.shape)
        # pool層2の作成
        with tf.name_scope('pool2') as scope:
            h_pool2 = max_pool_2x2(h_conv2)
            print("h_pool2:", h_pool2.shape)
        # fc1の作成
        with tf.name_scope('fc1') as scope:
            h_pool2_flat = tf.reshape(h_pool2, [-1, 6*6*128])

            W_fc1 = weight_variable([6*6*128, 256])
            b_fc1 = bias_variable([256])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            # dropoutの設定
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            print("h_fc1:", h_fc1.shape)
        # fc2の作成
        with tf.name_scope('fc2') as scope:
            W_fc2 = weight_variable([256, self.num_classes])
            b_fc2 = bias_variable([self.num_classes])
        # softmax関数による正規化
        with tf.name_scope('softmax') as scope:
            y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            print("y :", y_conv.shape)

        return y_conv

class Dataset:
    """
    読み込むtxtファイルは，
    <image1_path> class
    <image2_path> class
    みたいに記述しておく．
    """
    def __init__(self, train, test, _num_classes, _image_size):
        # メンバ変数
        self.train_image_paths = []
        self.train_labels = [] # [[0,0,0,1,0,0,0,0,0],...] みたいな感じ (1-of-k)
        self.test_image_paths = []
        self.test_labels = []
        self.image_size = _image_size
        self.num_classes = _num_classes

        def getPathandLabel(path, num_classes): # パスとラベルを取得する
            with open(path, 'r') as f:
                f_ = [line.rstrip().split() for line in f]
                image_paths = [l[0] for l in f_]

                labels = [] # 1-of-kで用意する
                for l in f_:
                    tmp = [0]*num_classes
                    tmp[int(l[1])] = 1
                    labels.append(tmp) 

                return image_paths, labels

        self.train_image_paths, self.train_labels = getPathandLabel(train, self.num_classes)
        self.test_image_paths, self.test_labels = getPathandLabel(test, self.num_classes)
        #numpyにしておく
        self.train_labels = np.asarray(self.train_labels) 
        self.test_labels  = np.asarray(self.test_labels) 


    def shuffle(self):
        # shuffle (dataとlabelの対応が崩れないように)

        indexl = [i for i in range(len(self.train_image_paths))]
        shuffled_indexl = list(indexl)
        random.shuffle(shuffled_indexl)

        shuffled_data = self.train_image_paths
        shuffled_labels = self.train_labels

        for i, (train, label) in enumerate(zip(self.train_image_paths, self.train_labels)):
            shuffled_data[shuffled_indexl[i]] = train
            shuffled_labels[shuffled_indexl[i]] = label

        self.train_image_paths = shuffled_data
        self.train_labels = shuffled_labels

        # indexの対応関係が破壊されてないかの確認
        #for i, (train, label) in enumerate(zip(self.train_image_paths, self.train_labels)):
        #    print(i, train, label)


    def getTrainBatch(self, batchsize, index):
        # 指定したindexからバッチサイズ分，データセットを読み込んでflattenなndarrayとして返す．resizeもする．
        # (あとでaugumentation諸々も実装したい)

        train_batch = []
        start = batchsize*index

        for path in self.train_image_paths[start:start+batchsize]:
            image = cv2.imread(path)
            image = cv2.resize(image, (self.image_size, self.image_size)) 

            # 一列にした後、0-1のfloat値にする
            train_batch.append(image.flatten().astype(np.float32)/255.0)

        train_batch = np.asarray(train_batch)
        labels_batch = self.train_labels[start:start+batchsize]

        return train_batch, labels_batch
        

    def getTestData(self):
        # testdataを全部とってくる

        test_images = []

        for path in self.test_image_paths:
            image = cv2.imread(path)
            image = cv2.resize(image, (self.image_size, self.image_size)) 

            test_images.append(image.flatten().astype(np.float32)/255.0)

        test_images = np.asarray(test_images)

        return test_images, self.test_labels


def loss(logits, labels):
    # 交差エントロピーの計算
    # log(0) = NaN になる可能性があるので1e-10~1の範囲で正規化
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10,1)))
    # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy


def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


def main():
    # NNアーキテクチャ．他のアーキテクチャを使いたかったらimportしてここに登録する．","を忘れずに
    archs = {
        'SimpleCNN': SimpleCNN
    }

    parser = argparse.ArgumentParser(description='Learning your dataset, and evaluate the trained model')
    parser.add_argument('train', help='File name of train data')
    parser.add_argument('test', help='File name of train data')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='SimpleCNN', help='Convnet architecture')

    parser.add_argument('--max_steps', '-s', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=10)

    parser.add_argument('--save_path', '-save', default='/home/akalab/tensorflow_works/model.ckpt', help='FullPath of output model')
    parser.add_argument('--train_dir', '-dir', default='/tmp/data', help='Directory to put the training data. (TensorBoard)')

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', '-d', type=float, default=0.5)
    args = parser.parse_args()

    arch = archs[args.arch]()
    
    #データセットの準備
    dataset = Dataset(args.train, args.test, arch.num_classes, arch.image_size)

    with tf.Graph().as_default():
        # imageとlabelを入れる仮のTensor
        images_placeholder = tf.placeholder(dtype="float", shape=(None, arch.image_size*arch.image_size*3)) # shapeの第一引数をNoneにすることによっておそらく可変長batchに対応できる
        labels_placeholder = tf.placeholder(dtype="float", shape=(None, arch.num_classes))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder(dtype="float")

        # モデルを作る
        logits = arch.inference(images_placeholder, keep_prob)
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
            #batch処理
            for i in range(int(len(dataset.train_image_paths)/args.batch_size)): # iがbatchのindexになる #バッチのあまりが出る
                batch, labels = dataset.getTrainBatch(args.batch_size, i)

                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                  images_placeholder: batch,
                  labels_placeholder: labels,
                  keep_prob: args.dropout_prob})

                # 最終バッチの処理
                if i >= int(len(dataset.train_image_paths)/args.batch_size)-1:
                    # 最終バッチの学習のあと，そのバッチを使って評価．毎step毎にデータセット全体をシャッフルしてるから多少は有効な値が取れそう(母集団に対して)
                    train_accuracy = sess.run(acc, feed_dict={
                    images_placeholder: batch,
                    labels_placeholder: labels,
                    keep_prob: 1.0})
                    print("step %d  training final-batch accuracy %g"%(step, train_accuracy))

                    # 1step終わるたびにTensorBoardに表示する値を追加する
                    summary_str = sess.run(summary_op, feed_dict={
                        images_placeholder: batch,
                        labels_placeholder: labels,
                        keep_prob: 1.0})
                    summary_writer.add_summary(summary_str, step)

        # testdataのaccuracy
        test_data, test_labels = dataset.getTestData()
        print("test accuracy %g" % sess.run(acc, feed_dict={
            images_placeholder: test_data,
            labels_placeholder: test_labels,
            keep_prob: 1.0}))

    # 最終的なモデルを保存
    save_path = saver.save(sess, args.save_path)
    print("save the trained model at :", save_path)

if __name__ == '__main__':
    main()
