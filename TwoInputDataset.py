import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import random
import sys
import re

from Dataset import Dataset

class TwoInputDataset(Dataset):
    """
    読み込むtxtファイルは，
    <image1_path> class
    <image2_path> class
    みたいに記述しておく．
    """
    def __init__(self, **kwargs):

        def getPathandLabel(path, num_classes): # パスとラベルを取得する (inner function)
            with open(path, 'r') as f:
                f_ = [line.rstrip().split() for line in f]
                image_paths = [l[0] for l in f_]

                labels = [] # 1-of-kで用意する
                for l in f_:
                    tmp = [0]*num_classes
                    tmp[int(l[1])] = 1
                    labels.append(tmp) 

                return image_paths, labels

        # 引数の数とかで分岐
        if len(kwargs) == 6 and kwargs["train1"] is not None and kwargs["train2"] is not None and kwargs["test1"] is not None and kwargs["test2"] is not None: #if train & test
            # 引数
            train1 = kwargs["train1"]
            train2 = kwargs["train2"]
            test1 = kwargs["test1"]
            test2 = kwargs["test2"]
            _num_classes = kwargs["num_classes"]
            _image_size = kwargs["image_size"]

            # メンバ変数
            self.train1_path = []
            self.train2_path = [] ## subwindow
            self.train2_label = [] # [[0,0,0,1,0,0,0,0,0],...] みたいな感じ (1-of-k)
            self.test1_path = []
            self.test2_path = []
            self.test2_label = []
            self.image_size = _image_size
            self.num_classes = _num_classes

            self.train1_path, _                 = getPathandLabel(train1, self.num_classes)
            self.train2_path, self.train2_label = getPathandLabel(train2, self.num_classes)
            self.test1_path, _                  = getPathandLabel(test1, self.num_classes)
            self.test2_path, self.test2_label   = getPathandLabel(test2, self.num_classes)
            #numpyにしておく
            self.train2_label = np.asarray(self.train2_label)
            self.test2_label  = np.asarray(self.test2_label)

        elif len(kwargs) == 4 and kwargs["test1"] is not None and kwargs["test2"] is not None: #if only test
            # 引数
            test1 = kwargs["test1"]
            test2 = kwargs["test2"]
            _num_classes = kwargs["num_classes"]
            _image_size = kwargs["image_size"]

            # メンバ変数
            self.test1_path = []
            self.test2_path = []
            self.test2_label = []
            self.image_size = _image_size
            self.num_classes = _num_classes

            self.test1_path, _                = getPathandLabel(test1, self.num_classes)
            self.test2_path, self.test2_label = getPathandLabel(test2, self.num_classes)
            #numpyにしておく
            self.test2_label  = np.asarray(self.test2_label) 

        else:
            print("Dasaset initializer args error. (need 4 or 6 args)")
            sys.exit()

    def shuffle(self):
        # shuffle (dataとlabelの対応が崩れないように)

        indexl = [i for i in range(len(self.train1_path))]
        shuffled_indexl = list(indexl)
        random.shuffle(shuffled_indexl)

        shuffled_data1 = self.train1_path
        shuffled_data2 = self.train2_path
        shuffled_labels = self.train2_label

        for i, (train1, train2, label) in enumerate(zip(self.train1_path, self.train2_path, self.train2_label)):
            shuffled_data1[shuffled_indexl[i]] = train1
            shuffled_data2[shuffled_indexl[i]] = train2
            shuffled_labels[shuffled_indexl[i]] = label

        self.train1_path = shuffled_data1
        self.train2_path = shuffled_data2
        self.train2_label = shuffled_labels


        indexl = [i for i in range(len(self.test1_path))]
        shuffled_indexl = list(indexl)
        random.shuffle(shuffled_indexl)

        shuffled_data1 = self.test1_path
        shuffled_data2 = self.test2_path
        shuffled_labels = self.test2_label

        for i, (test1, test2, label) in enumerate(zip(self.test1_path, self.test2_path, self.test2_label)):
            shuffled_data1[shuffled_indexl[i]] = test1
            shuffled_data2[shuffled_indexl[i]] = test2
            shuffled_labels[shuffled_indexl[i]] = label

        self.test1_path = shuffled_data1
        self.test2_path = shuffled_data2
        self.test2_label = shuffled_labels

        # indexの対応関係が破壊されてないかの確認
        # for i, (test1, test2, label) in enumerate(zip(self.test1_path, self.test2_path, self.test2_label)):
        #     print(i, test1, test2, label)


    def getTrainBatch(self, batchsize, index):
        # 指定したindexからバッチサイズ分，データセットを読み込んでflattenなndarrayとして返す．resizeもする．
        # (あとでaugumentation諸々も実装したい)
        # train1,2のそれぞれの画像名で一致させる

        train1_batch = []
        train2_batch = []
        start = batchsize * index
        end = start + batchsize - 1

        for i, path1 in enumerate(self.train1_path[start:end]):
            path2 = self.train2_path[i]

            image1 = cv2.imread(path1)
            image2 = cv2.imread(path2)
            image1 = cv2.resize(image1, (self.image_size, self.image_size)) 
            image2 = cv2.resize(image2, (self.image_size, self.image_size)) 

            # 一列にした後、0-1のfloat値にする
            train1_batch.append(image1.flatten().astype(np.float32)/255.0)
            train2_batch.append(image2.flatten().astype(np.float32)/255.0)

        train1_batch = np.asarray(train1_batch)
        train2_batch = np.asarray(train2_batch)
        labels_batch = self.train2_label[start:end]

        return train1_batch, train2_batch, labels_batch


    # def getTestData(self):
    #     # testdataを全部とってくる
    #
    #     test1_images = []
    #     test2_images = []
    #     for path1, path2 in zip(self.test1_path, self.test2_path):
    #         image1 = cv2.imread(path1)
    #         image2 = cv2.imread(path2)
    #         image1 = cv2.resize(image1, (self.image_size, self.image_size))
    #         image2 = cv2.resize(image2, (self.image_size, self.image_size))
    #         test1_images.append(image1.flatten().astype(np.float32)/255.0)
    #         test2_images.append(image2.flatten().astype(np.float32)/255.0)
    #
    #     test1_images = np.asarray(test1_images)
    #     test2_images = np.asarray(test2_images)
    #
    #     return test1_images, test2_images, self.test2_label

    def getTestData(self, batchsize, index=0):
        test1_batch = []
        test2_batch = []
        start = batchsize * index
        end = start + batchsize - 1

        for i, path1 in enumerate(self.test1_path[start:end]):
            path2 = self.test2_path[i]

            image1 = cv2.imread(path1)
            image2 = cv2.imread(path2)
            image1 = cv2.resize(image1, (self.image_size, self.image_size))
            image2 = cv2.resize(image2, (self.image_size, self.image_size))

            # 一列にした後、0-1のfloat値にする
            test1_batch.append(image1.flatten().astype(np.float32)/255.0)
            test2_batch.append(image2.flatten().astype(np.float32)/255.0)

        test1_batch = np.asarray(test1_batch)
        test2_batch = np.asarray(test2_batch)
        labels_batch = self.test2_label[start:end]

        return test1_batch, test2_batch, labels_batch
        
