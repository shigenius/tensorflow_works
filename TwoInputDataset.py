import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import random
import sys
import re

from TwoInputDistortion import distort
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
            self.train1_path = [] ## cropped images
            self.train2_path = [] ## full size images
            self.train1_label = [] # [[0,0,0,1,0,0,0,0,0],...] みたいな感じ (1-of-k)
            self.test1_path = []
            self.test2_path = []
            self.test1_label = []
            self.image_size = _image_size
            self.num_classes = _num_classes

            self.train1_path, self.train1_label = getPathandLabel(train1, self.num_classes)
            self.train2_path, _ = getPathandLabel(train2, self.num_classes)
            self.test1_path, self.test1_label = getPathandLabel(test1, self.num_classes)
            self.test2_path, _ = getPathandLabel(test2, self.num_classes)
            #numpyにしておく
            self.train1_label = np.asarray(self.train1_label)
            self.test1_label  = np.asarray(self.test1_label)

        elif len(kwargs) == 4 and kwargs["test1"] is not None and kwargs["test2"] is not None: #if only test
            # 引数
            test1 = kwargs["test1"]
            test2 = kwargs["test2"]
            _num_classes = kwargs["num_classes"]
            _image_size = kwargs["image_size"]

            # メンバ変数
            self.test1_path = []
            self.test2_path = []
            self.test1_label = []
            self.image_size = _image_size
            self.num_classes = _num_classes

            self.test1_path, self.test1_label = getPathandLabel(test1, self.num_classes)
            self.test2_path, _                = getPathandLabel(test2, self.num_classes)
            #numpyにしておく
            self.test1_label  = np.asarray(self.test1_label)

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
        shuffled_labels = self.train1_label

        for i, (train1, train2, label) in enumerate(zip(self.train1_path, self.train2_path, self.train1_label)):
            shuffled_data1[shuffled_indexl[i]] = train1
            shuffled_data2[shuffled_indexl[i]] = train2
            shuffled_labels[shuffled_indexl[i]] = label

        self.train1_path = shuffled_data1
        self.train2_path = shuffled_data2
        self.train1_label = shuffled_labels

        indexl = [i for i in range(len(self.test1_path))]
        shuffled_indexl = list(indexl)
        random.shuffle(shuffled_indexl)

        shuffled_data1 = self.test1_path
        shuffled_data2 = self.test2_path
        shuffled_labels = self.test1_label

        for i, (test1, test2, label) in enumerate(zip(self.test1_path, self.test2_path, self.test1_label)):
            shuffled_data1[shuffled_indexl[i]] = test1
            shuffled_data2[shuffled_indexl[i]] = test2
            shuffled_labels[shuffled_indexl[i]] = label

        self.test1_path = shuffled_data1
        self.test2_path = shuffled_data2
        self.test1_label = shuffled_labels

        # # indexの対応関係が破壊されてないかの確認
        # for i, (test1, test2, label) in enumerate(zip(self.test1_path, self.test2_path, self.test1_label)):
        #     print(i, test1, test2, label)

    def getBatch(self, batchsize, index, mode='train'):
        # 指定したindexからバッチサイズ分，データセットを読み込んでflattenなndarrayとして返す．resizeもする．
        # (あとでaugumentation諸々も実装したい)
        # train1,2のそれぞれの画像名で一致させる

        if mode == 'train':
            pathsA = self.train1_path
            pathsB = self.train2_path
            labels = self.train1_label
        else:
            pathsA = self.test1_path
            pathsB = self.test2_path
            labels = self.test1_label

        batchA = []
        batchB = []
        start = batchsize * index
        end = start + batchsize - 1

        for i, pathA in enumerate(pathsA[start:end]):
            pathB = pathsB[start+i]

            imageA = cv2.imread(pathA)
            imageB = cv2.imread(pathB)
            imageA, imageB = distort([imageA, imageB], flag=mdoe)

            imageA = cv2.resize(imageA, (self.image_size, self.image_size))
            imageB = cv2.resize(imageB, (self.image_size, self.image_size))

            # # for debugging
            # print(pathA, pathB)
            # cv2.imshow("imageA", imageA)
            # cv2.imshow("imageB", imageB)
            # cv2.waitKey(0)

            # 0-1のfloat値にする
            # batchA.append(imageA.astype(np.float32)/255.0)
            batchA.append(imageA.flatten().astype(np.float32) / 255.0)
            # batchB.append(imageB.astype(np.float32)/255.0)
            batchB.append(imageB.flatten().astype(np.float32) / 255.0)

        batchA = np.asarray(batchA)
        batchB = np.asarray(batchB)
        label_batch = labels[start:end]

        # # augumentation
        # if mode == 'train':
        #     datagen = ImageDataGenerator(rotation_range=10,
        #                                  width_shift_range=0.2,
        #                                  height_shift_range=0.2,
        #                                  fill_mode='constant',
        #                                  zoom_range=0.2)
        #     # datagen = ImageDataGenerator(zca_whitening=True)
        #     #datagen.fit(batchA) # zca whiteningを行う際に必要．実行するとなぜかzsh killedになる...
        #
        #     gA = datagen.flow(batchA, batch_size=batchsize)
        #     gB = datagen.flow(batchB, batch_size=batchsize)
        #     #gB = datagen.flow(batchB, batch_size=batchsize, save_to_dir='./temp', save_prefix='img', save_format='jpg')
        #     batchA = gA.next()
        #     batchB = gB.next()
        #     # # for debug
        #     # for i in range(batchsize-1):
        #     #     img = batchA[i]
        #     #     print(img)
        #     #     cv2.imshow("window", img)
        #     #     cv2.waitKey(0)

        # print(batchA.shape)
        # print(batchB.shape)
        # print(label_batch.shape)

        # batchA = np.reshape(batchA, (batchsize-1, -1))
        # batchB = np.reshape(batchB, (batchsize-1, -1))

        return batchA, batchB, label_batch

    def getTrainBatch(self, batchsize, index):
        return self.getBatch(batchsize, index, mode='train')

    """
    def getTestData(self, batchsize, index=0):
        return self.getBatch(batchsize, index, mode='test')
    """

    def getTestData(self):
        # testdataを全部とってくる
        # チェックする

        testA_images = []
        testB_images = []

        # print(self.test1_path, len(self.test1_path))
        # print(self.test2_path, len(self.test2_path))

        for pathA, pathB in zip(self.test1_path, self.test2_path):
            imageA = cv2.imread(pathA)
            imageB = cv2.imread(pathB)
            imageA, imageB = distort([imageA, imageB], flag='test')
            imageA = cv2.resize(imageA, (self.image_size, self.image_size))
            imageB = cv2.resize(imageB, (self.image_size, self.image_size))

            # # for debugging
            # print(pathA, pathB)
            # cv2.imshow("imageA", imageA)
            # cv2.imshow("imageB", imageB)
            # cv2.waitKey(0)

            testA_images.append(imageA.flatten().astype(np.float32)/255.0)
            testB_images.append(imageB.flatten().astype(np.float32)/255.0)
 
        testA_images = np.asarray(testA_images)
        testB_images = np.asarray(testB_images)
 
        return testA_images, testB_images, self.test1_label
