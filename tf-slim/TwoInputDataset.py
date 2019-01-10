import cv2
import numpy as np
from PIL import Image
import random

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TwoInputDistortion import Distortion
import utils

class TwoInputDataset():
    """
    読み込むtxtファイルは，
    <image1_path> class_number
    <image2_path> class_number
    みたいに記述しておく．
    """
    def __init__(self, **kwargs):
        self.train_path_c = []  ## cropped images
        self.train_path_o = []  ## full size images
        self.train_label_c = []  # [[0,0,0,1,0,0,0,0,0],...] みたいな感じ (1-of-k)
        self.test_path_c = []
        self.test_path_o = []
        self.test_label_c = []
        self.image_size = kwargs["image_size"]
        self.num_classes = kwargs["num_classes"]
        self.distortion = Distortion(gamma=2)


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

        # if train & test
        if len(kwargs) == 6 and kwargs["train_c"] is not None and kwargs["train_o"] is not None and kwargs["test_c"] is not None and kwargs["test_o"] is not None:
            train_c = kwargs["train_c"]
            train_o = kwargs["train_o"]
            test_c = kwargs["test_c"]
            test_o = kwargs["test_o"]

            self.train_path_c, self.train_label_c = getPathandLabel(train_c, self.num_classes)
            self.train_path_o, _ = getPathandLabel(train_o, self.num_classes)
            self.test_path_c, self.test_label_c = getPathandLabel(test_c, self.num_classes)
            self.test_path_o, _ = getPathandLabel(test_o, self.num_classes)

            #numpyにしておく
            self.train_label_c = np.asarray(self.train_label_c)
            self.test_label_c = np.asarray(self.test_label_c)

        # if only test
        elif len(kwargs) == 4 and kwargs["test_c"] is not None and kwargs["test_o"] is not None:
            # 引数
            test_c = kwargs["test_c"]
            test_o= kwargs["test_o"]

            self.test_path_c, self.test_label_c = getPathandLabel(test_c, self.num_classes)
            self.test_path_o, _                = getPathandLabel(test_o, self.num_classes)

            #numpyにしておく
            self.test_label_c = np.asarray(self.test_label_c)

        else:
            print("Dasaset initializer args error. (need 4 or 6 args)")
            sys.exit()

    def shuffle(self):
        # shuffle (dataとlabelの対応が崩れないように)
        def shuffle_data(paths_c, paths_o, labels):
            indexl = [i for i in range(len(paths_c))]
            shuffled_indexl = list(indexl)
            random.shuffle(shuffled_indexl)

            shuffled_c = paths_c
            shuffled_o = paths_o
            shuffled_labels = labels

            for i, (path_c, path_o, label) in enumerate(zip(paths_c, paths_o, labels)):
                shuffled_c[shuffled_indexl[i]] = path_c
                shuffled_o[shuffled_indexl[i]] = path_o
                shuffled_labels[shuffled_indexl[i]] = label

            return shuffled_c, shuffled_o, shuffled_labels

        if self.train_path_c:# 空じゃなければ
            self.train_path_c, self.train_path_o, self.train_label_c = shuffle_data(self.train_path_c, self.train_path_o, self.train_label_c)

        if self.test_path_c:
            self.test_path_c, self.test_path_o, self.test_path_c = shuffle_data(self.test_path_c, self.test_path_o, self.test_path_c)

        # # indexの対応関係が破壊されてないかの確認
        # for i, (test1, test2, label) in enumerate(zip(self.test_path_c, self.test_path_o, self.test_path_c)):
        #     print(i, test1, test2, label)

    def getBatch(self, batchsize, index, mode='train'):
        if mode == 'train':
            pathsA = self.train_path_c
            pathsB = self.train_path_o
            labels = self.train_label_c
        else:
            pathsA = self.test_path_c
            pathsB = self.test_path_o
            labels = self.test_label_c

        batchA = []
        batchB = []
        start = batchsize * index
        end = start + batchsize

        for i, (pathA, pathB) in enumerate(zip(pathsA[start:end], pathsB[start:end])):
            # pathB = pathsB[start+i]

            # imageA = cv2.imread(pathA)
            # imageB = cv2.imread(pathB)

            # # cv2.imread()はBGR
            imageA = np.array(Image.open(pathA)) / 255
            imageB = np.array(Image.open(pathB)) / 255
            #
            #
            # # imageA, imageB = self.distortion.distort(images=[imageA, imageB], flag=mode, p=1.0)
            #
            #imageA = cv2.resize(padding(imageA), (self.image_size, self.image_size))
            imageA = cv2.resize(imageA, (self.image_size, self.image_size))
            #imageB = cv2.resize(padding(imageB), (self.image_size, self.image_size))
            imageB = cv2.resize(imageB, (self.image_size, self.image_size))
            # imageA = utils.load_image(pathA, normalize=True)
            # imageB = utils.load_image(pathB, normalize=True)

            # batchA.append(imageA.astype(np.float32)/255.0)
            # batchB.append(imageB.astype(np.float32)/255.0)
            batchA.append(imageA)
            batchB.append(imageB)

            # batchA.append(imageA.flatten().astype(np.float32) / 255.0)
            # batchB.append(imageB.flatten().astype(np.float32) / 255.0)

        batchA = np.asarray(batchA)
        batchB = np.asarray(batchB)
        label_batch = labels[start:end]

        return {'batch': batchA, 'path': pathsA[start:end]}, {'batch': batchB, 'path': pathsB[start:end]}, label_batch

    def getTrainBatch(self, batchsize, index):
        return self.getBatch(batchsize, index, mode='train')


    def getTestData(self, batchsize, index=0):
        return self.getBatch(batchsize, index, mode='test')

def padding(image):
    # アス比の違う画像をゼロパディングして正方形にする
    w = image.shape[1]
    h = image.shape[0]
    if w == h:
        return image
    elif w > h:
        offset = w - h
        n = int(offset / 2)
        if offset % 2 == 0:
            dst = np.pad(image, [(n, n), (0, 0), (0, 0)], 'constant')
        else:
            dst = np.pad(image, [(n, n+1), (0, 0), (0, 0)], 'constant')
        return dst

    else:
        offset = h - w
        n = int(offset / 2)
        if offset % 2 == 0:
            dst = np.pad(image, [(0, 0), (n, n), (0, 0)], 'constant')
        else:
            dst = np.pad(image, [(0, 0), (n, n+1), (0, 0)], 'constant')
        return dst

if __name__ == '__main__':
    # # test code
    # dataset = TwoInputDataset(train_c='/Users/shigetomi/Desktop/dataset_walls/train2.txt',
    #                           train_o='/Users/shigetomi/Desktop/dataset_walls/train1.txt',
    #                           test_c='/Users/shigetomi/Desktop/dataset_walls/test2.txt',
    #                           test_o='/Users/shigetomi/Desktop/dataset_walls/test1.txt',
    #                           num_classes=6,
    #                           image_size=229)
    # dataset.shuffle()
    # cropped_batch, orig_batch, labels = dataset.getTrainBatch(batchsize=5, index=0)
    # cropped_test_batch, orig_test_batch, test_labels = dataset.getTestData(batchsize=5)
    #
    # # check image
    # for i,(c, o, l) in enumerate(zip(cropped_batch['batch'], orig_batch['batch'], labels)):
    #     print(i, cropped_batch['path'][i], '\n ', orig_batch['path'][i], '\n  class:', np.where(l == 1)[0][0])
    #     cv2.imshow("c", c)
    #     cv2.imshow("o", o)
    #     cv2.waitKey(0)

    # padding test
    image = np.array(Image.open("/Users/shigetomi/Desktop/dataset_GOR/cat/cat_cropped/image_0012.jpg")) / 255
    dst = cv2.resize(padding(image), (224, 224))
    print("image:", image)
    print("dst", dst)
    cv2.imshow("dst", dst)
    cv2.imshow("org", image)
    cv2.waitKey(0)
