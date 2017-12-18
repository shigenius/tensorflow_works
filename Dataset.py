import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import random

class Dataset:
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
        if len(kwargs) == 4 and kwargs["train"] is not None and kwargs["test"] is not None: #if train & test
            # 引数
            train = kwargs["train"]
            test = kwargs["test"]
            _num_classes = kwargs["num_classes"]
            _image_size = kwargs["image_size"]

            # メンバ変数
            self.train_image_paths = []
            self.train_labels = [] # [[0,0,0,1,0,0,0,0,0],...] みたいな感じ (1-of-k)
            self.test_image_paths = []
            self.test_labels = []
            self.image_size = _image_size
            self.num_classes = _num_classes

            self.train_image_paths, self.train_labels = getPathandLabel(train, self.num_classes)
            self.test_image_paths, self.test_labels = getPathandLabel(test, self.num_classes)
            #numpyにしておく
            self.train_labels = np.asarray(self.train_labels) 
            self.test_labels  = np.asarray(self.test_labels)

        elif len(kwargs) == 3 and kwargs["test"] is not None: #if only test
            # 引数
            test = kwargs["test"]
            _num_classes = kwargs["num_classes"]
            _image_size = kwargs["image_size"]

            # メンバ変数
            self.test_image_paths = []
            self.test_labels = []
            self.image_size = _image_size
            self.num_classes = _num_classes

            self.test_image_paths, self.test_labels = getPathandLabel(test, self.num_classes)
            #numpyにしておく
            self.test_labels  = np.asarray(self.test_labels) 

        else:
            print("Dasaset initializer args error. (need 3 or 4 args)")

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
            if image is None:
                continue
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
            if image is None:
                continue
            image = cv2.resize(image, (self.image_size, self.image_size)) 

            test_images.append(image.flatten().astype(np.float32)/255.0)

        test_images = np.asarray(test_images)

        return test_images, self.test_labels

