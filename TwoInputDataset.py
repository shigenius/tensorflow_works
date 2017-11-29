import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import random

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
        if len(kwargs) == 5 and kwargs["train"] is not None and kwargs["train_sub"] is not None and kwargs["test"] is not None: #if train & test
            # 引数
            train = kwargs["train"]
            train_sub = kwargs["train_sub"]
            test = kwargs["test"]
            _num_classes = kwargs["num_classes"]
            _image_size = kwargs["image_size"]

            # メンバ変数
            self.train_image_paths = []
            self.train_sub_image_paths = [] ## subwindow
            self.train_labels = [] # [[0,0,0,1,0,0,0,0,0],...] みたいな感じ (1-of-k)
            self.test_image_paths = []
            self.test_labels = []
            self.image_size = _image_size
            self.num_classes = _num_classes

            self.train_image_paths, self.train_labels     = getPathandLabel(train, self.num_classes)
            self.train_sub_image_paths, self.train_labels = getPathandLabel(train_sub, self.num_classes)
            self.test_image_paths, self.test_labels       = getPathandLabel(test, self.num_classes)
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
            print("Dasaset initializer args error. (need 3 or 5 args)")

    def shuffle(self):
        # shuffle (dataとlabelの対応が崩れないように)
        # getTrainBatch()と同様に修正する必要あり．今はlen(sub)-len(main)で溢れた分は無視する方向で

        indexl = [i for i in range(len(self.train_sub_image_paths))]
        shuffled_indexl = list(indexl)
        random.shuffle(shuffled_indexl)

        shuffled_data = self.train_image_paths
        shuffled_sub_data = self.train_sub_image_paths
        shuffled_labels = self.train_labels

        for i, (sub_path, path, label) in enumerate(zip(self.train_sub_image_paths, self.train_image_paths, self.train_labels)):
            shuffled_sub_data[shuffled_indexl[i]] = sub_path
            shuffled_data[shuffled_indexl[i]] = path
            shuffled_labels[shuffled_indexl[i]] = label

        self.train_sub_image_paths = shuffled_sub_data
        self.train_image_paths = shuffled_data
        self.train_labels = shuffled_labels

        # print(self.train_image_paths, len(self.train_image_paths))
        # print(self.train_sub_image_paths, len(self.train_sub_image_paths))
        # print(len(self.train_labels))

        # indexの対応関係が破壊されてないかの確認
        #for i, (train, label) in enumerate(zip(self.train_image_paths, self.train_labels)):
        #    print(i, train, label)


    def getTrainBatch(self, batchsize, index):
        # 指定したindexからバッチサイズ分，データセットを読み込んでflattenなndarrayとして返す．resizeもする．
        # (あとでaugumentation諸々も実装したい)
        # subデータセットのフレーム抜けを考慮して書き直す必要あり (今はリストのindex関係が正常になっていることを想定している)

        train_batch = []
        train_sub_batch = []
        start = batchsize*index

        for i, path in enumerate(self.train_sub_image_paths[start:start+batchsize]):
            image = cv2.imread(path)
            image = cv2.resize(image, (self.image_size, self.image_size)) 
            image_sub = cv2.imread(self.train_image_paths[start+i])
            image_sub = cv2.resize(image_sub, (self.image_size, self.image_size)) 

            # 一列にした後、0-1のfloat値にする
            train_batch.append(image.flatten().astype(np.float32)/255.0)
            train_sub_batch.append(image_sub.flatten().astype(np.float32)/255.0)

        train_batch = np.asarray(train_batch)
        train_sub_batch = np.asarray(train_sub_batch)
        labels_batch = self.train_labels[start:start+batchsize]

        return train_batch, train_sub_batch, labels_batch
        