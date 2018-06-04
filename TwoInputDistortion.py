#
import numpy as np
import random
import cv2
from scipy.ndimage.interpolation import rotate

import tensorflow as tf

class Distortion():
    def __init__(self, gamma=1.5):
        # create LUT for gamma
        self.lookUpTable = np.zeros((256, 1), dtype='uint8')
        for i in range(256):
            self.lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

        # ルックアップテーブルの生成
        min_table = 50
        max_table = 205
        diff_table = max_table - min_table

        self.LUT_HC = np.arange(256, dtype='uint8')
        self.LUT_LC = np.arange(256, dtype='uint8')

        # ハイコントラストLUT作成
        for i in range(0, min_table):
            self.LUT_HC[i] = 0
        for i in range(min_table, max_table):
            self.LUT_HC[i] = 255 * (i - min_table) / diff_table
        for i in range(max_table, 255):
            self.LUT_HC[i] = 255

        # ローコントラストLUT作成
        for i in range(256):
            self.LUT_LC[i] = min_table + i * (diff_table) / 255


    def gamma(self, imgs=[]):
        dsts = []
        for img in imgs:
            dst = cv2.LUT(img, self.lookUpTable)
            dsts.append(dst)

        return dsts

    def normalize(self, img):
        return (img - np.mean(src)) / np.std(src) * 1 + 0

    def random_brightness(self, imgs = []):
        # delta = np.random.uniform(-max_delta, max_delta)
        delta = np.random.normal(0, 20)
        #  np.random.normal(50,1,10000)標準正規乱数(平均50, 分散1) に従う乱数を10000万件出力
        # # ヒストグラムの確認用
        # import matplotlib.pyplot as plt
        # plt.hist(R, bins=100)
        # plt.show()

        dsts = []
        for img in imgs:
            dst = img + delta
            dst = np.minimum(dst, 255)
            dst = np.maximum(dst, 0)

            dsts.append(dst)

        return dsts

    def random_contrast(self, imgs = []):
        # a = np.random.uniform(0.2, 1.8)
        # dsts = []
        # for img in imgs:
        #     dst = (img - np.mean(img)) * a + 0
        #     # dst = (img - 127.5) * a + 0
        #     dst = np.minimum(dst, 255)
        #     dst = np.maximum(dst, 0)
        #
        #     dsts.append(dst.astype('uint8'))
        #
        # return dsts

        dsts = []
        p = np.random.rand()
        for img in imgs:
            if p < 0.2:
                dst = cv2.LUT(img, self.LUT_HC)
            elif p > 0.8:
                dst = cv2.LUT(img, self.LUT_LC)
            else:
                dst = img

            dsts.append(dst)

        return dsts

    def random_resize(self, imgs = [], range=(2, 8)):
        a = int(np.random.uniform(*range))

        dsts = []
        for img in imgs:
            dst = cv2.resize(img, None, fx=1/(2*a), fy=1/(2*a))
            dsts.append(dst)

        return dsts

    def random_rotate(self, imgs = [], mean_variance=(0, 5)):
        angle = np.random.normal(*mean_variance)

        dsts = []
        for img in imgs:
            h, w, _ = img.shape
            dst = rotate(img, angle)
            dst = cv2.resize(dst, (w, h))
            dsts.append(dst)

        return dsts

    def random_noise(self, img, num_noise = 1000):
        row, col, ch = img.shape

        # 白
        pts_x = np.random.randint(0, col - 1, num_noise)  # 0から(col-1)までの乱数をnum_noise個作る
        pts_y = np.random.randint(0, row - 1, num_noise)
        img[(pts_y, pts_x)] = (255, 255, 255)  # y,xの順番になることに注意

        # 黒
        pts_x = np.random.randint(0, col - 1, num_noise)
        pts_y = np.random.randint(0, row - 1, num_noise)
        img[(pts_y, pts_x)] = (0, 0, 0)
        return img

    def gaussian_noise(self, imgs):
        gauss_img = []
        for img in imgs:
            row, col, ch = img.shape
            mean = 0
            sigma = 10
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            # gauss = gauss.reshape(row, col, ch)
            dst = img + gauss
            dst = np.minimum(dst, 255)
            dst = np.maximum(dst, 0)
            gauss_img.append(dst)

        return gauss_img

    def random_erasing(self, image_origin, s=(0.02, 0.2), r=(0.3, 3)):
        img = np.copy(image_origin)

        # マスクする画素値をランダムで決める
        mask_value = np.random.randint(0, 256)

        h, w, _ = img.shape
        # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
        mask_area = np.random.randint(h * w * s[0], h * w * s[1])

        # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
        mask_aspect_ratio = np.random.rand() * r[1] + r[0]

        # マスクのサイズとアスペクト比からマスクの高さと幅を決める
        # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width
        img[top:bottom, left:right, :].fill(mask_value)
        return img

    def distort(self, images = [], flag='train', p=0.5):
        # augmentation
        if flag == 'train':
            if np.random.rand() < p:
                images = self.random_contrast(images)
            if np.random.rand() < p:
                images = self.gaussian_noise(images)
            if np.random.rand() < p:
                images = self.random_brightness(images)

            # if np.random.rand() < p:
            #     images[0] = self.random_noise(images[0], num_noise=50)
            #     images[1] = self.random_noise(images[1], num_noise=100)
            # if np.random.rand() < p:
            #     images = random_erasing(images)

            # spatial augmentation
            # if np.random.rand() < p:
            #     images = random_resize(images)
            if np.random.rand() < p:
                images = self.random_rotate(images)


        # filtering
        # images = self.gamma(images, gamma=2.0) # gamma=2 暗部が持ち上がる
        #image = self.normalize(image) #dst image is float64

        # # regularize
        # for i, image in enumerate(images):
        #     images[i] = np.minimum(image, 255)
        #     images[i] = np.maximum(image, 0)

        return images[0].astype('uint8'), images[1].astype('uint8')

if __name__ == '__main__':
    # test code
    #path = '/Users/shigetomi/Desktop/samplepictures/image_0011.jpg'
    pathA = '/Users/shigetomi/Desktop/dataset_shisas/shisa_engi1_l/IMG_0447_cropped/image_0002.jpg'
    pathB = '/Users/shigetomi/Desktop/dataset_shisas/shisa_engi1_l/IMG_0447/image_0002.jpg'
    # src = cv2.imread(path).astype(np.float32)
    srcA = cv2.imread(pathA)
    srcB = cv2.imread(pathB)
    cv2.imshow('srcA', srcA)
    cv2.imshow('srcB', srcB)
    distortion = Distortion(gamma=2)
    dstA, dstB = distortion.distort(images=[srcA, srcB], flag='train', p=1.0)
    cv2.imshow('dstA', dstA)
    cv2.imshow('dstB', dstB)
    #cv2.imwrite("/Users/shigetomi/Desktop/1.png", dst)

    print("srcA", srcA)
    print("dstA", dstA)
    # cv2.imshow('dst', dst)
    cv2.waitKey(0)