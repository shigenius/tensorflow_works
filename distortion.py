#
import numpy as np
import random
import cv2
from scipy.ndimage.interpolation import rotate, shift

import tensorflow as tf


def random_brightness(image, max_delta=63):
    delta = np.random.uniform(-max_delta, max_delta)
    newimg = image + delta

    newimg = np.minimum(newimg, 255)
    newimg = np.maximum(newimg, 0)
    return newimg.astype('uint8')
    # return tf.image.random_brightness(image, max_delta=63, seed=None)

def random_contrast(img, range=(1, 5)):
    a = np.random.uniform(*range)
    newimg = (img - np.mean(img)) * a + 0
    newimg = np.minimum(newimg, 255)
    newimg = np.maximum(newimg, 0)
    return newimg.astype('uint8')

def gamma(image, gamma=2):
    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    return cv2.LUT(image, lookUpTable)

def normalize(img):
    return (img - np.mean(src)) / np.std(src) * 1 + 0

def random_shift(img, range = (-100, 100)):
    pass

def random_resize(img, range=(2, 8)):
    a = int(np.random.uniform(*range))
    dst = cv2.resize(img, None, fx=1/(2*a), fy=1/(2*a))
    return dst.astype('uint8')

def random_rotate(image, angle_range=(-10, 10)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = cv2.resize(image, (w, h))
    return image

def random_noise(img):
    row, col, ch = img.shape

    # 白
    pts_x = np.random.randint(0, col - 1, 1000)  # 0から(col-1)までの乱数を千個作る
    pts_y = np.random.randint(0, row - 1, 1000)
    img[(pts_y, pts_x)] = (255, 255, 255)  # y,xの順番になることに注意

    # 黒
    pts_x = np.random.randint(0, col - 1, 1000)
    pts_y = np.random.randint(0, row - 1, 1000)
    img[(pts_y, pts_x)] = (0, 0, 0)
    return img

def random_erasing(image_origin, s=(0.02, 0.4), r=(0.3, 3)):
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

def distort(image, flag='train', p=0.5):
    # filtering
    #image = gamma(image, gamma=2) # gamma=2 暗部が持ち上がる
    #image = normalize(image) #dst image is float64

    # augmentation
    if flag == 'train':
        if np.random.rand() > p:
            image = random_contrast(image)
        if np.random.rand() > p:
            image = random_brightness(image)
        if np.random.rand() > p:
            image = random_noise(image)
        if np.random.rand() > p:
            image = random_erasing(image)

        # spatial augmentation
        if np.random.rand() > p:
            image = random_resize(image)
        # if np.random.rand() > p:
        #     image = random_shift(image)
        if np.random.rand() > p:
            image = random_rotate(image)

    return image

if __name__ == '__main__':
    # test code
    path = '/Users/shigetomi/Desktop/samplepictures/image_0011.jpg'
    # path = '/Users/shigetomi/Desktop/dataset_shisas/shisa_engi1_l/IMG_0447_cropped/image_0002.jpg'
    # src = cv2.imread(path).astype(np.float32)
    src = cv2.imread(path)
    cv2.imshow('src', src)

    # img = src * 1.2 + 40 # 輝度値が2倍になる
    # img = (img - np.mean(src)) / np.std(src) * 32 + 120 #標準偏差32,平均120に変更

    dst = distort(src)
    cv2.imshow('dst', dst)
    cv2.imwrite("/Users/shigetomi/Desktop/1.png", dst)

    #print(src, src.shape, src.dtype)
    #print(dst, dst.shape, dst.dtype)
    # cv2.imshow('dst', dst)
    cv2.waitKey(0)