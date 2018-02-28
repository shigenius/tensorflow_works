#
import numpy as np
import random
import cv2

import tensorflow as tf


def random_brightness(image, max_delta=63):
    delta = np.random.uniform(-max_delta, max_delta)
    newimg = image + delta

    newimg = np.minimum(newimg, 255)
    newimg = np.maximum(newimg, 0)
    return newimg.astype('uint8')
    # return tf.image.random_brightness(image, max_delta=63, seed=None)

def random_contrast(image, max_a=1):
    pass

def gamma(image, gamma=2):
    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    img_src = cv2.imread('/Users/shigetomi/Desktop/samplepictures/0000065595.jpg', 1)
    return cv2.LUT(image, lookUpTable)

def normalize(img):
    return (img - np.mean(src)) / np.std(src) * 1 + 0

insize = 227
cropwidth = 256 - insize

def read_dist_image(image, flag='train', center=False, flip=False):
    #image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    # if center:
    #     top = left = cropwidth / 2
    # else:
    #     top = random.randint(0, cropwidth - 1)
    #     left = random.randint(0, cropwidth - 1)
    # bottom = insize+top
    # right = insize+left

    # clipping
    # image = image[:, top:bottom, left:right].astype(np.float32)

    # # left-right flipping
    # if flip and random.randint(0, 1) == 0:
    #     image =  image[:, :, ::-1]

    # augmentation
    if flag == 'train':
        if random.randint(0, 0) == 1:
            image = random_brightness(image)

        # write spatial augmentation process

    # filtering
    image = gamma(image, gamma=2) # gamma=2 暗部が持ち上がる
    # image = normalize(image) #dst image is float64
    return image

if __name__ == '__main__':
    # test code
    path = '/Users/shigetomi/Desktop/samplepictures/0000065595.jpg'
    # path = '/Users/shigetomi/Desktop/dataset_shisas/shisa_engi1_l/IMG_0447_cropped/image_0002.jpg'
    # src = cv2.imread(path).astype(np.float32)
    src = cv2.imread(path)

    # img = src * 1.2 + 40 # 輝度値が2倍になる
    # img = (img - np.mean(src)) / np.std(src) * 32 + 120 #標準偏差32,平均120に変更

    dst = read_dist_image(src)
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.imwrite("/Users/shigetomi/Desktop/1.png", dst)

    print(dst, dst.shape, dst.dtype)
    # cv2.imshow('dst', dst)
    cv2.waitKey(0)