#
import numpy as np
import random
import cv2
from scipy.ndimage.interpolation import rotate

import tensorflow as tf


def random_brightness(imgs = [], max_delta=63):
    delta = np.random.uniform(-max_delta, max_delta)

    dsts = []
    for img in imgs:
        dst = img + delta
        dst = np.minimum(dst, 255)
        dst = np.maximum(dst, 0)

        dsts.append(dst.astype('uint8'))

    return dsts
    # return tf.image.random_brightness(image, max_delta=63, seed=None)

def random_contrast(imgs = [], range=(1, 5)):
    a = np.random.uniform(*range)

    dsts = []
    for img in imgs:
        dst = (img - np.mean(img)) * a + 0
        dst = np.minimum(dst, 255)
        dst = np.maximum(dst, 0)

        dsts.append(dst.astype('uint8'))

    return dsts

def gamma(imgs = [], gamma=2):
    # create LUT
    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    dsts = []
    for img in imgs:
        dst = cv2.LUT(img, lookUpTable)
        dsts.append(dst)

    return dsts

def normalize(img):
    return (img - np.mean(src)) / np.std(src) * 1 + 0

def random_resize(imgs = [], range=(2, 8)):
    a = int(np.random.uniform(*range))

    dsts = []
    for img in imgs:
        dst = cv2.resize(img, None, fx=1/(2*a), fy=1/(2*a))
        dsts.append(dst.astype('uint8'))

    return dsts

def random_rotate(imgs = [], angle_range=(-10, 10)):
    angle = np.random.randint(*angle_range)

    dsts = []
    for img in imgs:
        h, w, _ = img.shape
        dst = rotate(img, angle)
        dst = cv2.resize(dst, (w, h))
        dsts.append(dst)

    return dsts

def random_noise(img, num_noise = 1000):
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

def random_erasing(image_origin, s=(0.02, 0.2), r=(0.3, 3)):
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

def distort(images = [], flag='train', p=0.5):
    # augmentation
    if flag == 'train':
        if np.random.rand() > p:
            images = random_contrast(images)
        if np.random.rand() > p:
            images = random_brightness(images)
        if np.random.rand() > p:
            images[0] = random_noise(images[0], num_noise = 50)
            images[1] = random_noise(images[1], num_noise = 1000)
        # if np.random.rand() > p:
        #     images = random_erasing(images)

        # spatial augmentation
        # if np.random.rand() > p:
        #     images = random_resize(images)
        if np.random.rand() > p:
            images = random_rotate(images)

    # filtering
    images = gamma(images, gamma=2.0) # gamma=2 暗部が持ち上がる
    #image = normalize(image) #dst image is float64

    return images[0], images[1]

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

    dstA, dstB = distort([srcA, srcB], p=0.5)
    cv2.imshow('dstA', dstA)
    cv2.imshow('dstB', dstB)
    #cv2.imwrite("/Users/shigetomi/Desktop/1.png", dst)

    print("srcA", srcA)
    print("dstA", dstA)
    # cv2.imshow('dst', dst)
    cv2.waitKey(0)