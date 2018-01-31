#
import numpy as np
import random
import cv2

import tensorflow as tf


def random_brightness(image, max_delta=63, seed=None):
    # Args: image: An image tensor with 3 or more dimensions.

    # delta = np.random.uniform(-max_delta, max_delta)
    # newimg = image + delta
    # return newimg
    return tf.image.random_brightness(image, max_delta=63, seed=None)



def random_contrast(image, lower, upper, seed=None):
    f = np.random.uniform(-lower, upper)
    mean = (image[0] + image[1] + image[2]).astype(np.float32) / 3
    ximg = np.zeros(image.shape, np.float32)
    for i in range(0, 3):
        ximg[i] = (image[i] - mean) * f + mean
    return ximg

def image_whitening(img):
    img = img.astype(np.float32)
    d, w, h = img.shape
    num_pixels = d * w * h
    mean = img.mean()
    variance = np.mean(np.square(img)) - np.square(mean)
    stddev = np.sqrt(variance)
    min_stddev = 1.0 / np.sqrt(num_pixels)
    scale = stddev if stddev > min_stddev else min_stddev
    img -= mean
    img /= scale
    return img

insize = 227
cropwidth = 256 - insize

def read_dist_image(image, center=False, flip=False):
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

    # random brightness
    if random.randint(0, 0) == 0:
        image = random_brightness(image)

    # random contrast
    #if random.randint(0, 1) == 0:
    if random.randint(0, 0) == 1:
        image = random_contrast(image, lower=0.2, upper=1.8)

    # whitening
    # image = image_whitening(image)
    # image.flags.writeable = True
    return image

if __name__ == '__main__':
    # test code
    path = '/Users/shigetomi/Desktop/samplepictures/0000065595.jpg'
    org = cv2.imread(path).astype(np.float32) /255.0
    print(org, org.shape)

    # op
    random_brightness = tf.image.random_brightness(org, max_delta=63, seed=None)

    # run distortion
    with tf.Session() as sess:
        dst = sess.run(random_brightness)
        print(dst, dst.shape)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)