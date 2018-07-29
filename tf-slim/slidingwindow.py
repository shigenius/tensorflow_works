import numpy as np
import cv2

def sliding_window(input, windowsize=128, stride=1):
    # input : cv2 single image (shape=x,y,ch)
    # output: cropped images (shape=batchsize,x,y,ch)

    return np.asarray([input[sy:sy+windowsize, sx:sx+windowsize, :]
                        for sy in range(0, input.shape[0]-windowsize, stride)
                            for sx in range(0, input.shape[1]-windowsize, stride)])


if __name__ == '__main__':
    # test
    path = '/Users/shigetomi/Desktop/dataset_shisas/shisa_engi1_l/IMG_0447/image_0002.jpg'
    src = cv2.imread(path)

    crops = sliding_window(src, windowsize=256, stride=100)

    print(crops.shape)
    cv2.imshow('input', src)
    for i in crops:
        cv2.imshow('croppes', i)
        cv2.waitKey(0)
