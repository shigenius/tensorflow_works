#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import random

import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='make dataset for specific object detection')
    parser.add_argument('dataset', help='Dataset path')
    # parser.add_argument('-r', '--testsetrate', type=int, default=0.1, help='test sampling rate')
    args = parser.parse_args()

    class_dir = [f.name for f in os.scandir(path=args.dataset) if f.is_dir()]
    print("dataset:", args.dataset)
    print("classes:", class_dir)

    video_dir = [[f.name for f in os.scandir(path=args.dataset + "/" + c) if f.is_dir()] for c in class_dir]
    print("video dir:", video_dir)

    #image_list = [[[f.name for f in os.scandir(path=args.dataset+"/"+c+"/"+v) if f.is_file() and re.search('.jpg', f.name)] for v in video_dir[class_dir.index(c)]] for c in class_dir]
    #print("image_list:", image_list)

    train1_list = [args.dataset + "/" + c + "/" + v + "/" + i.name + " " + str(class_dir.index(c)) for c in class_dir
                   for v in video_dir[class_dir.index(c)] if not re.search('_cropped', v) for i in
                   os.scandir(path=args.dataset + "/" + c + "/" + v) if i.is_file() and re.search('.jpg', i.name)]

    # train2_list = [args.dataset + "/" + c + "/" + v + "/" + i.name + " " + str(class_dir.index(c)) for c in class_dir
    #                for v in video_dir[class_dir.index(c)] if re.search('_cropped', v) for i in
    #                os.scandir(path=args.dataset + "/" + c + "/" + v) if i.is_file() and re.search('.jpg', i.name)]

    # params
    MIN_CROP_SIZE = 100 # 矩形の最小サイズ
    CROPPING_PERIMAGE = 5 # 一枚の画像からn枚crop画像を生成

    pattern = r'([0-9]+)' # 数値抽出
    # ? : 0 or 1
    # + : 1 以上
    # * : 0 以上

    for text in train1_list:
        image_path = text.split(' ')[0]
        image_index = str(int(re.findall(pattern, image_path.split('/')[-1])[0])) # str

        logfile_path = '/'.join(image_path.split('/')[:-1]) + '_cropped/subwindow_log.txt'
        with open(logfile_path, 'r') as f:
            for row in f:
                if row.split(',')[0] == image_index:
                    params = [float(param) for param in row.split(',')[1:]]# center_x, center_y, size_x, size_y
                    corner_point = [{'x':int(params[0]-params[2]/2), 'y':int(params[1]-params[3]/2)},
                                    {'x':int(params[0]+params[2]/2), 'y':int(params[1]-params[3]/2)},
                                    {'x':int(params[0]-params[2]/2), 'y':int(params[1]+params[3]/2)},
                                    {'x':int(params[0]+params[2]/2), 'y':int(params[1]+params[3]/2)}]# [左上,右上,左下,右下]
                    # dict = {image_path : params}
                    break

        count = 0
        while(count < CROPPING_PERIMAGE):
            I = cv2.imread(image_path) # ndarray dtype=uint8

            # set randam crop params
            center_x, center_y, cropsize = generateRandamParameter(I, MIN_CROP_SIZE)
            rect_x = int(center_x - cropsize/2) #矩形の始点(ひだりうえ)のx座標
            rect_y = int(center_y - cropsize/2)

            cropped_corner_point = [{'x':rect_x,          'y':rect_y},
                                    {'x':rect_x+cropsize, 'y':rect_y},
                                    {'x':rect_x,          'y':rect_y+cropsize},
                                    {'x':rect_x+cropsize, 'y':rect_y+cropsize}]# [左上,右上,左下,右下]

            if isOverTheRectangle(cropped_corner_point, corner_point):
                continue

            # fillの処理
            padding_image =  paddingImage(I, cropsize)
            rectangle = padding_image[int(cropsize+rect_y): int(cropsize*2+rect_y), int(cropsize+rect_x): int(cropsize*2+rect_x)]

            # 確認用の画像表示
            cv2.rectangle(I, (rect_x, rect_y), (rect_x + cropsize, rect_y + cropsize), (255, 0, 0,), 3, 4)
            cv2.rectangle(I, (corner_point[0]['x'], corner_point[0]['y']), (corner_point[3]['x'], corner_point[3]['y']), (0, 0, 255,), 3, 4)
            cv2.imshow('padding_image', rectangle)
            cv2.imshow('org_image', I)
            cv2.waitKey(0)

            # imwriteとmkdirの処理をかく
            # 生成するnegative dataの数を指定する?あとから必要数を間引きするみたいな クラス間の画像数平均をとってもいいかも

            count += 1

def generateRandamParameter(org, min_crop_size):
    # input : input : orignal image(ndarray), min_cropsize (int)
    # output : randam center point, and randam cropsize
    xsize = org.shape[1]
    ysize = org.shape[0]

    center_x = random.uniform(0, xsize)
    center_y = random.uniform(0, ysize)
    # 小さい方に合わせる
    if xsize > ysize:
        max_crop_size = ysize / 2
    else:
        max_crop_size = xsize / 2

    cropsize = int(random.uniform(min_crop_size, max_crop_size))

    return center_x, center_y, cropsize

def isOverTheRectangle(target_rect, correct_rect):
    # 矩形が重なっていたらTrue, 重なっていなかったらFalse
    if (correct_rect[0]['x'] < target_rect[0]['x'] < correct_rect[1]['x'] and correct_rect[0]['y'] < target_rect[0]['y'] < correct_rect[2]['y']) or (correct_rect[0]['x'] < target_rect[1]['x'] < correct_rect[1]['x'] and correct_rect[0]['y'] < target_rect[1]['y'] < correct_rect[2]['y']) or (correct_rect[0]['x'] < target_rect[2]['x'] < correct_rect[1]['x'] and correct_rect[0]['y'] < target_rect[2]['y'] < correct_rect[2]['y']) or (correct_rect[0]['x'] < target_rect[3]['x'] < correct_rect[1]['x'] and correct_rect[0]['y'] < target_rect[3]['y'] < correct_rect[2]['y']):
        return True
    else:
        return False

def paddingImage(org, cropsize):
    # input : orignal image(ndarray), cropsize (int)
    xsize = org.shape[1]
    ysize = org.shape[0]

    padding_image = np.zeros((int(xsize + (cropsize * 2)), int(xsize + (cropsize * 2)), 3)).astype(np.uint8)
    padding_image[int(cropsize):int(cropsize + ysize), int(cropsize):int(cropsize + xsize)] = org  # 余白をつける
    return padding_image

if __name__ == '__main__':
    main()
