#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import random

import cv2
import numpy as np
import csv

def main():
    parser = argparse.ArgumentParser(description='make dataset for specific object detection')
    parser.add_argument('dataset', help='Dataset path')
    parser.add_argument('-n', '--negative', type=str, default='negative', help='name of negative sample directory')
    args = parser.parse_args()

    class_dir = [f.name for f in os.scandir(path=args.dataset) if f.is_dir() and not re.search(args.negative, f.name)]
    print("dataset:", args.dataset)
    print("classes:", class_dir)

    video_dir = [[f.name for f in os.scandir(path=args.dataset + "/" + c) if f.is_dir()] for c in class_dir]
    print("video dir:", video_dir)


    image_list = [[[f.name for f in os.scandir(path=args.dataset+"/"+c+"/"+v) if f.is_file() and re.search('.jpg', f.name)] for v in video_dir[class_dir.index(c)]] for c in class_dir]
    n_image_per_class = [0]*len(class_dir)
    for i in range(len(image_list)):
        for j in range(len(image_list[i])):
            n_image_per_class[i] += len(image_list[i][j])


    # print("n_image_per_class", n_image_per_class)
    mean = sum(n_image_per_class)/len(n_image_per_class)
    n_negative_sample = int(mean/2)
    print("number of maked negative samples :", n_negative_sample)

    # train1_list = [args.dataset + "/" + c + "/" + v + "/" + i.name + " " + str(class_dir.index(c)) for c in class_dir
    #                for v in video_dir[class_dir.index(c)] if not re.search('_cropped', v) for i in
    #                os.scandir(path=args.dataset + "/" + c + "/" + v) if i.is_file() and re.search('.jpg', i.name)]

    train2_list = [args.dataset + "/" + c + "/" + v + "/" + i.name + " " + str(class_dir.index(c)) for c in class_dir
                   for v in video_dir[class_dir.index(c)] if re.search('_cropped', v) for i in
                   os.scandir(path=args.dataset + "/" + c + "/" + v) if i.is_file() and re.search('.jpg', i.name)]

    # params
    MIN_CROP_SIZE = 100 # 矩形の最小サイズ
    CROPPING_PERIMAGE = 1 # 一枚の画像からn枚crop画像を生成

    if not os.path.exists(args.dataset + "/" + args.negative + "/negative"):
        os.makedirs(args.dataset + "/" + args.negative + "/negative")
        os.makedirs(args.dataset + "/" + args.negative + "/negative_cropped")

    log = open(args.dataset + "/" + args.negative + "/negative_cropped" + "/subwindow_log.txt", 'w')
    writer = csv.writer(log, lineterminator='\n')  # 改行コード（\n）を指定しておく
    writer.writerow(("frame", "center_x", "center_y", "size_x", "size_y"))

    for n, text in enumerate(random.sample(train2_list, n_negative_sample), 1):
        image_path = text.split(' ')[0].replace('_cropped', '')
        corner_point = getCorrectCornerFromLog(image_path)

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

            cv2.imwrite(args.dataset + "/" + args.negative + "/negative/" + "image_" + '%04d' % n + ".jpg", I)
            cv2.imwrite(args.dataset + "/" + args.negative + "/negative_cropped/" + "image_" + '%04d' % n + ".jpg", rectangle)
            writer.writerow((n, center_x, center_y, cropsize, cropsize))

            # 確認用の画像表示
            # cv2.rectangle(I, (rect_x, rect_y), (rect_x + cropsize, rect_y + cropsize), (255, 0, 0,), 3, 4)
            # cv2.rectangle(I, (corner_point[0]['x'], corner_point[0]['y']), (corner_point[3]['x'], corner_point[3]['y']), (0, 0, 255,), 3, 4)
            # cv2.imshow('padding_image', rectangle)
            # cv2.imshow('org_image', I)
            # cv2.waitKey(0)

            count += 1

def getCorrectCornerFromLog(image_path):
    # input : image_path
    pattern = r'([0-9]+)'  # 数値抽出
    image_index = str(int(re.findall(pattern, image_path.split('/')[-1])[0]))  # str

    logfile_path = '/'.join(image_path.split('/')[:-1]) + '_cropped/subwindow_log.txt'
    with open(logfile_path, 'r') as f:
        for row in f:
            if row.split(',')[0] == image_index:
                params = [float(param) for param in row.split(',')[1:]]  # center_x, center_y, size_x, size_y
                p = [{'x': int(params[0] - params[2] / 2), 'y': int(params[1] - params[3] / 2)},
                     {'x': int(params[0] + params[2] / 2), 'y': int(params[1] - params[3] / 2)},
                     {'x': int(params[0] - params[2] / 2), 'y': int(params[1] + params[3] / 2)},
                     {'x': int(params[0] + params[2] / 2), 'y': int(params[1] + params[3] / 2)}]  # [左上,右上,左下,右下]

                return p

def generateRandamParameter(org, min_crop_size):
    # input : orignal image(ndarray), min_cropsize (int)
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
