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
    parser.add_argument('-r', '--testsetrate', type=int, default=0.1, help='test sampling rate')
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

    train2_list = [args.dataset + "/" + c + "/" + v + "/" + i.name + " " + str(class_dir.index(c)) for c in class_dir
                   for v in video_dir[class_dir.index(c)] if re.search('_cropped', v) for i in
                   os.scandir(path=args.dataset + "/" + c + "/" + v) if i.is_file() and re.search('.jpg', i.name)]

    print(train1_list)
    print(train2_list)

    # cropped_video_dir/subwindow_log.txtから情報を取ってくる
    # params
    minimam_crop_size = 100 # 矩形の最小サイズ
    perimage = 5 # 一枚の画像からn枚crop画像を生成

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
                    # dict = {image_path : params}
                    break
        for j in range(perimage):
            I = cv2.imread(image_path) # ndarray dtype=uint8
            xsize = I.shape[1]
            ysize = I.shape[0]

            ### randam crop params
            center_x = random.uniform(0, xsize)
            center_y = random.uniform(0, ysize)
            # 小さい方に合わせる
            if xsize > ysize:
                max_crop_size = ysize/2
            else:
                max_crop_size = xsize/2
            cropsize = random.uniform(minimam_crop_size, max_crop_size)
            rect_x = center_x - cropsize/2 #矩形の始点(ひだりうえ)のx座標
            rect_y = center_y - cropsize/2

            # 正解データと被りすぎているデータを排除する処理をかく！

            # fillの処理
            padding_image = np.zeros((int(xsize+(cropsize*2)), int(xsize+(cropsize*2)), 3)).astype(np.uint8)
            padding_image[int(cropsize):int(cropsize + ysize), int(cropsize):int(cropsize + xsize)] = I # 余白をつける
            # print(padding_image[int(cropsize):int(cropsize + ysize),  int(cropsize):int(cropsize + xsize)])
            # print(padding_image.dtype)

            rectangle = padding_image[int(cropsize+rect_y): int(cropsize*2+rect_y), int(cropsize+rect_x): int(cropsize*2+rect_x)]
            cv2.imshow('padding_image', rectangle)
            cv2.waitKey(0)

            # imwriteの処理をかく


if __name__ == '__main__':
    main()
