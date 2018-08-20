#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import random

def main():
    parser = argparse.ArgumentParser(description='make dataset for specific object detection')
    parser.add_argument('dataset', help='Dataset path')
    parser.add_argument('-r', '--testsetrate' ,type=int, default=0.2 ,help='sum of test and valid sampling rate from train')
    args = parser.parse_args()

    class_dir = sorted([f.name for f in os.scandir(path=args.dataset) if f.is_dir()])
    print("dataset:", args.dataset)
    print("classes:", class_dir)

    # label.txtをつくる
    with open(args.dataset + "/label.txt", 'w') as f:
        for c in class_dir:
            f.writelines(c + '\n')


    video_dir = [[f.name for f in os.scandir(path=args.dataset+"/"+c) if f.is_dir()] for c in class_dir]

    orig_images = [args.dataset+"/"+c+"/"+v+"/"+i.name+" "+str(class_dir.index(c)) for c in class_dir for v in video_dir[class_dir.index(c)] if not re.search('_cropped', v) for i in os.scandir(path=args.dataset+"/"+c+"/"+v) if i.is_file() and re.search('.jpg', i.name)]
    crop_images = [args.dataset+"/"+c+"/"+v+"/"+i.name+" "+str(class_dir.index(c)) for c in class_dir for v in video_dir[class_dir.index(c)] if re.search('_cropped', v) for i in os.scandir(path=args.dataset+"/"+c+"/"+v) if i.is_file() and re.search('.jpg', i.name)]

    # ランダムサンプリングでtest, valid, trainに分割
    num_sampling = int(len(crop_images)*args.testsetrate)

    test_crop = sorted(random.sample(crop_images, num_sampling))
    test_crop = [i for i in test_crop if re.search('^(?!.*\_d\d).*\.jpg$', i.split(" ")[0])]
    valid_crop = sorted(random.sample(test_crop, int(num_sampling/2)))
    test_crop = sorted(list(set(test_crop) - set(valid_crop)))
    train_crop = sorted(list(set(crop_images) - set(test_crop) - set(valid_crop)))

    test_orig = sorted(list(map(lambda x: re.sub(r'_cropped','',  x), test_crop)))
    assert set(test_orig) & set(orig_images) == set(test_orig) # test_origを保障する
    valid_orig = sorted(list(map(lambda x: re.sub(r'_cropped', '', x), valid_crop)))
    assert set(valid_orig) & set(orig_images) == set(valid_orig)
    train_orig = sorted(list(map(lambda x: re.sub(r'_cropped','',  x), train_crop)))
    assert set(train_orig) <= set(orig_images) # 実際に存在するか
    #print("len(crop_images)", len(crop_images), "len(orig_images)", len(orig_images), "len(test_crop)",len(test_crop), "len(train_crop)",len(train_crop), "len(test_orig)", len(test_orig), "len(train_orig)", len(train_orig))
    

    with open(args.dataset + "/train_orig.txt", 'w') as f:
        for l in train_orig:
            f.writelines(l + '\n')

    with open(args.dataset + "/train_crop.txt", 'w') as f:
        for l in train_crop:
            f.writelines(l + '\n')

    with open(args.dataset + "/valid_orig.txt", 'w') as f:
        for l in valid_orig:
            f.writelines(l + '\n')

    with open(args.dataset + "/valid_crop.txt", 'w') as f:
        for l in valid_crop:
            f.writelines(l + '\n')

    with open(args.dataset + "/test_orig.txt", 'w') as f:
        for l in test_orig:
            f.writelines(l + '\n')

    with open(args.dataset + "/test_crop.txt", 'w') as f:
        for l in test_crop:
            f.writelines(l + '\n')

    print("process finished.")

if __name__ == '__main__':
    main()
