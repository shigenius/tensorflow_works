#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import random

def main():
    parser = argparse.ArgumentParser(description='make dataset for specific object detection')
    parser.add_argument('dataset', help='Dataset path')
    parser.add_argument('-r', '--testsetrate' ,type=int, default=0.1 ,help='test sampling rate')
    args = parser.parse_args()

    class_dir = [f.name for f in os.scandir(path=args.dataset) if f.is_dir()]
    print("dataset:", args.dataset)
    print("classes:", class_dir)

    # label.txtをつくる
    with open(args.dataset + "/label.txt", 'w') as f:
        for c in class_dir:
            f.writelines(c + '\n')


    video_dir = [[f.name for f in os.scandir(path=args.dataset+"/"+c) if f.is_dir()] for c in class_dir]

    #image_list = [[[f.name for f in os.scandir(path=args.dataset+"/"+c+"/"+v) if f.is_file() and re.search('.jpg', f.name)] for v in video_dir[class_dir.index(c)]] for c in class_dir]
    #print("image_list:", image_list)

    # train1と2に分ける (2がcropしたやつ)
    train1_list = [args.dataset+"/"+c+"/"+v+"/"+i.name+" "+str(class_dir.index(c)) for c in class_dir for v in video_dir[class_dir.index(c)] if not re.search('_cropped', v) for i in os.scandir(path=args.dataset+"/"+c+"/"+v) if i.is_file() and re.search('.jpg', i.name)]
    train2_list = [args.dataset+"/"+c+"/"+v+"/"+i.name+" "+str(class_dir.index(c)) for c in class_dir for v in video_dir[class_dir.index(c)] if re.search('_cropped', v) for i in os.scandir(path=args.dataset+"/"+c+"/"+v) if i.is_file() and re.search('.jpg', i.name)]

    # ランダムでtest setをつくる
    test2 = sorted(random.sample(train2_list, int(len(train2_list)*args.testsetrate)))
    train2 = sorted(list(set(train2_list) - set(test2)))

    test1 = sorted(list(map(lambda x: re.sub(r'_cropped','',  x), test2)))
    assert set(test1) & set(train1_list) == set(test1) # test1を保障する
    train1 = sorted(list(map(lambda x: re.sub(r'_cropped','',  x), train2)))
    assert set(train1) <= set(train1_list) # 実際に存在するか
    #print("len(train2_list)", len(train2_list), "len(train1_list)", len(train1_list), "len(test2)",len(test2), "len(train2)",len(train2), "len(test1)", len(test1), "len(train1)", len(train1))
    

    with open(args.dataset + "/train1.txt", 'w') as f:
        for l in train1:
            f.writelines(l + '\n')

    with open(args.dataset + "/train2.txt", 'w') as f:
        for l in train2:
            f.writelines(l + '\n')

    with open(args.dataset + "/test1.txt", 'w') as f:
        for l in test1:
            f.writelines(l + '\n')

    with open(args.dataset + "/test2.txt", 'w') as f:
        for l in test2:
            f.writelines(l + '\n')

    print("process finished.")

if __name__ == '__main__':
    main()
