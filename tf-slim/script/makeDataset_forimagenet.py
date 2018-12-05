## affer running dataset_forFIT.py
import os
import argparse
import re
import shutil
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # tensorflow_works
import makeRandomCropping_NegativeData as negative


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('imagenet_dataset_path', type=str, help='full-path of the dataset')
    parser.add_argument('specific_dataset_path', type=str, help='full-path of the Specific-Object dataset. if you want to use multiple target, you can assign txt-file that described datasets-path. For more information, look line 25')
    parser.add_argument('-n', '--target_name', type=str, default='target', help='target object name.')
    # target.txt format: <dataset path> <catecory name>
    args = parser.parse_args()
    if '.txt' in args.specific_dataset_path:
        target_list = {f.split(" ")[0]: f.split(" ")[1].replace("\n", "") for f in open(args.specific_dataset_path)}

    else:
        target_list = {args.specific_dataset_path: args.target_name}

    # get dict, that shape is {day: video_paths}
    imagenet_dataset_path = args.imagenet_dataset_path

    class_dir = [f.name for f in os.scandir(path=args.imagenet_dataset_path) if f.is_dir()]
    print("imagenet_dataset:", imagenet_dataset_path)
    print("specific_object_dataset:", target_list)

    # <specific_object_dataset>/negativeをcpする．(存在しない場合は作成)
    if not "negative" in class_dir:
        if "negative" in [f.name for f in os.scandir(path=target_list.keys()[0]) if f.is_dir()]:
            shutil.copytree(target_list.keys()[0]+"/negative/negative_cropped", imagenet_dataset_path+"/negative")
            print("Copy negative set from dataset")


        else:
            # make negative-data in specific_object_dataset
            print("\n make negative data \n-----")
            from collections import namedtuple
            Negative = namedtuple("Negative", "dataset output negative")
            args_negative = Negative(target_list.keys()[0], imagenet_dataset_path, "negative")
            negative.main(args_negative)
            print("\n finished \n-----")

        class_dir.append("negative")

    class_dir.extend(target_list.values())
    class_dir = sorted(class_dir)


    print("classes:", class_dir)

    train_files = []
    valid_files = []
    test_files = []

    # random divide
    temp = []
    for c in class_dir:
        if c == "negtive" or c in target_list.values(): # negativeは別処理
            continue

        target_dir_path = os.path.join(imagenet_dataset_path, c)
        files = [os.path.join(target_dir_path, item.name) for item in os.scandir(path=target_dir_path) if
                 item.is_file() and re.search('.*\.[JPG|jpg]$', item.name)]
        temp.extend(files)

    train_files = sorted(random.sample(temp, int(len(temp)/2))) # ランダムで2分割
    valid_files = sorted(list(set(temp) - set(train_files)))
    train_files_dict = {c: [f for f in train_files if re.search(c, f)] for c in class_dir}
    valid_files_dict = {c: [f for f in valid_files if re.search(c, f)] for c in class_dir}

    # about negative
    negative_files = [i for i in find_all_files(os.path.join(imagenet_dataset_path, "negative", "negative_cropped")) if re.search('.*\.jpg$', i)]
    train_negative = random.sample(negative_files, int(len(negative_files) / 2))
    offset = sorted(list(set(negative_files) - set(train_negative)))
    valid_negative = random.sample(offset, int(len(offset) / 2))
    test_files = sorted(list(set(offset) - set(valid_negative)))

    train_files_dict['negative'].extend(train_negative)
    valid_files_dict['negative'].extend(valid_negative)
    test_files_dict = {c: [f for f in test_files if re.search(c, f)] for c in class_dir}


    # add specific object dataset
    for key in target_list.keys():
        for l in open(os.path.join(key, "train_crop.txt")):
            file = l.split(' ')[0]
            train_files_dict[target_list[key]].extend([file])

        for l in open(os.path.join(key, "valid_crop.txt")):
            file = l.split(' ')[0]
            valid_files_dict[target_list[key]].extend([file])

        for l in open(os.path.join(key, "test_crop.txt")):
            file = l.split(' ')[0]
            test_files_dict[target_list[key]].extend([file])


    for l in train_files_dict.values():
        for f in l:
            assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])
    for l in valid_files_dict.values():
        for f in l:
            assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])
    for l in test_files_dict.values():
        for f in l:
            assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])

    # write path
    with open(os.path.join(imagenet_dataset_path, "train.txt"), 'w') as f:
        for c in train_files_dict.keys():
            for i in train_files_dict[c]:
                f.writelines(i + " " + str(class_dir.index(c)) + '\n')

    with open(os.path.join(imagenet_dataset_path, "valid.txt"), 'w') as f:
        for c in valid_files_dict.keys():
            for i in valid_files_dict[c]:
                f.writelines(i + " " + str(class_dir.index(c)) + '\n')

    with open(os.path.join(imagenet_dataset_path, "test.txt"), 'w') as f:
        for c in test_files_dict.keys():
            for i in test_files_dict[c]:
                f.writelines(i + " " + str(class_dir.index(c)) + '\n')

    # write label.txt
    with open(os.path.join(imagenet_dataset_path, "label.txt"), 'w') as f:
        for n, c in enumerate(sorted(class_dir)):
            f.writelines(str(n) + ' ' + c + '\n')

