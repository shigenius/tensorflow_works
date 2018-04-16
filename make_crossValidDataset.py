import os
import time
from datetime import datetime
import argparse
import re

def getymd(path):
    ctime_et = os.path.getmtime(path)
    # ctime_utc = datetime(*time.localtime(ctime_et)[:6]) # convert epoch time to utc
    ctime_utc = time.localtime(ctime_et)[:3]
    return ctime_utc

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_path', type=str, help='full-path of the dataset')
    args = parser.parse_args()

    # get dict that shape is {day: video_paths}
    dataset = args.dataset_path
    #dataset = "/Users/shigetomi/Desktop/dataset_roadsign"
    videos = [file for file in find_all_files(dataset) if re.compile(".MOV|.m4v|.mov|.mp4").search(os.path.splitext(file)[1])]
    ymd_list = [[v, getymd(v)] for v in videos]
    takingphoto_days = list(set([pair[1] for pair in ymd_list]))
    day_and_videos = {day: tuple([ymd[0] for ymd in ymd_list if ymd[1] == day]) for day in takingphoto_days}
    #print(day_and_videos)
    print("takingphoto days:", takingphoto_days)


    # create label.txt
    class_dir = [f.name for f in os.scandir(path=args.dataset_path) if f.is_dir()]
    print("dataset:", args.dataset_path)
    print("classes:", class_dir)
    with open(args.dataset_path + "/label.txt", 'w') as f:
        for n, c in enumerate(class_dir):
            f.writelines(str(n) + ' ' + c + '\n')

    # create train & test files for each day
    for day in day_and_videos:
        valid_files_crop = []
        for video in day_and_videos[day]:
            target_dir = os.path.splitext(video)[0]+'_cropped'
            files = [os.path.join(target_dir, item.name) for item in os.scandir(path=target_dir) if item.is_file() and re.search('.jpg', item.name)]
            valid_files_crop.extend(files)

        valid_files_withClassID_crop = sorted([f+" "+str(class_dir.index(c)) for c in class_dir for f in valid_files_crop if re.search(c, f)])
        valid_files_withClassID_orig = sorted(list(map(lambda x: re.sub(r'_cropped', '', x), valid_files_withClassID_crop)))

        # check absence
        for f in valid_files_withClassID_crop:
            assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])
        for f in valid_files_withClassID_orig:
            assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])

        #print(valid_files_withClassID_crop)
        #print(valid_files_withClassID_orig)

        train_files_crop = []
        for otherdays in day_and_videos:
            if otherdays != day:
                for video in day_and_videos[otherdays]:
                    target_dir = os.path.splitext(video)[0] + '_cropped'
                    files = [os.path.join(target_dir, item.name) for item in os.scandir(path=target_dir) if item.is_file() and re.search('.jpg', item.name)]
                    train_files_crop.extend(files)

        negative_sample_dir_path = args.dataset_path+"/negative/negative_cropped"
        negative_samples = [os.path.join(negative_sample_dir_path, item.name) for item in os.scandir(path=negative_sample_dir_path) if item.is_file() and re.search('.jpg', item.name)]
        train_files_crop.extend(negative_samples) # negative class

        #print(len(train_files_crop), train_files_crop)

        train_files_withClassID_crop = sorted([f + " " + str(class_dir.index(c)) for c in class_dir for f in train_files_crop if re.search(c, f)])
        train_files_withClassID_orig = sorted(list(map(lambda x: re.sub(r'_cropped', '', x), train_files_withClassID_crop)))

        # check absence
        for f in train_files_withClassID_crop:
            assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])
        for f in train_files_withClassID_orig:
            assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])

        # print(train_files_withClassID_crop)
        # print(train_files_withClassID_orig)
        day = ('{0:02d}'.format(day[0]), '{0:02d}'.format(day[1]), '{0:02d}'.format(day[2])) # zero padding
        temp_sentence = '_'.join(map(str, list(day)))

        with open(args.dataset_path + "/" + temp_sentence +"train_orig.txt", 'w') as f:
            for l in train_files_withClassID_orig:
                f.writelines(l + '\n')

        with open(args.dataset_path + "/" + temp_sentence +"train_crop.txt", 'w') as f:
            for l in train_files_withClassID_crop:
                f.writelines(l + '\n')

        with open(args.dataset_path + "/" + temp_sentence +"test_orig.txt", 'w') as f:
            for l in valid_files_withClassID_orig:
                f.writelines(l + '\n')

        with open(args.dataset_path + "/" + temp_sentence +"test_crop.txt", 'w') as f:
            for l in valid_files_withClassID_crop:
                f.writelines(l + '\n')

    print("process finished.")