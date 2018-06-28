import os
import time
from datetime import datetime
import argparse
import re
import subprocess

def getymd(path):
    # Unix only
    ctime_et = os.path.getmtime(path)
    # ctime_utc = datetime(*time.localtime(ctime_et)[:6]) # convert epoch time to utc
    ctime_utc = time.localtime(ctime_et)[:3]
    return ctime_utc

def getCreateTime(videopath):
    # dependent with ffmpeg
    proc = subprocess.Popen(['ffmpeg', '-i', videopath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_value = proc.communicate()[1]
    #print('\tstdout:', repr(stdout_value).split("\\n"))

    pattern = r"creation_time.*"
    ctime = [text for text in repr(stdout_value).split("\\n") if re.compile(pattern).search(text)]
    ctime = re.compile("\d+\-\d+\-\d+.*[^\.]+").search(ctime[0]).group()
    ctime = tuple(re.split('[.]', ctime))[0]
    ctime = re.sub(r'[a-zA-Z]+', " ", ctime)
    # ctime = tuple(map(lambda x: int(x), ctime))
    # print(ctime)
    tdatetime = datetime.strptime(ctime, '%Y-%m-%d %H:%M:%S')
    # print(tdatetime)
    # print(tdatetime.timestamp())
    return tdatetime

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def get_luminous_condition_cluster(dataset_path):
    videos = [file for file in find_all_files(dataset_path) if re.compile(".MOV|.m4v|.mov|.mp4").search(os.path.splitext(file)[1])]
    video_date = {v: getCreateTime(v) for v in videos}
    # print(video_date)
    luminous_cluster = []
    for key in video_date.keys():
        # print(key, video_date[key])
        previous = video_date[key].replace(hour=video_date[key].hour - 1)
        following = video_date[key].replace(hour=video_date[key].hour + 1)
        # print(previous)
        # print(following)
        # print(previous.timestamp())
        # print(following.timestamp())
        for i, l in enumerate(luminous_cluster):
            for item in l:
                if previous.timestamp() < l[item].timestamp() < following.timestamp():
                    luminous_cluster[i].update({key: video_date[key]})
                    break
            else: # loopの中でbreakされなかった場合
                continue
            break # loopの中でbreakされた場合break


        if key not in [item for sublist in luminous_cluster for item in sublist]:# flatten
            # print("append!")
            luminous_cluster.append({key: video_date[key]})

    return luminous_cluster

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_path', type=str, help='full-path of the dataset')
    # parser.add_argument('--list', '-l', action='store_const', const=True, default=False)
    args = parser.parse_args()

    # get dict that shape is {day: video_paths}
    dataset_path = args.dataset_path

    # create label.txt
    class_dir = [f.name for f in os.scandir(path=args.dataset_path) if f.is_dir()]
    print("dataset:", args.dataset_path)
    print("classes:", class_dir)
    with open(args.dataset_path + "/label.txt", 'w') as f:
        for n, c in enumerate(class_dir):
            f.writelines(str(n) + ' ' + c + '\n')

    luminous_cluster = get_luminous_condition_cluster(dataset_path)

    for i, item in enumerate(luminous_cluster):
        print(i, item.values())



    trainidx = int(input('please select train\'s index>'))

    # print(luminous_cluster[trainidx])

    train_files_crop = []
    train_files_orig = []
    valid_files_crop = []
    valid_files_orig = []
    test_files_crop = []
    test_files_orig = []

    # create train set
    for video in luminous_cluster[trainidx]:
        target_dir = os.path.splitext(video)[0] + '_cropped'
        files = [os.path.join(target_dir, item.name) for item in os.scandir(path=target_dir) if item.is_file() and re.search('.*\.jpg$', item.name)]
        train_files_crop.extend(files)


    # negative_sample_dir_path = args.dataset_path + "/negative/negative_cropped"
    # negative_samples = [os.path.join(negative_sample_dir_path, item.name) for item in os.scandir(path=negative_sample_dir_path) if item.is_file() and re.search('.jpg', item.name)]
    # train_files_crop.extend(negative_samples) # negative class

    train_files_withClassID_crop = sorted([f + " " + str(class_dir.index(c)) for c in class_dir for f in train_files_crop if re.search(c, f)])
    train_files_withClassID_orig = sorted(list(map(lambda x: re.sub(r'_cropped', '', x), train_files_withClassID_crop)))

    for f in train_files_withClassID_crop:
        assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])
    for f in train_files_withClassID_orig:
        assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])

    print("train_files_withClassID_crop", train_files_withClassID_crop)
    print("train_files_withClassID_orig", train_files_withClassID_orig)


    valididx = int(input('please select valid\'s index>'))

    # create valid set
    for video in luminous_cluster[valididx]:
        target_dir = os.path.splitext(video)[0] + '_cropped'
        files = [os.path.join(target_dir, item.name) for item in os.scandir(path=target_dir) if item.is_file() and re.search('^(?!.*\_d\d).*\.jpg$', item.name)]
        valid_files_crop.extend(files)

    # negative_sample_dir_path = args.dataset_path + "/negative/negative_cropped"
    # negative_samples = [os.path.join(negative_sample_dir_path, item.name) for item in os.scandir(path=negative_sample_dir_path) if item.is_file() and re.search('^(?!.*\_d\d).*\.jpg$', item.name)]
    # valid_files_crop.extend(negative_samples) # negative class

    valid_files_withClassID_crop = sorted([f + " " + str(class_dir.index(c)) for c in class_dir for f in valid_files_crop if re.search(c, f)])
    valid_files_withClassID_orig = sorted(list(map(lambda x: re.sub(r'_cropped', '', x), valid_files_withClassID_crop)))

    for f in valid_files_withClassID_crop:
        assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])
    for f in valid_files_withClassID_orig:
        assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])

    print("valid_files_withClassID_crop", valid_files_withClassID_crop)
    print("valid_files_withClassID_orig", valid_files_withClassID_orig)

    # create test set
    dellist = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]
    test_cluster = dellist(luminous_cluster, [trainidx, valididx])
    for other_cluster in test_cluster:
        for video in other_cluster:
            target_dir = os.path.splitext(video)[0] + '_cropped'
            files = [os.path.join(target_dir, item.name) for item in os.scandir(path=target_dir) if item.is_file() and re.search('^(?!.*\_d\d).*\.jpg$', item.name)]
            test_files_crop.extend(files)

    # negative_sample_dir_path = args.dataset_path + "/negative/negative_cropped"
    # negative_samples = [os.path.join(negative_sample_dir_path, item.name) for item in os.scandir(path=negative_sample_dir_path) if item.is_file() and re.search('^(?!.*\_d\d).*\.jpg$', item.name)]
    # test_files_crop.extend(negative_samples) # negative class

    test_files_withClassID_crop = sorted([f + " " + str(class_dir.index(c)) for c in class_dir for f in test_files_crop if re.search(c, f)])
    test_files_withClassID_orig = sorted(list(map(lambda x: re.sub(r'_cropped', '', x), test_files_withClassID_crop)))

    for f in test_files_withClassID_crop:
        assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])
    for f in test_files_withClassID_orig:
        assert os.path.exists(f.split(" ")[0]), "file:{0}".format(f.split(" ")[0])

    print("test_files_withClassID_crop", test_files_withClassID_crop)
    print("test_files_withClassID_orig", test_files_withClassID_orig)


    with open(args.dataset_path + "/" +"train_orig.txt", 'w') as f:
        for l in train_files_withClassID_orig:
            f.writelines(l + '\n')

    with open(args.dataset_path + "/" +"train_crop.txt", 'w') as f:
        for l in train_files_withClassID_crop:
            f.writelines(l + '\n')

    with open(args.dataset_path + "/" +"valid_orig.txt", 'w') as f:
        for l in valid_files_withClassID_orig:
            f.writelines(l + '\n')

    with open(args.dataset_path + "/" +"valid_crop.txt", 'w') as f:
        for l in valid_files_withClassID_crop:
            f.writelines(l + '\n')

    with open(args.dataset_path + "/" +"test_orig.txt", 'w') as f:
        for l in test_files_withClassID_orig:
            f.writelines(l + '\n')

    with open(args.dataset_path + "/" +"test_crop.txt", 'w') as f:
        for l in test_files_withClassID_crop:
            f.writelines(l + '\n')

    print("All process finished.")
