import os
import argparse
import re
from makeDataset_forFIT import get_luminous_condition_cluster
import csv

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_path', type=str, help='full-path of the dataset')
    parser.add_argument('csv_path', help='File name of csv')
    parser.add_argument('label_path', help='File name of label.txt')
    args = parser.parse_args()

    # get dict that shape is {day: video_paths}
    dataset_path = args.dataset_path
    print("dataset:", args.dataset_path)

    luminous_cluster = get_luminous_condition_cluster(dataset_path)

    for i, item in enumerate(luminous_cluster):
        print(i, item, "\n")


    label_file = open(args.label_path, "r", encoding="utf-8", errors="", newline="")

    num_class = max([int(row.split(' ')[0]) for row in label_file])
    num_class += 1
    print("num_class:", num_class)

    label_file = open(args.label_path, "r", encoding="utf-8", errors="", newline="")
    labels = [row.split(' ')[1].strip() for row in label_file]
    print(labels)


    csv_file = open(args.csv_path, "r", encoding="utf-8", errors="", newline="")
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                   skipinitialspace=True)
    header = next(f)


    n_corr = {}
    num = {}
    pattern = r'^\/[^\/]*\/[^\/]*\/'
    for row in f:
        if re.search('negative', row[0]):
            continue
        if re.search(pattern, row[0]):
            row[0] = re.sub(pattern, "/Users/shigetomi/Desktop/", row[0]) # replace remote path to local path
        row[0] = re.sub('_cropped.*$', '',  row[0])
        # print(row[0])
        row[0] = re.sub('\(|\)', '\(|\)', row[0])
        luminous = [[i, key, cluster[key]] for i, cluster in enumerate(luminous_cluster) for key in cluster.keys() if re.search(row[0], key)]
        flatten = lambda list: [e for inner_list in list for e in inner_list]
        luminous = flatten(luminous)

        # print(luminous)
        # print(luminous)
        if luminous[0] not in n_corr.keys():
            n_corr[luminous[0]] = 0
        if luminous[0] not in num.keys():
            num[luminous[0]] = 0

        num[luminous[0]] += 1
        if row[1] == 'True' or row[1] == 'TRUE':
            n_corr[luminous[0]] += 1

        #
        # esti = int(row[2])
        # gt = int(row[3])
        # if row[1] == 'TRUE' or 'True':
        #     # acc[] =
        #     pass

    print("corr", n_corr)
    print("num", num)
    acc = {key: round(n_corr[key]/num[key], 2) for key in n_corr.keys()}
    print("acc", acc)
    result = [[acc[key], luminous_cluster[key]] for key in acc.keys()]
    for item in result:
        print("acc:", item[0], "targets:", item[1], "\n")

    print("mean acc:", sum(n_corr.values())/sum(num.values()))


