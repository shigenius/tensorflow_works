import csv
import argparse

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('csv_path', help='File name of csv')
    parser.add_argument('label_path', help='File name of label.txt')
    # parser.add_argument('num_class', help='num of class',
    #                     default=6)
    args = parser.parse_args()

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

    confusion_mat = [[0 for i in range(num_class)] for j in range(num_class)]
    for row in f:
        esti = int(row[2])
        gt = int(row[3])
        confusion_mat[gt][esti] += 1

    print("confision_mat:", confusion_mat)

    estimation_acc = [l[i]/sum(l) if l[i] != 0 else 0 for i, l in enumerate(confusion_mat)]
    print("estimation acc:", estimation_acc)

    tmp = sum([l[i] for i, l in enumerate(confusion_mat)])
    print(tmp)
    mean_acc = tmp / sum([sum(l) for i, l in enumerate(confusion_mat)]) if tmp != 0 else 0
    print("mean acc:", mean_acc)

    confusion_mat = [[round(n / sum(l), 2) if n != 0 else 0 for n in l] for l in confusion_mat]
    ndarr = np.array(confusion_mat)
    print(ndarr)

    plt.figure(figsize=(15, 8))
    sn.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.savefig('figure.png')
    plt.gca().xaxis.tick_top()
    plt.xticks(np.arange(num_class), labels, rotation=15)
    plt.yticks(np.arange(num_class), labels[::-1], rotation=0)
    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()

    plt.show()
