import cv2
import argparse

import os
import re

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def resize_all(target_dir):
    target_dir_path = args.target

    pattern = r"\.(jpg|png|jpeg)$"
    target_files = [f for f in find_all_files(target_dir_path) if re.search(pattern, f)]

    # for f in files:
    #     print(f)
    # target_files = sorted([os.path.join(target_dir_path, file.name) for file in os.scandir(path=target_dir_path) if
    #                        file.is_file() and re.search(pattern, file.name)])

    for path in target_files:
        print(path)
        image = cv2.imread(path)
        resized = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))

        cv2.imwrite(path, resized)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('target', help='Path to images directory')
    args = parser.parse_args()

    resize_all(args.target)