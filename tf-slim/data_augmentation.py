import argparse
import re
import cv2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TwoInputDistortion import Distortion

#
# 指定したdatasetをaugmentして各ディレクトリに保存する．
#

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-dataset', type=str, default='/Users/shigetomi/Desktop/dataset_roadsign', help='full-path of the dataset')
    parser.add_argument('-r', type=int, default=4, help='rate of extension')
    args = parser.parse_args()

    dist = Distortion(gamma=1.5)

    dataset_path = args.dataset
    rate = args.r
    print("starting Data Augmentation for the dataset :" + dataset_path)

    tree = {c.name:
                {v.name:
                    {i.name for i in os.scandir(path=dataset_path+"/"+c.name+"/"+v.name) if i.is_file() and re.search('.jpg', i.name)}
                for v in os.scandir(path=dataset_path + "/" + c.name) if v.is_dir()}
            for c in os.scandir(path=dataset_path) if c.is_dir()}
    print(tree)

    for r in range(rate-1):
        for c in tree:
            for v in tree[c]:
                if re.search('_cropped', v):
                    for i in tree[c][v]:
                        # check absence
                        crop_path = dataset_path + '/' + c + '/' + v + '/' + i
                        orig_path = re.sub(r'_cropped', '', crop_path)
                        assert os.path.exists(orig_path)
                        assert os.path.exists(crop_path)

                        cropped = cv2.imread(crop_path)
                        original = cv2.imread(orig_path)
                        dst_o, dst_c = dist.distort(images=[original, cropped], flag='train', p=1.0)
                        # cv2.imshow("c", dst_c)
                        # cv2.imshow("o", dst_o)
                        # cv2.waitKey(0)

                        save_path_c = os.path.splitext(crop_path)[0] + '_d' + str(r) + os.path.splitext(crop_path)[1]
                        save_path_o = os.path.splitext(orig_path)[0] + '_d' + str(r) + os.path.splitext(orig_path)[1]
                        print(save_path_c)
                        print(save_path_o)

                        cv2.imwrite(save_path_c, dst_c)
                        cv2.imwrite(save_path_o, dst_o)
