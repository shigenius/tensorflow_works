import argparse
import os
import re

import resize_all_images_by_half
from pathlib import Path
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('target', help='Path to target directory')
    args = parser.parse_args()


    pattern = r"\.(mp4|m4v|mov|MOV)$"
    all_video_path = [f for f in resize_all_images_by_half.find_all_files(args.target) if re.search(pattern, os.path.split(f)[1])]

    dir_ext_pair = [os.path.splitext(f) for f in all_video_path]
    dir_cropped = [i[0]+"_cropped" for i in dir_ext_pair]

    # make dirs to save images
    [os.mkdir(i[0]) for i in dir_ext_pair if not os.path.exists(i[0])]
    [os.mkdir(i) for i in dir_cropped if not os.path.exists(i)]

    p_list = [Path(i[0]) for i in dir_ext_pair]
    must_be_sampling_list = [str(p) for p in p_list if len([i for i in p.glob("*.jpg")]) == 0]
    target_dir_ext_pair = [i for i in dir_ext_pair if i[0] in must_be_sampling_list]
    print("video to images:")
    for i in target_dir_ext_pair:
        print(i)

    for i in target_dir_ext_pair:
        target = i[0] +  i[1]
        save_dir = i[0]
        print("\n"+ target)
        subprocess.check_call(['ffmpeg', '-i',
                               target,
                               '-r', '5',
                               os.path.join(save_dir, 'image_%04d.jpg')])
        # ffmpeg -i $label/$file -r 5 ./$label/$dir/image_%04d.jpg

    print("finished")

