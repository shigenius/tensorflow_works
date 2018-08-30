import argparse

import os
import re

def replace_label(label_from, label_to, targettxt, out_suffix="_IDreplaced"):
    dict_from = {l.split(' ')[1].rstrip(): l.split(" ")[0] for l in open(label_from, "r")}
    dict_to = {l.split(' ')[1].rstrip(): l.split(" ")[0] for l in open(label_to, "r")}

    assert len(dict_from.keys()) == len(dict_to.keys())
    # convert_dict = {key:[dict_from[key], dict_to[key]] for key in dict_from.keys()}
    convert_dict = {dict_from[key]: dict_to[key] for key in dict_from.keys()}
    print("convert:")
    for key in sorted(convert_dict.keys()):
        print(key, "->", convert_dict[key])

    targetlines = [l.rstrip().split(' ') for l in open(targettxt, "r")]
    output = [l[0] + " " + convert_dict[l[1]] + "\n" for l in targetlines]

    # write
    dir_file_pair = os.path.split(targettxt) # 同じdirにtxtをつくる
    filename_ext_pair = os.path.splitext(dir_file_pair[1])
    output_path = os.path.join(dir_file_pair[0], filename_ext_pair[0] + out_suffix + filename_ext_pair[1])

    with open(output_path, mode='w') as f:
        f.writelines(output)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('label_from')
    parser.add_argument('label_to')
    parser.add_argument('target_txt')
    args = parser.parse_args()

    replace_label(args.label_from, args.label_to, args.target_txt)
    print("finished.")