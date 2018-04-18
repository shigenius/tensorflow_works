import os
import subprocess
import argparse
import re

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_path', type=str, help='full-path of the dataset')
    parser.add_argument('--batch_size', '-b', type=str, default='10', help='batch size')
    parser.add_argument('--train_step', '-s', type=str, default='3', help='num of train step')
    parser.add_argument('--protobuffer', '-pb', type=str, default='/Users/shigetomi/Downloads/imagenet/classify_image_graph_def.pb', help='path of protobuffer(pretrained model & graph)')
    parser.add_argument('--num_ofclass', '-nc', type=str, default='6', help='num of class')
    parser.add_argument('--log_name', '-ln', type=str, default='0418_roadsign', help='log name')
    args = parser.parse_args()

    dataset = args.dataset_path
    print("starting cross-validation by the dataset :" + dataset)

    files = [item.name for item in os.scandir(path=dataset) if item.is_file() and re.search('\d+_\d+_\d+.*\.txt', item.name)]

    # print(files)

    regex = r'\d+_\d+_\d+'
    pattern = re.compile(regex)
    days = sorted(list(set([re.match(pattern, file).group() for file in files])))
    days = [day.split('_') for day in days]

    print(len(days), days)

    for day in days:
        day_str = day[0] + '_' + day[1] + '_' + day[2]
        hoge = dataset + '/' + day_str
        # subprocess.check_call(['python', 'graphdef_test.py',
        #                        '--train1', hoge + 'train_crop.txt', # cropped train
        #                        '--test1', hoge + 'test_crop.txt',
        #                        '--train2', hoge + 'train_orig.txt', # original train
        #                        '--test2', hoge + 'test_orig.txt',
        #                        '-b', args.batch_size, '-s', args.train_step,
        #                        '-pb', args.protobuffer,
        #                        '-nc', args.num_ofclass,
        #                        '-save', './model/twostep.ckpt',
        #                        '-cb', day_str])

        command = 'python graphdef_test.py --train1 ' + hoge + 'train_crop.txt ' + '--test1 ' + hoge + 'test_crop.txt ' + '--train2 '+ hoge + 'train_orig.txt ' + '--test2 '+ hoge + 'test_orig.txt ' + '-b '+ args.batch_size+ ' -s '+ args.train_step + ' -pb '+ args.protobuffer + ' -nc ' + args.num_ofclass + ' -save '+ './model/twostep.ckpt ' + '-cb '+ day_str + ' > '+ 'log/'+args.log_name+day_str+'_log.txt'
        print(command)
        subprocess.call(command, shell=True)