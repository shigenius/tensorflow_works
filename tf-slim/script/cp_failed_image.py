import argparse
import csv
import shutil
import re
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('log_path', type=str, help='full-path of the log(csv)')
    parser.add_argument('output_dir', type=str, help='')
    args = parser.parse_args()

    with open(args.log_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)

        for row in reader:
            print(row)
            if re.search('/home/akalab/', row[0]):
                row[0] = row[0].replace('/home/akalab/', '/Users/shigetomi/Desktop/') # replace remote path to local path
            dirname, filename = os.path.split(row[0])
            wfilename = dirname.split('/')[-1] + '_' +filename

            if row[1] == 'FALSE' or 'False':
                print(dirname, wfilename)
                shutil.copy(row[0], args.output_dir + '/' + wfilename)