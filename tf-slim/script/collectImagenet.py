#
import sys,os
from urllib import urlopen
from urllib import urlretrieve
from urllib2 import URLError,HTTPError
import commands
import subprocess
import argparse
import random
from PIL import Image
import os.path
import re

def cmd(cmd):
    return commands.getoutput(cmd)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',        type=str,   default='./images')
parser.add_argument('--num_of_classes',  type=int,   default=1000)
parser.add_argument('--num_of_pics',   type=int,   default=10)
parser.add_argument('--offset',   type=int,   default=0)
parser.add_argument('--offset_class',   type=int,   default=-1)

args = parser.parse_args()

dict={}
for line in open('words.txt', 'r'):
    line=line.split('\t')
    head = line[0]
    name = line[1].split(',')[0]
    name = name.replace("\n", "")
    name = name.replace(" ", "-")
    dict[head]=name

_dict = {'person': 'n00007846', 'plant': 'n00017222', 'road': 'n00174003', 'traffic-light': 'n06874185', 'bicycle': 'n02834778', 'car': 'n02958343', 'office-building': 'n03842012', 'house': 'n07971449'}
_dict = {_dict[key]: key for key in _dict.keys()}
ids = [key for key in _dict.keys()]
#
# ids = open('imagenet.synset.obtain_synset_list', 'r').read()
# ids = ids.split()
# random.shuffle(ids)

offset_flag = True if args.offset != 0 else False

cmd("mkdir %s"%args.data_dir)
for i in range(args.num_of_classes):

    if i < args.offset_class:
        continue

    id = ids[i].rstrip()
    category = dict[id]
    if offset_flag == True:
        cnt = args.offset + 1
        offset_flag = False
    else:
        cnt = 0

    if len(category)>0:
        cmd("mkdir %s/%s"%(args.data_dir,category))
        print(category)
        try:
            print(id)
            urls=urlopen("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+id).read()

            urls=urls.split()
            random.shuffle(urls)
            j = 0
            while cnt<args.num_of_pics if args.num_of_pics<len(urls) else len(urls):

                url = urls[j]
                j+=1
                if j>=len(urls):
                    break
                print(cnt, url)

                filename = os.path.split(url)[1]
                try:
                    output = "%s/%s/%d_%s"%(args.data_dir,category,cnt,filename)
                    urlretrieve(url,output )
                    try:
                        img = Image.open(output)
                        size = os.path.getsize(output)
                        if size==2051: #flickr Error
                            cmd("rm %s"%output)
                            cnt-=1
                    except IOError:
                        cmd("rm %s"%output)
                        cnt-=1
                except HTTPError, e:
                    cnt-=1
                    print e.reason
                except URLError, e:
                    cnt-=1
                    print e.reason
                except IOError, e:
                    cnt-=1
                    print e
                cnt+=1
        except HTTPError, e:
            print e.reason
        except URLError, e:
            print e.reason
        except IOError, e:
            print e



#
# import sys
# import os
# # from urllib import request
# from PIL import Image
#
# import random
#
# def download(url, decode=False):
#     response = urlopen(url)
#     if response.geturl() == "https://s.yimg.com/pw/images/en-us/photo_unavailable.png":
#         # Flickr :This photo is no longer available iamge.
#         raise Exception("This photo is no longer available iamge.")
#
#     body = response.read()
#     if decode == True:
#         body = body.decode()
#     return body
#
# def write(path, img):
#     file = open(path, 'wb')
#     file.write(img)
#     file.close()
#
# def down_sampling_dict(dict, n):
#     keys = [key for key in dict.keys()]
#     sampledkeys = random.sample(keys, n)
#     return {key:dict[key] for key in sampledkeys}
#
# # # see http://image-net.org/archive/words.txt
# # classes = {"apple":"n07739125", "banana":"n07753592", "orange":"n07747607"}
#
# classes={}
# for line in open('words.txt', 'r'):
#     line=line.split('\t')
#     id = line[0]
#     name = line[1].split(',')[0]
#     name = name.replace("\n", "")
#     classes[name] = id
#
# num_of_classes = args.num_of_classes if len(classes) >= args.num_of_classes else len(classes)
# sampled_clasees = down_sampling_dict(classes, num_of_classes)
#
# for dir, id in sampled_clasees.items():
#     print(dir, id)
#     if not os.path.exists("./images/"+dir):
#         os.makedirs("./images/"+dir)
#     urls = download("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+id, decode=False).split()
#     num_of_pics = args.num_of_pics if len(urls) >= args.num_of_pics else len(urls)
#     sampled_urls = random.sample(urls, num_of_pics)
#     for i, url in enumerate(sampled_urls):
#         try:
#             print(url)
#             file = os.path.split(url)[1]
#             path = "./" + dir + "/" + file
#             temp = download(url)
#             write(path, temp)
#             print("done:" + str(i) + ":" + file)
#         except:
#             print("Unexpected error:", sys.exc_info()[0])
#             print("error:" + str(i) + ":" + file)
#
# print("end")